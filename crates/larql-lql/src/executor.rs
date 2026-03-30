/// LQL Executor — dispatches parsed AST statements to backend operations.
///
/// Manages session state (active vindex/model) and formats output.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::ast::*;
use crate::error::LqlError;

/// The active backend for the session.
enum Backend {
    /// Pre-extracted vindex — fast, supports mutation.
    Vindex {
        path: PathBuf,
        config: larql_inference::VindexConfig,
        index: larql_inference::VectorIndex,
    },
    /// No backend loaded.
    None,
}

/// Session state for the REPL / batch executor.
pub struct Session {
    backend: Backend,
}

impl Session {
    pub fn new() -> Self {
        Self {
            backend: Backend::None,
        }
    }

    /// Execute a single LQL statement, returning formatted output lines.
    pub fn execute(&mut self, stmt: &Statement) -> Result<Vec<String>, LqlError> {
        match stmt {
            Statement::Pipe { left, right } => {
                let mut out = self.execute(left)?;
                out.extend(self.execute(right)?);
                Ok(out)
            }
            Statement::Use { target } => self.exec_use(target),
            Statement::Stats { vindex } => self.exec_stats(vindex.as_deref()),
            Statement::Walk { prompt, top, layers, mode, compare } => {
                self.exec_walk(prompt, *top, layers.as_ref(), *mode, *compare)
            }
            Statement::Describe { entity, layer, relations_only } => {
                self.exec_describe(entity, *layer, *relations_only)
            }
            Statement::Select { fields, conditions, nearest, order, limit } => {
                self.exec_select(fields, conditions, nearest.as_ref(), order.as_ref(), *limit)
            }
            Statement::Explain { prompt, layers, verbose } => {
                self.exec_explain(prompt, layers.as_ref(), *verbose)
            }
            Statement::ShowRelations { layer, with_examples } => {
                self.exec_show_relations(*layer, *with_examples)
            }
            Statement::ShowLayers { range } => self.exec_show_layers(range.as_ref()),
            Statement::ShowFeatures { layer, conditions, limit } => {
                self.exec_show_features(*layer, conditions, *limit)
            }
            Statement::ShowModels => self.exec_show_models(),
            Statement::Extract { model, output, components, layers } => {
                self.exec_extract(model, output, components.as_deref(), layers.as_ref())
            }
            Statement::Compile { vindex, output, format } => {
                self.exec_compile(vindex, output, *format)
            }
            Statement::Diff { a, b, layer, relation, limit } => {
                self.exec_diff(a, b, *layer, relation.as_deref(), *limit)
            }
            Statement::Insert { entity, relation, target, layer, confidence } => {
                self.exec_insert(entity, relation, target, *layer, *confidence)
            }
            Statement::Infer { prompt, top, compare } => {
                self.exec_infer(prompt, *top, *compare)
            }
            Statement::Delete { conditions } => self.exec_delete(conditions),
            Statement::Update { set, conditions } => self.exec_update(set, conditions),
            Statement::Merge { source, target, conflict } => {
                self.exec_merge(source, target.as_deref(), *conflict)
            }
        }
    }

    // ── USE ──

    fn exec_use(&mut self, target: &UseTarget) -> Result<Vec<String>, LqlError> {
        match target {
            UseTarget::Vindex(path) => {
                let path = PathBuf::from(path);
                if !path.exists() {
                    return Err(LqlError::Execution(format!(
                        "vindex not found: {}",
                        path.display()
                    )));
                }

                let config = larql_inference::load_vindex_config(&path)
                    .map_err(|e| LqlError::Execution(format!("failed to load vindex config: {e}")))?;

                let mut cb = larql_inference::vector_index::SilentLoadCallbacks;
                let index = larql_inference::VectorIndex::load_vindex(&path, &mut cb)
                    .map_err(|e| LqlError::Execution(format!("failed to load vindex: {e}")))?;

                let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
                let out = vec![format!(
                    "Using: {} ({} layers, {} features, model: {})",
                    path.display(),
                    config.num_layers,
                    format_number(total_features),
                    config.model,
                )];

                self.backend = Backend::Vindex { path, config, index };
                Ok(out)
            }
            UseTarget::Model { id, auto_extract } => {
                // Direct model access is not yet implemented — nudge toward vindex
                let mut out = vec![format!(
                    "Direct model access not yet implemented. Extract first:"
                )];
                out.push(format!(
                    "  EXTRACT MODEL \"{}\" INTO \"{}.vindex\";",
                    id,
                    id.split('/').last().unwrap_or(id)
                ));
                if *auto_extract {
                    out.push("  (AUTO_EXTRACT noted — will be supported in a future version)".into());
                }
                Ok(out)
            }
        }
    }

    // ── STATS ──

    fn exec_stats(&self, _vindex_path: Option<&str>) -> Result<Vec<String>, LqlError> {
        match &self.backend {
            Backend::Vindex { path, config, index } => {
                let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
                let gate_size = index.total_gate_vectors();
                let down_size = index.total_down_meta();

                let file_size = dir_size(&path);

                let mut out = Vec::new();
                out.push(format!("Model:           {}", config.model));
                out.push(format!("Family:          {}", config.family));
                out.push(format!("Layers:          {}", config.num_layers));
                out.push(format!("Hidden size:     {}", config.hidden_size));
                out.push(format!("Intermediate:    {}", config.intermediate_size));
                out.push(format!("Vocab size:      {}", config.vocab_size));
                out.push(format!("Features/layer:  {}", config.intermediate_size));
                out.push(format!("Total features:  {}", format_number(total_features)));
                out.push(format!("Gate vectors:    {}", format_number(gate_size)));
                out.push(format!("Down metadata:   {}", format_number(down_size)));
                out.push(format!("Index size:      {}", format_bytes(file_size)));
                out.push(format!("Embed scale:     {}", config.embed_scale));
                out.push(format!("Path:            {}", path.display()));
                Ok(out)
            }
            Backend::None => Err(LqlError::NoBackend),
        }
    }

    // ── WALK ──
    //
    // Pure vindex feature scan. No attention. Shows what gate features fire
    // for the last token's embedding. This is a knowledge browser, not inference.

    fn exec_walk(
        &self,
        prompt: &str,
        top: Option<u32>,
        layers: Option<&Range>,
        _mode: Option<WalkMode>,
        _compare: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (path, _config, index) = self.require_vindex()?;
        let top_k = top.unwrap_or(10) as usize;

        let (embed, embed_scale) = larql_inference::load_vindex_embeddings(path)
            .map_err(|e| LqlError::Execution(format!("failed to load embeddings: {e}")))?;
        let tokenizer = larql_inference::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::Execution(format!("failed to load tokenizer: {e}")))?;

        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::Execution(format!("tokenize error: {e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        if token_ids.is_empty() {
            return Err(LqlError::Execution("empty prompt".into()));
        }

        let last_tok = *token_ids.last().unwrap();
        let token_str = tokenizer
            .decode(&[last_tok], true)
            .unwrap_or_else(|_| format!("T{last_tok}"));

        let embed_row = embed.row(last_tok as usize);
        let query: larql_inference::ndarray::Array1<f32> =
            embed_row.mapv(|v| v * embed_scale);

        let all_layers = index.loaded_layers();
        let walk_layers: Vec<usize> = if let Some(range) = layers {
            (range.start as usize..=range.end as usize)
                .filter(|l| all_layers.contains(l))
                .collect()
        } else {
            all_layers
        };

        let start = std::time::Instant::now();
        let trace = index.walk(&query, &walk_layers, top_k);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut out = Vec::new();
        out.push(format!(
            "Feature scan for {:?} (token {:?}, {} layers)",
            prompt,
            token_str.trim(),
            walk_layers.len(),
        ));
        out.push(String::new());

        // Show top features per layer — only layers with meaningful hits
        for (layer, hits) in &trace.layers {
            if hits.is_empty() {
                continue;
            }
            for hit in hits.iter().take(3) {
                let down_top: String = hit
                    .meta
                    .top_k
                    .iter()
                    .take(3)
                    .map(|t| t.token.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push(format!(
                    "  L{:2}: F{:<5} gate={:+.1}  top={:15}  down=[{}]",
                    layer, hit.feature, hit.gate_score,
                    format!("{:?}", hit.meta.top_token), down_top,
                ));
            }
        }

        out.push(format!("\n{:.1}ms", elapsed_ms));
        out.push(String::new());
        out.push("Note: pure vindex scan (no attention). For inference use INFER.".into());

        Ok(out)
    }

    // ── INFER ──
    //
    // Full forward pass with attention. Requires model weights — either from
    // a model_weights.bin inside the vindex, or a model loaded via USE MODEL.

    fn exec_infer(
        &mut self,
        prompt: &str,
        top: Option<u32>,
        compare: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (path, config, index) = self.require_vindex()?;
        let top_k = top.unwrap_or(5) as usize;

        // Try to load model weights from vindex
        if !config.has_model_weights {
            return Err(LqlError::Execution(format!(
                "INFER requires model weights. This vindex was built without --include-weights.\n\
                 Options:\n\
                 1. Rebuild: larql extract-index --model \"{}\" --output \"{}\" --include-weights\n\
                 2. Use the CLI: larql walk --index \"{}\" --predict --model \"{}\"",
                config.model,
                path.display(),
                path.display(),
                config.model,
            )));
        }

        let mut cb = larql_inference::vector_index::SilentLoadCallbacks;
        let weights = larql_inference::load_model_weights_from_vindex(path, &mut cb)
            .map_err(|e| LqlError::Execution(format!("failed to load model weights: {e}")))?;
        let tokenizer = larql_inference::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::Execution(format!("failed to load tokenizer: {e}")))?;

        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::Execution(format!("tokenize error: {e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        // Walk FFN: full forward pass with attention, but swap dense FFN for vindex KNN
        let walk_ffn = larql_inference::vector_index::WalkFfn::new(&weights, index, 10);
        let start = std::time::Instant::now();
        let result = larql_inference::predict_with_ffn(
            &weights,
            &tokenizer,
            &token_ids,
            top_k,
            &walk_ffn,
        );
        let walk_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut out = Vec::new();
        out.push("Predictions (walk FFN):".into());
        for (i, (tok, prob)) in result.predictions.iter().enumerate() {
            out.push(format!(
                "  {:2}. {:20} ({:.2}%)",
                i + 1,
                tok,
                prob * 100.0
            ));
        }
        out.push(format!("  {:.0}ms", walk_ms));

        if compare {
            // Dense ground truth
            let start = std::time::Instant::now();
            let dense = larql_inference::predict(&weights, &tokenizer, &token_ids, top_k);
            let dense_ms = start.elapsed().as_secs_f64() * 1000.0;

            out.push(String::new());
            out.push("Predictions (dense):".into());
            for (i, (tok, prob)) in dense.predictions.iter().enumerate() {
                out.push(format!(
                    "  {:2}. {:20} ({:.2}%)",
                    i + 1,
                    tok,
                    prob * 100.0
                ));
            }
            out.push(format!("  {:.0}ms", dense_ms));
        }

        Ok(out)
    }

    // ── DESCRIBE ──
    //
    // Pure vindex knowledge browser. Shows what the model "knows" about an
    // entity by finding which gate features fire for its embedding.
    // Filters noise and groups by layer band (early=morphological, mid=semantic, late=output).

    fn exec_describe(
        &self,
        entity: &str,
        layer: Option<u32>,
        _relations_only: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (path, _config, index) = self.require_vindex()?;

        let (embed, embed_scale) = larql_inference::load_vindex_embeddings(path)
            .map_err(|e| LqlError::Execution(format!("failed to load embeddings: {e}")))?;
        let tokenizer = larql_inference::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::Execution(format!("failed to load tokenizer: {e}")))?;

        let encoding = tokenizer
            .encode(entity, false)
            .map_err(|e| LqlError::Execution(format!("tokenize error: {e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        if token_ids.is_empty() {
            return Ok(vec![format!("{entity}\n  (not found)")]);
        }

        let last_tok = *token_ids.last().unwrap();
        let embed_row = embed.row(last_tok as usize);
        let query: larql_inference::ndarray::Array1<f32> =
            embed_row.mapv(|v| v * embed_scale);

        // If AT LAYER specified, only scan that layer
        let all_layers = index.loaded_layers();
        let scan_layers: Vec<usize> = if let Some(l) = layer {
            vec![l as usize]
        } else {
            all_layers
        };

        let trace = index.walk(&query, &scan_layers, 20);

        // Collect all down-projection outputs, filtering noise:
        // - Only keep features with positive gate scores (entity activates the gate)
        // - Only keep tokens that are mostly ASCII printable (filter encoding garbage)
        // - Deduplicate case-insensitive
        let mut token_data: HashMap<String, (f32, Vec<usize>, usize)> = HashMap::new();

        for (layer_idx, hits) in &trace.layers {
            for hit in hits {
                if hit.gate_score <= 0.0 {
                    continue;
                }
                let tok = &hit.meta.top_token;
                // Filter: skip tokens that are mostly non-Latin (heuristic for noise)
                if !is_readable_token(tok) {
                    continue;
                }
                let key = tok.to_lowercase();
                let entry = token_data
                    .entry(key)
                    .or_insert((0.0f32, Vec::new(), 0));
                if hit.gate_score > entry.0 {
                    entry.0 = hit.gate_score;
                }
                entry.1.push(*layer_idx);
                entry.2 += 1;
            }
        }

        let mut sorted: Vec<(&str, f32, usize, usize, usize)> = token_data
            .iter()
            .map(|(tok, (score, layers, count))| {
                let min_l = *layers.iter().min().unwrap_or(&0);
                let max_l = *layers.iter().max().unwrap_or(&0);
                (tok.as_str(), *score, min_l, max_l, *count)
            })
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = vec![entity.to_string()];

        if sorted.is_empty() {
            out.push("  (no features found)".into());
            return Ok(out);
        }

        // Group into layer bands
        let mut early = Vec::new();  // L0-13: morphological/syntactic
        let mut mid = Vec::new();    // L14-27: semantic/factual
        let mut late = Vec::new();   // L28-33: output/formatting

        for &(tok, score, min_l, max_l, count) in sorted.iter().take(30) {
            let entry = format!(
                "    {:20} score={:.1}  L{}-{}  ({}x)",
                tok, score, min_l, max_l, count
            );
            if max_l <= 13 {
                early.push(entry);
            } else if min_l >= 28 {
                late.push(entry);
            } else {
                mid.push(entry);
            }
        }

        if !mid.is_empty() {
            out.push("  Knowledge (L14-27):".into());
            for e in mid.iter().take(10) {
                out.push(e.clone());
            }
        }
        if !late.is_empty() {
            out.push("  Output (L28-33):".into());
            for e in late.iter().take(5) {
                out.push(e.clone());
            }
        }
        if !early.is_empty() {
            out.push("  Morphological (L0-13):".into());
            for e in early.iter().take(5) {
                out.push(e.clone());
            }
        }

        Ok(out)
    }

    // ── SELECT ──

    fn exec_select(
        &self,
        _fields: &[Field],
        conditions: &[Condition],
        _nearest: Option<&NearestClause>,
        order: Option<&OrderBy>,
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let (_path, _config, index) = self.require_vindex()?;

        // For vindex, SELECT works by scanning down_meta for matching features
        let all_layers = index.loaded_layers();
        let limit = limit.unwrap_or(20) as usize;

        // Extract filter conditions
        let entity_filter = conditions.iter().find(|c| c.field == "entity").and_then(|c| {
            if let Value::String(ref s) = c.value { Some(s.as_str()) } else { None }
        });
        let _relation_filter = conditions.iter().find(|c| c.field == "relation").and_then(|c| {
            if let Value::String(ref s) = c.value { Some(s.as_str()) } else { None }
        });
        let layer_filter = conditions.iter().find(|c| c.field == "layer").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });
        let feature_filter = conditions.iter().find(|c| c.field == "feature").and_then(|c| {
            if let Value::Integer(n) = c.value { Some(n as usize) } else { None }
        });

        struct Row {
            layer: usize,
            feature: usize,
            top_token: String,
            c_score: f32,
        }

        let mut rows: Vec<Row> = Vec::new();

        let scan_layers: Vec<usize> = if let Some(l) = layer_filter {
            vec![l]
        } else {
            all_layers
        };

        for layer in &scan_layers {
            if let Some(metas) = index.down_meta_at(*layer) {
                for (feat_idx, meta_opt) in metas.iter().enumerate() {
                    if let Some(feature_f) = feature_filter {
                        if feat_idx != feature_f {
                            continue;
                        }
                    }
                    if let Some(meta) = meta_opt {
                        // Filter by entity (match against top_token as proxy)
                        if let Some(ent) = entity_filter {
                            if !meta.top_token.to_lowercase().contains(&ent.to_lowercase()) {
                                continue;
                            }
                        }
                        rows.push(Row {
                            layer: *layer,
                            feature: feat_idx,
                            top_token: meta.top_token.clone(),
                            c_score: meta.c_score,
                        });
                    }
                }
            }
        }

        // Sort
        if let Some(ord) = order {
            match ord.field.as_str() {
                "confidence" | "c_score" => {
                    rows.sort_by(|a, b| {
                        let cmp = a.c_score.partial_cmp(&b.c_score).unwrap_or(std::cmp::Ordering::Equal);
                        if ord.descending { cmp.reverse() } else { cmp }
                    });
                }
                "layer" => {
                    rows.sort_by(|a, b| {
                        let cmp = a.layer.cmp(&b.layer);
                        if ord.descending { cmp.reverse() } else { cmp }
                    });
                }
                _ => {}
            }
        }

        rows.truncate(limit);

        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:<8} {:<20} {:>10}",
            "Layer", "Feature", "Token", "Score"
        ));
        out.push("-".repeat(50));

        for row in &rows {
            out.push(format!(
                "L{:<7} F{:<7} {:20} {:>10.4}",
                row.layer, row.feature, row.top_token, row.c_score
            ));
        }

        if rows.is_empty() {
            out.push("  (no matching edges)".into());
        }

        Ok(out)
    }

    // ── EXPLAIN ──

    fn exec_explain(
        &self,
        prompt: &str,
        layers: Option<&Range>,
        _verbose: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (path, _config, index) = self.require_vindex()?;

        let (embed, embed_scale) = larql_inference::load_vindex_embeddings(path)
            .map_err(|e| LqlError::Execution(format!("failed to load embeddings: {e}")))?;
        let tokenizer = larql_inference::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::Execution(format!("failed to load tokenizer: {e}")))?;

        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::Execution(format!("tokenize error: {e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        if token_ids.is_empty() {
            return Err(LqlError::Execution("empty prompt".into()));
        }

        let last_tok = *token_ids.last().unwrap();
        let embed_row = embed.row(last_tok as usize);
        let query: larql_inference::ndarray::Array1<f32> =
            embed_row.mapv(|v| v * embed_scale);

        let all_layers = index.loaded_layers();
        let walk_layers: Vec<usize> = if let Some(range) = layers {
            (range.start as usize..=range.end as usize)
                .filter(|l| all_layers.contains(l))
                .collect()
        } else {
            all_layers
        };

        let trace = index.walk(&query, &walk_layers, 5);

        let mut out = Vec::new();
        for (layer, hits) in &trace.layers {
            for hit in hits {
                let down_tokens: String = hit
                    .meta
                    .top_k
                    .iter()
                    .take(3)
                    .map(|t| t.token.clone())
                    .collect::<Vec<_>>()
                    .join(", ");

                out.push(format!(
                    "L{}: F{} → {} (gate={:.1}, down=[{}])",
                    layer, hit.feature, hit.meta.top_token, hit.gate_score, down_tokens
                ));
            }
        }

        Ok(out)
    }

    // ── SHOW ──

    fn exec_show_relations(
        &self,
        layer_filter: Option<u32>,
        _with_examples: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (_path, _config, index) = self.require_vindex()?;

        // Only scan knowledge layers (L14-27) by default — that's where
        // factual relations live. Early layers are morphological noise,
        // late layers are output formatting.
        let all_layers = index.loaded_layers();
        let scan_layers: Vec<usize> = if let Some(l) = layer_filter {
            vec![l as usize]
        } else {
            all_layers.iter().copied().filter(|l| *l >= 14 && *l <= 27).collect()
        };

        // Collect readable tokens with high c_score (confident down projections)
        struct TokenInfo {
            count: usize,
            max_score: f32,
            min_layer: usize,
            max_layer: usize,
        }

        let mut tokens: HashMap<String, TokenInfo> = HashMap::new();

        for &layer in &scan_layers {
            if let Some(metas) = index.down_meta_at(layer) {
                for meta_opt in metas.iter() {
                    if let Some(meta) = meta_opt {
                        // Filter: readable tokens only, minimum confidence
                        if !is_readable_token(&meta.top_token) {
                            continue;
                        }
                        if meta.c_score < 0.1 {
                            continue;
                        }
                        let key = meta.top_token.to_lowercase();
                        let entry = tokens.entry(key).or_insert(TokenInfo {
                            count: 0,
                            max_score: 0.0,
                            min_layer: layer,
                            max_layer: layer,
                        });
                        entry.count += 1;
                        if meta.c_score > entry.max_score {
                            entry.max_score = meta.c_score;
                        }
                        if layer < entry.min_layer {
                            entry.min_layer = layer;
                        }
                        if layer > entry.max_layer {
                            entry.max_layer = layer;
                        }
                    }
                }
            }
        }

        // Sort by count (features that output this token)
        let mut sorted: Vec<(String, &TokenInfo)> = tokens.iter()
            .map(|(tok, info)| (tok.clone(), info))
            .collect();
        sorted.sort_by(|a, b| b.1.count.cmp(&a.1.count));
        sorted.truncate(30);

        let mut out = Vec::new();
        let layer_label = if layer_filter.is_some() {
            format!("L{}", layer_filter.unwrap())
        } else {
            "L14-27".into()
        };
        out.push(format!(
            "{:<25} {:>8} {:>8} {:>10}",
            format!("Token ({})", layer_label), "Count", "Score", "Layers"
        ));
        out.push("-".repeat(55));

        for (tok, info) in &sorted {
            out.push(format!(
                "{:<25} {:>8} {:>8.2} {:>5}-{}",
                tok,
                info.count,
                info.max_score,
                info.min_layer,
                info.max_layer,
            ));
        }

        if sorted.is_empty() {
            out.push("  (no readable features found)".into());
        }

        Ok(out)
    }

    fn exec_show_layers(&self, range: Option<&Range>) -> Result<Vec<String>, LqlError> {
        let (_path, _config, index) = self.require_vindex()?;

        let all_layers = index.loaded_layers();
        let show_layers: Vec<usize> = if let Some(r) = range {
            (r.start as usize..=r.end as usize)
                .filter(|l| all_layers.contains(l))
                .collect()
        } else {
            all_layers
        };

        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:>10} {:>10} {:>15}",
            "Layer", "Features", "With Meta", "Top Token"
        ));
        out.push("-".repeat(48));

        for layer in &show_layers {
            let gate_count = index
                .gate_vectors_at(*layer)
                .map(|m| m.shape()[0])
                .unwrap_or(0);
            let (meta_count, top_tok) = if let Some(metas) = index.down_meta_at(*layer) {
                let count = metas.iter().filter(|m| m.is_some()).count();
                // Find most common top_token at this layer
                let mut freq: HashMap<&str, usize> = HashMap::new();
                for m in metas.iter().flatten() {
                    *freq.entry(&m.top_token).or_default() += 1;
                }
                let top = freq
                    .into_iter()
                    .max_by_key(|(_, c)| *c)
                    .map(|(t, _)| t.to_string())
                    .unwrap_or_default();
                (count, top)
            } else {
                (0, String::new())
            };

            out.push(format!(
                "L{:<7} {:>10} {:>10} {:>15}",
                layer,
                format_number(gate_count),
                format_number(meta_count),
                top_tok
            ));
        }

        Ok(out)
    }

    fn exec_show_features(
        &self,
        layer: u32,
        _conditions: &[Condition],
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let (_path, _config, index) = self.require_vindex()?;
        let limit = limit.unwrap_or(20) as usize;

        let metas = index
            .down_meta_at(layer as usize)
            .ok_or_else(|| LqlError::Execution(format!("no metadata for layer {layer}")))?;

        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:<20} {:>10} {:>30}",
            "Feature", "Top Token", "Score", "Down outputs"
        ));
        out.push("-".repeat(72));

        let mut count = 0;
        for (feat_idx, meta_opt) in metas.iter().enumerate() {
            if count >= limit {
                break;
            }
            if let Some(meta) = meta_opt {
                let down_tokens: String = meta
                    .top_k
                    .iter()
                    .take(5)
                    .map(|t| format!("{}", t.token))
                    .collect::<Vec<_>>()
                    .join(", ");

                out.push(format!(
                    "F{:<7} {:<20} {:>10.4} {:>30}",
                    feat_idx, meta.top_token, meta.c_score, down_tokens
                ));
                count += 1;
            }
        }

        Ok(out)
    }

    fn exec_show_models(&self) -> Result<Vec<String>, LqlError> {
        // List .vindex directories in the current directory
        let mut out = Vec::new();
        out.push(format!(
            "{:<35} {:>10} {:>8} {:>12}",
            "Model", "Size", "Layers", "Status"
        ));
        out.push("-".repeat(70));

        let cwd = std::env::current_dir().unwrap_or_default();
        if let Ok(entries) = std::fs::read_dir(&cwd) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let index_json = path.join("index.json");
                    if index_json.exists() {
                        if let Ok(config) = larql_inference::load_vindex_config(&path) {
                            let size = dir_size(&path);
                            out.push(format!(
                                "{:<35} {:>10} {:>8} {:>12}",
                                path.file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy(),
                                format_bytes(size),
                                config.num_layers,
                                "ready",
                            ));
                        }
                    }
                }
            }
        }

        if out.len() == 2 {
            out.push("  (no vindexes found in current directory)".into());
        }

        Ok(out)
    }

    // ── EXTRACT (stub — wires to existing extract-index) ──

    fn exec_extract(
        &mut self,
        model: &str,
        output: &str,
        _components: Option<&[Component]>,
        _layers: Option<&Range>,
    ) -> Result<Vec<String>, LqlError> {
        Err(LqlError::NotImplemented(
            "EXTRACT".into(),
            format!(
                "Use the CLI: larql extract-index --model \"{}\" --output \"{}\"",
                model, output
            ),
        ))
    }

    // ── COMPILE (stub) ──

    fn exec_compile(
        &self,
        _vindex: &VindexRef,
        _output: &str,
        _format: Option<OutputFormat>,
    ) -> Result<Vec<String>, LqlError> {
        Err(LqlError::NotImplemented(
            "COMPILE".into(),
            "vindex → safetensors recompilation is not yet implemented".into(),
        ))
    }

    // ── DIFF (stub) ──

    fn exec_diff(
        &self,
        _a: &VindexRef,
        _b: &VindexRef,
        _layer: Option<u32>,
        _relation: Option<&str>,
        _limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        Err(LqlError::NotImplemented(
            "DIFF".into(),
            "vindex comparison is not yet implemented".into(),
        ))
    }

    // ── INSERT (stub) ──

    fn exec_insert(
        &mut self,
        _entity: &str,
        _relation: &str,
        _target: &str,
        _layer: Option<u32>,
        _confidence: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        match &self.backend {
            Backend::Vindex { .. } => {
                Err(LqlError::NotImplemented(
                    "INSERT".into(),
                    "edge → vector synthesis is not yet implemented".into(),
                ))
            }
            Backend::None => Err(LqlError::NoBackend),
        }
    }

    // ── DELETE (stub) ──

    fn exec_delete(&mut self, _conditions: &[Condition]) -> Result<Vec<String>, LqlError> {
        Err(LqlError::NotImplemented(
            "DELETE".into(),
            "edge deletion is not yet implemented".into(),
        ))
    }

    // ── UPDATE (stub) ──

    fn exec_update(
        &mut self,
        _set: &[Assignment],
        _conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        Err(LqlError::NotImplemented(
            "UPDATE".into(),
            "edge update is not yet implemented".into(),
        ))
    }

    // ── MERGE (stub) ──

    fn exec_merge(
        &mut self,
        _source: &str,
        _target: Option<&str>,
        _conflict: Option<ConflictStrategy>,
    ) -> Result<Vec<String>, LqlError> {
        Err(LqlError::NotImplemented(
            "MERGE".into(),
            "vindex merge is not yet implemented".into(),
        ))
    }

    // ── Helpers ──

    fn require_vindex(
        &self,
    ) -> Result<(&Path, &larql_inference::VindexConfig, &larql_inference::VectorIndex), LqlError>
    {
        match &self.backend {
            Backend::Vindex {
                path,
                config,
                index,
            } => Ok((path, config, index)),
            Backend::None => Err(LqlError::NoBackend),
        }
    }
}

/// Get total size of a directory in bytes.
fn dir_size(path: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{n}")
    }
}

/// Heuristic: is a token readable enough to show to the user?
/// Filters out encoding garbage, isolated combining marks, etc.
fn is_readable_token(tok: &str) -> bool {
    let tok = tok.trim();
    if tok.is_empty() || tok.len() > 30 {
        return false;
    }
    // Count characters that are Latin, digit, common punctuation, or space
    let readable = tok.chars().filter(|c| {
        c.is_ascii_alphanumeric()
            || *c == ' '
            || *c == '-'
            || *c == '\''
            || *c == '.'
            || *c == ','
    }).count();
    let total = tok.chars().count();
    // At least half the characters should be readable
    readable * 2 >= total && total > 0
}

fn format_bytes(b: u64) -> String {
    if b >= 1_073_741_824 {
        format!("{:.2} GB", b as f64 / 1_073_741_824.0)
    } else if b >= 1_048_576 {
        format!("{:.1} MB", b as f64 / 1_048_576.0)
    } else if b >= 1024 {
        format!("{:.1} KB", b as f64 / 1024.0)
    } else {
        format!("{b} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser;

    // ── Session state: no backend ──

    #[test]
    fn no_backend_stats() {
        let mut session = Session::new();
        let stmt = parser::parse("STATS;").unwrap();
        let result = session.execute(&stmt);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, LqlError::NoBackend));
    }

    #[test]
    fn no_backend_walk() {
        let mut session = Session::new();
        let stmt = parser::parse(r#"WALK "test" TOP 5;"#).unwrap();
        let result = session.execute(&stmt);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LqlError::NoBackend));
    }

    #[test]
    fn no_backend_describe() {
        let mut session = Session::new();
        let stmt = parser::parse(r#"DESCRIBE "France";"#).unwrap();
        assert!(matches!(
            session.execute(&stmt).unwrap_err(),
            LqlError::NoBackend
        ));
    }

    #[test]
    fn no_backend_select() {
        let mut session = Session::new();
        let stmt = parser::parse("SELECT * FROM EDGES;").unwrap();
        assert!(matches!(
            session.execute(&stmt).unwrap_err(),
            LqlError::NoBackend
        ));
    }

    #[test]
    fn no_backend_explain() {
        let mut session = Session::new();
        let stmt = parser::parse(r#"EXPLAIN WALK "test";"#).unwrap();
        assert!(matches!(
            session.execute(&stmt).unwrap_err(),
            LqlError::NoBackend
        ));
    }

    #[test]
    fn no_backend_show_relations() {
        let mut session = Session::new();
        let stmt = parser::parse("SHOW RELATIONS;").unwrap();
        assert!(matches!(
            session.execute(&stmt).unwrap_err(),
            LqlError::NoBackend
        ));
    }

    #[test]
    fn no_backend_show_layers() {
        let mut session = Session::new();
        let stmt = parser::parse("SHOW LAYERS;").unwrap();
        assert!(matches!(
            session.execute(&stmt).unwrap_err(),
            LqlError::NoBackend
        ));
    }

    #[test]
    fn no_backend_show_features() {
        let mut session = Session::new();
        let stmt = parser::parse("SHOW FEATURES 26;").unwrap();
        assert!(matches!(
            session.execute(&stmt).unwrap_err(),
            LqlError::NoBackend
        ));
    }

    // ── USE errors ──

    #[test]
    fn use_nonexistent_vindex() {
        let mut session = Session::new();
        let stmt =
            parser::parse(r#"USE "/nonexistent/path/fake.vindex";"#).unwrap();
        let result = session.execute(&stmt);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), LqlError::Execution(_)));
    }

    #[test]
    fn use_model_not_implemented() {
        let mut session = Session::new();
        let stmt =
            parser::parse(r#"USE MODEL "google/gemma-3-4b-it";"#).unwrap();
        let result = session.execute(&stmt);
        // Returns guidance, not an error
        assert!(result.is_ok());
        let lines = result.unwrap();
        assert!(lines[0].contains("not yet implemented"));
    }

    #[test]
    fn use_model_auto_extract_noted() {
        let mut session = Session::new();
        let stmt = parser::parse(
            r#"USE MODEL "google/gemma-3-4b-it" AUTO_EXTRACT;"#,
        )
        .unwrap();
        let result = session.execute(&stmt).unwrap();
        assert!(result.iter().any(|l| l.contains("AUTO_EXTRACT")));
    }

    // ── Not-implemented stubs ──

    #[test]
    fn extract_not_implemented() {
        let mut session = Session::new();
        let stmt = parser::parse(
            r#"EXTRACT MODEL "model" INTO "out.vindex";"#,
        )
        .unwrap();
        let result = session.execute(&stmt);
        assert!(matches!(
            result.unwrap_err(),
            LqlError::NotImplemented(ref name, _) if name == "EXTRACT"
        ));
    }

    #[test]
    fn compile_not_implemented() {
        let mut session = Session::new();
        let stmt = parser::parse(
            r#"COMPILE CURRENT INTO MODEL "out/" FORMAT safetensors;"#,
        )
        .unwrap();
        assert!(matches!(
            session.execute(&stmt).unwrap_err(),
            LqlError::NotImplemented(ref name, _) if name == "COMPILE"
        ));
    }

    #[test]
    fn diff_not_implemented() {
        let mut session = Session::new();
        let stmt =
            parser::parse(r#"DIFF "a.vindex" "b.vindex";"#).unwrap();
        assert!(matches!(
            session.execute(&stmt).unwrap_err(),
            LqlError::NotImplemented(ref name, _) if name == "DIFF"
        ));
    }

    #[test]
    fn insert_no_backend() {
        let mut session = Session::new();
        let stmt = parser::parse(
            r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#,
        )
        .unwrap();
        assert!(matches!(
            session.execute(&stmt).unwrap_err(),
            LqlError::NoBackend
        ));
    }

    #[test]
    fn delete_not_implemented() {
        let mut session = Session::new();
        let stmt = parser::parse(
            r#"DELETE FROM EDGES WHERE entity = "x";"#,
        )
        .unwrap();
        assert!(matches!(
            session.execute(&stmt).unwrap_err(),
            LqlError::NotImplemented(ref name, _) if name == "DELETE"
        ));
    }

    #[test]
    fn update_not_implemented() {
        let mut session = Session::new();
        let stmt = parser::parse(
            r#"UPDATE EDGES SET target = "y" WHERE entity = "x";"#,
        )
        .unwrap();
        assert!(matches!(
            session.execute(&stmt).unwrap_err(),
            LqlError::NotImplemented(ref name, _) if name == "UPDATE"
        ));
    }

    #[test]
    fn merge_not_implemented() {
        let mut session = Session::new();
        let stmt =
            parser::parse(r#"MERGE "source.vindex";"#).unwrap();
        assert!(matches!(
            session.execute(&stmt).unwrap_err(),
            LqlError::NotImplemented(ref name, _) if name == "MERGE"
        ));
    }

    // ── INFER ──

    #[test]
    fn infer_no_backend() {
        let mut session = Session::new();
        let stmt = parser::parse(r#"INFER "test" TOP 5;"#).unwrap();
        assert!(matches!(
            session.execute(&stmt).unwrap_err(),
            LqlError::NoBackend
        ));
    }

    // ── is_readable_token ──

    #[test]
    fn readable_tokens() {
        assert!(is_readable_token("French"));
        assert!(is_readable_token("Paris"));
        assert!(is_readable_token("capital-of"));
        assert!(is_readable_token("is"));
        assert!(is_readable_token("Europe"));
    }

    #[test]
    fn unreadable_tokens() {
        assert!(!is_readable_token("ইসলামাবাদ"));
        assert!(!is_readable_token("южна"));
        assert!(!is_readable_token("ളാ"));
        assert!(!is_readable_token("ڪ"));
        assert!(!is_readable_token(""));
    }

    // ── SHOW MODELS works without backend ──

    #[test]
    fn show_models_no_crash() {
        let mut session = Session::new();
        let stmt = parser::parse("SHOW MODELS;").unwrap();
        // SHOW MODELS scans CWD, should not crash even without a backend
        let result = session.execute(&stmt);
        assert!(result.is_ok());
    }

    // ── Pipe: errors propagate ──

    #[test]
    fn pipe_error_propagates() {
        let mut session = Session::new();
        let stmt = parser::parse(
            r#"STATS |> WALK "test";"#,
        )
        .unwrap();
        // First statement (STATS) requires backend, should error
        assert!(session.execute(&stmt).is_err());
    }

    // ── Format helpers ──

    #[test]
    fn format_number_small() {
        assert_eq!(format_number(42), "42");
        assert_eq!(format_number(999), "999");
    }

    #[test]
    fn format_number_thousands() {
        assert_eq!(format_number(1_000), "1.0K");
        assert_eq!(format_number(10_240), "10.2K");
        assert_eq!(format_number(348_160), "348.2K");
    }

    #[test]
    fn format_number_millions() {
        assert_eq!(format_number(1_000_000), "1.00M");
        assert_eq!(format_number(2_917_432), "2.92M");
    }

    #[test]
    fn format_bytes_small() {
        assert_eq!(format_bytes(512), "512 B");
    }

    #[test]
    fn format_bytes_kb() {
        assert_eq!(format_bytes(2048), "2.0 KB");
    }

    #[test]
    fn format_bytes_mb() {
        let mb = 5 * 1_048_576;
        assert_eq!(format_bytes(mb), "5.0 MB");
    }

    #[test]
    fn format_bytes_gb() {
        let gb = 6_420_000_000;
        assert!(format_bytes(gb).contains("GB"));
    }
}
