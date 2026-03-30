//! Graph-based FFN backend — replaces the gate matmul with a precomputed index.
//!
//! Offline: for each layer, compute gate activations for every embedding token,
//! record the top-K features per token. This is the "graph" — a token→features map.
//!
//! Runtime: project residual into embedding space (find nearest tokens),
//! look up their precomputed feature lists, run sparse up/down on those features.
//!
//! Eliminates the gate matmul entirely. One embedding projection + hash lookup
//! replaces 500ms of BLAS.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use ndarray::Array2;

use crate::error::InferenceError;
use crate::ffn::{sigmoid, FfnBackend};
use crate::model::ModelWeights;

/// Precomputed gate index: for each (layer, token_id), which features activate.
/// Built offline from the gate weight matrix and embedding matrix.
/// Serializable to disk for reuse across predict calls.
pub struct GateIndex {
    /// layer → per-token feature lists. index[layer][token_id] = [(feature_id, gate_act), ...]
    index: HashMap<usize, Vec<Vec<(usize, f32)>>>,
    /// How many top tokens to match the residual against at runtime.
    pub top_tokens: usize,
    /// How many features were indexed per token (for metadata).
    pub features_per_token: usize,
}

/// Callbacks for gate index build progress.
pub trait IndexBuildCallbacks {
    fn on_layer_start(&mut self, _layer: usize, _total_layers: usize) {}
    fn on_layer_done(&mut self, _layer: usize, _elapsed_ms: f64) {}
}

pub struct SilentIndexCallbacks;
impl IndexBuildCallbacks for SilentIndexCallbacks {}

impl GateIndex {
    /// Build the gate index from model weights.
    ///
    /// For each layer, for each token in the vocabulary:
    /// 1. Compute `gate_activation = SiLU(embedding[token] * embed_scale @ gate.T)`
    /// 2. Store top `features_per_token` features by magnitude
    ///
    /// This is the expensive offline step — one gate matmul per layer.
    pub fn build(
        weights: &ModelWeights,
        layers: &[usize],
        features_per_token: usize,
        top_tokens: usize,
        callbacks: &mut dyn IndexBuildCallbacks,
    ) -> Self {
        let vocab_size = weights.vocab_size;
        let embed_scale = weights.arch.embed_scale();
        let total = layers.len();

        // Scale embeddings once (Gemma convention)
        let scaled_embed = &weights.embed * embed_scale;

        let mut index = HashMap::new();

        for (idx, &layer) in layers.iter().enumerate() {
            callbacks.on_layer_start(layer, total);
            let start = std::time::Instant::now();

            let gate_key = weights.arch.ffn_gate_key(layer);
            let w_gate = match weights.tensors.get(&gate_key) {
                Some(w) => w,
                None => continue,
            };

            let intermediate = w_gate.shape()[0];
            let k = features_per_token.min(intermediate);

            // Process tokens in batches to avoid OOM on the (vocab × intermediate) matrix.
            // 8192 tokens × 10240 features × 4 bytes = 320MB per batch (vs 10GB for full vocab).
            let batch_size = 8192;
            let mut layer_index: Vec<Vec<(usize, f32)>> = Vec::with_capacity(vocab_size);

            for batch_start in (0..vocab_size).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(vocab_size);
                let embed_chunk = scaled_embed.slice(ndarray::s![batch_start..batch_end, ..]);
                let gate_proj = embed_chunk.dot(&w_gate.t());

                for row_idx in 0..(batch_end - batch_start) {
                    let mut features: Vec<(usize, f32)> = gate_proj
                        .row(row_idx)
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(i, v)| (i, v * sigmoid(v)))
                        .collect();

                    if k < intermediate {
                        features.select_nth_unstable_by(k, |a, b| {
                            b.1.abs().partial_cmp(&a.1.abs()).unwrap()
                        });
                        features.truncate(k);
                    }
                    features.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                    layer_index.push(features);
                }
            }

            index.insert(layer, layer_index);

            let _ = idx; // used for progress via callbacks
            callbacks.on_layer_done(layer, start.elapsed().as_secs_f64() * 1000.0);
        }

        GateIndex {
            index,
            top_tokens,
            features_per_token,
        }
    }

    /// Build the gate index and stream directly to disk — never holds more than
    /// one layer's worth of index data in memory at a time.
    pub fn build_streaming(
        weights: &ModelWeights,
        layers: &[usize],
        features_per_token: usize,
        top_tokens: usize,
        path: &Path,
        callbacks: &mut dyn IndexBuildCallbacks,
    ) -> Result<(), InferenceError> {
        let vocab_size = weights.vocab_size;
        let embed_scale = weights.arch.embed_scale();
        let total = layers.len();

        let scaled_embed = &weights.embed * embed_scale;

        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Header
        let header = serde_json::json!({
            "_header": true,
            "type": "gate_index",
            "top_tokens": top_tokens,
            "features_per_token": features_per_token,
            "layers": layers,
        });
        serde_json::to_writer(&mut writer, &header)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        writer.write_all(b"\n")?;

        for (idx, &layer) in layers.iter().enumerate() {
            callbacks.on_layer_start(layer, total);
            let start = std::time::Instant::now();

            let gate_key = weights.arch.ffn_gate_key(layer);
            let w_gate = match weights.tensors.get(&gate_key) {
                Some(w) => w,
                None => continue,
            };

            let intermediate = w_gate.shape()[0];
            let k = features_per_token.min(intermediate);
            let batch_size = 8192;
            let mut tok_id = 0usize;

            for batch_start in (0..vocab_size).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(vocab_size);
                let embed_chunk = scaled_embed.slice(ndarray::s![batch_start..batch_end, ..]);
                let gate_proj = embed_chunk.dot(&w_gate.t());

                for row_idx in 0..(batch_end - batch_start) {
                    let mut features: Vec<(usize, f32)> = gate_proj
                        .row(row_idx)
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(i, v)| (i, v * sigmoid(v)))
                        .collect();

                    if k < intermediate {
                        features.select_nth_unstable_by(k, |a, b| {
                            b.1.abs().partial_cmp(&a.1.abs()).unwrap()
                        });
                        features.truncate(k);
                    }
                    features.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

                    if !features.is_empty() {
                        let flat: Vec<f32> =
                            features.iter().flat_map(|&(f, a)| [f as f32, a]).collect();
                        let record = serde_json::json!({
                            "l": layer,
                            "t": tok_id,
                            "f": flat,
                        });
                        serde_json::to_writer(&mut writer, &record)
                            .map_err(|e| InferenceError::Parse(e.to_string()))?;
                        writer.write_all(b"\n")?;
                    }

                    tok_id += 1;
                }
            }

            writer.flush()?;
            let _ = idx;
            callbacks.on_layer_done(layer, start.elapsed().as_secs_f64() * 1000.0);
        }

        writer.flush()?;
        Ok(())
    }

    /// Save the gate index to an NDJSON file.
    /// Format: header line, then one line per (layer, token) entry.
    pub fn save(&self, path: &Path) -> Result<(), InferenceError> {
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Header
        let header = serde_json::json!({
            "_header": true,
            "type": "gate_index",
            "top_tokens": self.top_tokens,
            "features_per_token": self.features_per_token,
            "layers": self.index.keys().collect::<Vec<_>>(),
        });
        serde_json::to_writer(&mut writer, &header)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        writer.write_all(b"\n")?;

        // One line per (layer, token) with compact feature lists
        let mut layers: Vec<usize> = self.index.keys().copied().collect();
        layers.sort();
        for layer in layers {
            let layer_data = &self.index[&layer];
            for (tok_id, features) in layer_data.iter().enumerate() {
                if features.is_empty() {
                    continue;
                }
                // Compact format: [feat_id, gate_act, feat_id, gate_act, ...]
                let flat: Vec<f32> = features.iter().flat_map(|&(f, a)| [f as f32, a]).collect();
                let record = serde_json::json!({
                    "l": layer,
                    "t": tok_id,
                    "f": flat,
                });
                serde_json::to_writer(&mut writer, &record)
                    .map_err(|e| InferenceError::Parse(e.to_string()))?;
                writer.write_all(b"\n")?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Load a gate index from an NDJSON file.
    pub fn load(path: &Path, top_tokens: usize) -> Result<Self, InferenceError> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);

        let mut index: HashMap<usize, Vec<Vec<(usize, f32)>>> = HashMap::new();
        let mut features_per_token = 0;

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let obj: serde_json::Value =
                serde_json::from_str(line).map_err(|e| InferenceError::Parse(e.to_string()))?;

            if obj.get("_header").is_some() {
                features_per_token = obj["features_per_token"].as_u64().unwrap_or(100) as usize;
                continue;
            }

            let layer = obj["l"].as_u64().unwrap() as usize;
            let tok_id = obj["t"].as_u64().unwrap() as usize;
            let flat: Vec<f32> = obj["f"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();

            // Decode flat pairs: [feat_id, gate_act, feat_id, gate_act, ...]
            let features: Vec<(usize, f32)> = flat
                .chunks_exact(2)
                .map(|pair| (pair[0] as usize, pair[1]))
                .collect();

            let layer_vec = index.entry(layer).or_default();
            // Extend to fit tok_id
            while layer_vec.len() <= tok_id {
                layer_vec.push(Vec::new());
            }
            layer_vec[tok_id] = features;
        }

        Ok(GateIndex {
            index,
            top_tokens,
            features_per_token,
        })
    }

    /// Number of layers indexed.
    pub fn num_layers(&self) -> usize {
        self.index.len()
    }

    /// Total entries across all layers.
    pub fn total_entries(&self) -> usize {
        self.index.values().map(|v| v.len()).sum()
    }

    /// Look up candidate features from pre-resolved token matches.
    /// Returns deduplicated feature IDs (no activation values — caller computes real gates).
    pub fn lookup_from_tokens(
        &self,
        token_scores: &[(usize, f32)],
        layer: usize,
        total_k: usize,
    ) -> Vec<usize> {
        let layer_index = match self.index.get(&layer) {
            Some(idx) => idx,
            None => return vec![],
        };

        // Union features from matched tokens, dedup, keep highest-magnitude precomputed score for ranking
        let mut feature_map: HashMap<usize, f32> = HashMap::new();
        for &(tok_id, _) in token_scores {
            if tok_id < layer_index.len() {
                for &(feat_id, gate_act) in &layer_index[tok_id] {
                    let entry = feature_map.entry(feat_id).or_insert(0.0);
                    if gate_act.abs() > entry.abs() {
                        *entry = gate_act;
                    }
                }
            }
        }

        // Select top-K by precomputed activation magnitude (used only for ranking/selection)
        let mut features: Vec<(usize, f32)> = feature_map.into_iter().collect();
        let k = total_k.min(features.len());
        if k > 0 && k < features.len() {
            features.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            features.truncate(k);
        }
        features.into_iter().map(|(id, _)| id).collect()
    }

    /// Precompute entity feature lists for all layers at once.
    /// Returns a vec indexed by layer number (sparse — unlisted layers are empty).
    /// Zero allocation at query time — just index into the vec.
    pub fn precompute_entity(
        &self,
        token_ids: &[u32],
        top_k: usize,
    ) -> Vec<Vec<usize>> {
        let token_scores: Vec<(usize, f32)> = token_ids.iter().map(|&t| (t as usize, 1.0)).collect();
        let max_layer = self.index.keys().copied().max().unwrap_or(0);
        let mut result = vec![Vec::new(); max_layer + 1];
        for (&layer, _) in &self.index {
            result[layer] = self.lookup_from_tokens(&token_scores, layer, top_k);
        }
        result
    }

    /// Look up which features should activate for a given residual at a layer.
    ///
    /// Projects residual against the embedding matrix, finds the top-N nearest tokens,
    /// unions their precomputed feature lists, deduplicates, returns top-K by activation.
    #[allow(dead_code)]
    fn lookup(
        &self,
        layer: usize,
        residual: &ndarray::ArrayView1<f32>,
        embed: &Array2<f32>,
        total_k: usize,
    ) -> Vec<(usize, f32)> {
        let layer_index = match self.index.get(&layer) {
            Some(idx) => idx,
            None => return vec![],
        };

        let vocab_size = embed.shape()[0];

        // Project residual against embedding matrix → (vocab_size,) logits
        // This is hidden-dim dot products, same cost as one row of the embed matrix.
        let mut token_scores: Vec<(usize, f32)> = Vec::with_capacity(vocab_size);
        for tok in 0..vocab_size {
            let row = embed.row(tok);
            let dot: f32 = residual.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
            token_scores.push((tok, dot));
        }

        // Top-N tokens by score
        let n = self.top_tokens.min(vocab_size);
        if n < vocab_size {
            token_scores.select_nth_unstable_by(n, |a, b| b.1.partial_cmp(&a.1).unwrap());
            token_scores.truncate(n);
        }

        // Union all features from top-N tokens, dedup by feature_id (keep max activation)
        let mut feature_map: HashMap<usize, f32> = HashMap::new();
        for &(tok_id, _token_score) in &token_scores {
            if tok_id < layer_index.len() {
                for &(feat_id, gate_act) in &layer_index[tok_id] {
                    let entry = feature_map.entry(feat_id).or_insert(0.0);
                    if gate_act.abs() > entry.abs() {
                        *entry = gate_act;
                    }
                }
            }
        }

        // Collect and select top-K overall
        let mut features: Vec<(usize, f32)> = feature_map.into_iter().collect();
        let k = total_k.min(features.len());
        if k > 0 && k < features.len() {
            features.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            features.truncate(k);
        }
        features
    }
}

/// Graph FFN backend: uses a precomputed gate index instead of the gate matmul.
///
/// Runtime: residual → embedding projection → token lookup → feature list → sparse up/down.
/// Eliminates the gate matmul (500ms → ~0.01ms for the lookup).
pub struct GraphFfn<'a> {
    pub weights: &'a ModelWeights,
    pub gate_index: &'a GateIndex,
    /// Max features to use per position.
    pub top_k: usize,
}

impl<'a> FfnBackend for GraphFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let (out, _) = self.forward_inner(layer, x);
        out
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        self.forward_inner(layer, x)
    }

    fn name(&self) -> &str {
        "graph"
    }
}

impl<'a> GraphFfn<'a> {
    fn forward_inner(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let arch = &*self.weights.arch;
        let w_gate = self.weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        let w_up = self.weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
        let hidden = x.shape()[1];
        let intermediate = w_up.shape()[0];
        let seq_len = x.shape()[0];
        let gate_raw = w_gate.as_slice().unwrap();
        let up_raw = w_up.as_slice().unwrap();

        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));
        let mut out = Array2::<f32>::zeros((seq_len, hidden));

        // Embedding projection for feature selection (BLAS matmul, not scalar loop)
        let embed_scale = self.weights.arch.embed_scale();
        let embed_proj = x.dot(&self.weights.embed.t()) * embed_scale;

        for s in 0..seq_len {
            let x_row = x.row(s);

            // Step 1: find nearest tokens via embedding projection (already computed)
            let logits = embed_proj.row(s);
            let mut token_scores: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            let n = self.gate_index.top_tokens.min(token_scores.len());
            if n < token_scores.len() {
                token_scores.select_nth_unstable_by(n, |a, b| b.1.partial_cmp(&a.1).unwrap());
                token_scores.truncate(n);
            }

            // Step 2: look up candidate features from index, dedup
            let features = self.gate_index.lookup_from_tokens(&token_scores, layer, self.top_k);
            let k = features.len();
            if k == 0 {
                continue;
            }

            // Step 3: compute ACTUAL gate values from the residual (not precomputed)
            let mut gate_buf = vec![0.0f32; k * hidden];
            let mut up_buf = vec![0.0f32; k * hidden];
            for (i, &feat) in features.iter().enumerate() {
                let src = feat * hidden;
                gate_buf[i * hidden..(i + 1) * hidden].copy_from_slice(&gate_raw[src..src + hidden]);
                up_buf[i * hidden..(i + 1) * hidden].copy_from_slice(&up_raw[src..src + hidden]);
            }

            let gate_sub = ndarray::ArrayView2::from_shape((k, hidden), &gate_buf[..k * hidden]).unwrap();
            let up_sub = ndarray::ArrayView2::from_shape((k, hidden), &up_buf[..k * hidden]).unwrap();

            let gate_proj = gate_sub.dot(&x_row); // (K,) actual gate values
            let up_proj = up_sub.dot(&x_row);     // (K,) up values

            // activation = SiLU(gate) * up
            for (i, &feat) in features.iter().enumerate() {
                let g = gate_proj[i];
                let act_val = (g * sigmoid(g)) * up_proj[i];
                full_activation[[s, feat]] = act_val;
            }

            // Step 4: sparse down projection — only accumulate non-zero features
            let down_raw = w_down.as_slice().unwrap();
            let mut out_row = out.row_mut(s);
            for (_i, &feat) in features.iter().enumerate() {
                let act_val = full_activation[[s, feat]];
                if act_val.abs() < 1e-12 {
                    continue;
                }
                // w_down is (hidden, intermediate), row-major: row j starts at j*intermediate
                for j in 0..hidden {
                    out_row[j] += act_val * down_raw[j * intermediate + feat];
                }
            }
        }

        (out, full_activation)
    }
}

// ── Entity-routed FFN: preselect features once, reuse across all layers ──

/// Entity-routed FFN backend: resolves entity tokens once at construction,
/// then uses the gate index for O(1) feature lookup per layer.
/// Eliminates both the gate matmul AND per-layer embedding projection.
///
/// Flow:
/// 1. Construction: input embedding → top-N tokens (one-time embedding projection)
/// 2. Per-layer forward: token_ids → GateIndex hash lookup → feature_ids
/// 3. Gather gate+up rows for selected features, compute SiLU(gate)*up, sparse down
pub struct EntityRoutedFfn<'a> {
    pub weights: &'a ModelWeights,
    pub gate_index: &'a GateIndex,
    /// Pre-resolved token IDs from input embedding.
    pub entity_tokens: Vec<(usize, f32)>,
    /// Max features per layer.
    pub top_k: usize,
}

impl<'a> EntityRoutedFfn<'a> {
    /// Create from a pre-FFN hidden state. Projects against embeddings once
    /// to identify entity tokens, which are reused for all layers.
    pub fn from_hidden(
        weights: &'a ModelWeights,
        gate_index: &'a GateIndex,
        hidden_state: &ndarray::Array1<f32>,
        top_k: usize,
    ) -> Self {
        let embed = &weights.embed;
        let embed_scale = weights.arch.embed_scale();
        let vocab_size = embed.shape()[0];

        // Single BLAS gemv: hidden_state @ embed.T → (vocab_size,)
        let logits = embed.dot(hidden_state) * embed_scale;

        let mut token_scores: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        let n = gate_index.top_tokens.min(vocab_size);
        if n < vocab_size {
            token_scores.select_nth_unstable_by(n, |a, b| b.1.partial_cmp(&a.1).unwrap());
            token_scores.truncate(n);
        }

        EntityRoutedFfn {
            weights,
            gate_index,
            entity_tokens: token_scores,
            top_k,
        }
    }

    /// Create directly from known token IDs (e.g., from input tokens).
    pub fn from_token_ids(
        weights: &'a ModelWeights,
        gate_index: &'a GateIndex,
        token_ids: &[u32],
        top_k: usize,
    ) -> Self {
        let entity_tokens: Vec<(usize, f32)> =
            token_ids.iter().map(|&t| (t as usize, 1.0)).collect();
        EntityRoutedFfn {
            weights,
            gate_index,
            entity_tokens,
            top_k,
        }
    }
}

impl<'a> FfnBackend for EntityRoutedFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let (out, _) = self.forward_inner(layer, x);
        out
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        self.forward_inner(layer, x)
    }

    fn name(&self) -> &str {
        "entity-routed"
    }
}

impl<'a> EntityRoutedFfn<'a> {
    fn forward_inner(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let arch = &*self.weights.arch;
        let w_gate = self.weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        let w_up = self.weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
        let hidden = x.shape()[1];
        let intermediate = w_up.shape()[0];
        let seq_len = x.shape()[0];
        let gate_raw = w_gate.as_slice().unwrap();
        let up_raw = w_up.as_slice().unwrap();
        let down_raw = w_down.as_slice().unwrap();

        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));
        let mut out = Array2::<f32>::zeros((seq_len, hidden));

        // Feature selection: hash lookup from pre-resolved entity tokens (no matmul)
        let features = self
            .gate_index
            .lookup_from_tokens(&self.entity_tokens, layer, self.top_k);
        let k = features.len();
        if k == 0 {
            return (out, full_activation);
        }

        // Pre-gather gate and up rows for selected features
        let mut gate_buf = vec![0.0f32; k * hidden];
        let mut up_buf = vec![0.0f32; k * hidden];
        for (i, &feat) in features.iter().enumerate() {
            let src = feat * hidden;
            gate_buf[i * hidden..(i + 1) * hidden].copy_from_slice(&gate_raw[src..src + hidden]);
            up_buf[i * hidden..(i + 1) * hidden].copy_from_slice(&up_raw[src..src + hidden]);
        }

        let gate_sub =
            ndarray::ArrayView2::from_shape((k, hidden), &gate_buf[..k * hidden]).unwrap();
        let up_sub =
            ndarray::ArrayView2::from_shape((k, hidden), &up_buf[..k * hidden]).unwrap();

        for s in 0..seq_len {
            let x_row = x.row(s);

            // Sparse gate + up projection: (K, hidden) @ (hidden,) → (K,)
            let gate_proj = gate_sub.dot(&x_row);
            let up_proj = up_sub.dot(&x_row);

            // SiLU(gate) * up
            for (i, &feat) in features.iter().enumerate() {
                let g = gate_proj[i];
                let act_val = (g * sigmoid(g)) * up_proj[i];
                full_activation[[s, feat]] = act_val;
            }

            // Sparse down projection
            let mut out_row = out.row_mut(s);
            for (_i, &feat) in features.iter().enumerate() {
                let act_val = full_activation[[s, feat]];
                if act_val.abs() < 1e-12 {
                    continue;
                }
                for j in 0..hidden {
                    out_row[j] += act_val * down_raw[j * intermediate + feat];
                }
            }
        }

        (out, full_activation)
    }
}

// ── Clustered gate index: hierarchical two-level feature selection ──

struct LayerClusters {
    centroids: ndarray::Array2<f32>,
    members: Vec<Vec<usize>>,
}

/// Clustered gate index: K-means on gate vectors per layer.
pub struct ClusteredGateIndex {
    layers: HashMap<usize, LayerClusters>,
    pub num_clusters: usize,
    pub top_c: usize,
}

impl ClusteredGateIndex {
    pub fn build(
        weights: &ModelWeights,
        layers: &[usize],
        num_clusters: usize,
        top_c: usize,
        kmeans_iters: usize,
        mut on_layer: impl FnMut(usize, usize),
    ) -> Self {
        let mut layer_map = HashMap::new();
        let total = layers.len();
        for (idx, &layer) in layers.iter().enumerate() {
            on_layer(idx, total);
            let gate_key = weights.arch.ffn_gate_key(layer);
            let w_gate = match weights.tensors.get(&gate_key) {
                Some(w) => w,
                None => continue,
            };
            layer_map.insert(layer, Self::kmeans(w_gate, num_clusters, kmeans_iters));
        }
        ClusteredGateIndex { layers: layer_map, num_clusters, top_c }
    }

    fn kmeans(w_gate: &Array2<f32>, k: usize, iters: usize) -> LayerClusters {
        let n = w_gate.shape()[0];
        let d = w_gate.shape()[1];
        let k = k.min(n);
        let mut centroids = ndarray::Array2::<f32>::zeros((k, d));
        for c in 0..k { centroids.row_mut(c).assign(&w_gate.row(c * n / k)); }
        let mut assignments = vec![0usize; n];
        for _iter in 0..iters {
            let scores = w_gate.dot(&centroids.t());
            for i in 0..n {
                let row = scores.row(i);
                let (best_c, _) = row.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
                assignments[i] = best_c;
            }
            let mut sums = ndarray::Array2::<f32>::zeros((k, d));
            let mut counts = vec![0usize; k];
            for i in 0..n {
                let c = assignments[i];
                counts[c] += 1;
                for j in 0..d { sums[[c, j]] += w_gate[[i, j]]; }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    let cnt = counts[c] as f32;
                    for j in 0..d { centroids[[c, j]] = sums[[c, j]] / cnt; }
                }
                let norm: f32 = centroids.row(c).iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 1e-12 { for j in 0..d { centroids[[c, j]] /= norm; } }
            }
        }
        let mut members = vec![Vec::new(); k];
        for i in 0..n { members[assignments[i]].push(i); }
        LayerClusters { centroids, members }
    }

    pub fn lookup(&self, layer: usize, residual: &ndarray::ArrayView1<f32>, top_k: usize) -> Vec<usize> {
        let lc = match self.layers.get(&layer) { Some(lc) => lc, None => return vec![] };
        let scores = lc.centroids.dot(residual);
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        let c = self.top_c.min(indexed.len());
        if c < indexed.len() {
            indexed.select_nth_unstable_by(c, |a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(c);
        }
        let mut features: Vec<usize> = Vec::new();
        for &(cid, _) in &indexed { features.extend_from_slice(&lc.members[cid]); }
        features.sort_unstable();
        features.dedup();
        features.truncate(top_k);
        features
    }

    pub fn num_layers(&self) -> usize { self.layers.len() }
    pub fn avg_cluster_size(&self) -> f64 {
        let (mut t, mut c) = (0usize, 0usize);
        for lc in self.layers.values() { for m in &lc.members { t += m.len(); c += 1; } }
        if c > 0 { t as f64 / c as f64 } else { 0.0 }
    }
}

/// Clustered FFN backend.
pub struct ClusteredFfn<'a> {
    pub weights: &'a ModelWeights,
    pub cluster_index: &'a ClusteredGateIndex,
    pub top_k: usize,
}

impl<'a> FfnBackend for ClusteredFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> { self.forward_inner(layer, x).0 }
    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) { self.forward_inner(layer, x) }
    fn name(&self) -> &str { "clustered" }
}

impl<'a> ClusteredFfn<'a> {
    fn forward_inner(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let arch = &*self.weights.arch;
        let w_gate = self.weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        let w_up = self.weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
        let hidden = x.shape()[1];
        let intermediate = w_up.shape()[0];
        let seq_len = x.shape()[0];
        let gate_raw = w_gate.as_slice().unwrap();
        let up_raw = w_up.as_slice().unwrap();
        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));
        let mut out = Array2::<f32>::zeros((seq_len, hidden));

        for s in 0..seq_len {
            let x_row = x.row(s);
            let features = self.cluster_index.lookup(layer, &x_row, self.top_k);
            let k = features.len();
            if k == 0 { continue; }
            let mut gate_buf = vec![0.0f32; k * hidden];
            let mut up_buf = vec![0.0f32; k * hidden];
            for (i, &feat) in features.iter().enumerate() {
                let src = feat * hidden;
                gate_buf[i * hidden..(i + 1) * hidden].copy_from_slice(&gate_raw[src..src + hidden]);
                up_buf[i * hidden..(i + 1) * hidden].copy_from_slice(&up_raw[src..src + hidden]);
            }
            let gate_sub = ndarray::ArrayView2::from_shape((k, hidden), &gate_buf).unwrap();
            let up_sub = ndarray::ArrayView2::from_shape((k, hidden), &up_buf).unwrap();
            let gate_proj = gate_sub.dot(&x_row);
            let up_proj = up_sub.dot(&x_row);

            for (i, &feat) in features.iter().enumerate() {
                let g = gate_proj[i];
                full_activation[[s, feat]] = (g * sigmoid(g)) * up_proj[i];
            }

            // Down projection: sparse accumulation for small K, BLAS for large K
            let mut out_row = out.row_mut(s);
            if k < 256 {
                let down_raw = w_down.as_slice().unwrap();
                for &feat in &features {
                    let a = full_activation[[s, feat]];
                    if a.abs() < 1e-12 { continue; }
                    for j in 0..hidden { out_row[j] += a * down_raw[j * intermediate + feat]; }
                }
            } else {
                let act_row = full_activation.row(s);
                let out_vec = w_down.dot(&act_row);
                ndarray::Zip::from(&mut out_row).and(&out_vec).for_each(|o, &v| *o += v);
            }
        }
        (out, full_activation)
    }
}

// ── Cached FFN: precomputed FFN outputs, zero matmuls at runtime ──

/// Cached FFN backend: stores precomputed FFN output matrices per layer.
/// Built by running a calibration forward pass for each entity.
/// Runtime: ArcArray clone = refcount bump (no memcpy), no matrix multiplications.
pub struct CachedFfn {
    /// layer → shared FFN output matrix. Clone is O(1) refcount bump.
    cache: HashMap<usize, ndarray::ArcArray2<f32>>,
    hidden_size: usize,
}

impl CachedFfn {
    /// Build cache by running a dense forward pass, capturing FFN outputs at each layer.
    pub fn calibrate(
        weights: &ModelWeights,
        token_ids: &[u32],
    ) -> Self {
        use crate::ffn::WeightFfn;
        use crate::forward::trace_forward_with_ffn;

        let num_layers = weights.num_layers;
        let hidden = weights.hidden_size;
        let all_layers: Vec<usize> = (0..num_layers).collect();

        // Run forward pass capturing activations (to get FFN outputs)
        let ffn = WeightFfn { weights };
        let _trace = trace_forward_with_ffn(
            weights, token_ids, &all_layers, true, 1, &ffn,
        );

        // For each layer, compute the FFN delta:
        // FFN delta = post-FFN residual - post-attention residual
        // But we don't have those separately from trace. Instead, we can
        // re-derive: run attention to get post-attn, then the FFN output is
        // what the dense backend would produce.
        //
        // Simpler approach: run each layer's FFN on the captured residual.
        // The residual at layer L is the POST-layer-L state (after attn+FFN).
        // We need the PRE-FFN state (post-attention). We can get the FFN output
        // by running FFN on the normed residual.
        //
        // Actually the cleanest: run a second pass capturing FFN outputs directly.

        // Approach: run layer-by-layer, capture FFN output at each layer.
        let seq_len = token_ids.len();
        let embed_scale = weights.arch.embed_scale();
        let mut h = ndarray::Array2::<f32>::zeros((seq_len, hidden));
        for (i, &tok_id) in token_ids.iter().enumerate() {
            let row = weights.embed.row(tok_id as usize);
            for j in 0..hidden { h[[i, j]] = row[j] * embed_scale; }
        }

        let mut cache = HashMap::new();
        let norm_offset = weights.arch.norm_weight_offset();

        for layer in 0..num_layers {
            // Run attention
            let h_post_attn = match crate::forward::run_attention_public(weights, &h, layer) {
                Some(ha) => ha,
                None => { h = h.clone(); continue; }
            };

            // Compute FFN output on the post-attention residual
            let arch = &*weights.arch;
            let pre_ffn_key = if arch.has_post_norms() {
                arch.pre_feedforward_layernorm_key(layer)
            } else {
                Some(arch.post_attention_layernorm_key(layer))
            };
            let h_ffn = crate::residual::rms_norm(
                &h_post_attn,
                pre_ffn_key.and_then(|k| weights.vectors.get(&k)),
                norm_offset,
            );

            let ffn_out = ffn.forward(layer, &h_ffn);

            // Cache the full FFN output matrix (all positions)
            cache.insert(layer, ffn_out.clone().into_shared());

            // Apply FFN to get post-layer residual (for next layer)
            h = if arch.has_post_norms() {
                let normed = crate::residual::rms_norm(
                    &ffn_out,
                    arch.post_feedforward_layernorm_key(layer)
                        .and_then(|k| weights.vectors.get(&k)),
                    norm_offset,
                );
                &h_post_attn + &normed
            } else {
                &h_post_attn + &ffn_out
            };
        }

        CachedFfn { cache, hidden_size: hidden }
    }
}

impl CachedFfn {
    /// Direct access to cached output matrices (for zero-copy throughput paths).
    pub fn get_cache_vecs(&self) -> &HashMap<usize, ndarray::ArcArray2<f32>> {
        &self.cache
    }

    /// Save cache to a binary file. Format: JSON header line + raw f32 per layer.
    pub fn save(&self, path: &Path) -> Result<(), InferenceError> {
        let file = std::fs::File::create(path)?;
        let mut w = BufWriter::new(file);

        // Determine seq_len from first cached layer
        let seq_len = self.cache.values().next().map(|a| a.shape()[0]).unwrap_or(0);

        let mut sorted_layers: Vec<usize> = self.cache.keys().copied().collect();
        sorted_layers.sort();

        let header = serde_json::json!({
            "_type": "ffn_cache",
            "hidden_size": self.hidden_size,
            "seq_len": seq_len,
            "num_layers": self.cache.len(),
            "layers": sorted_layers,
        });
        serde_json::to_writer(&mut w, &header)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        w.write_all(b"\n")?;

        // Write each layer's data as raw f32 in layer order
        let mut layers: Vec<usize> = self.cache.keys().copied().collect();
        layers.sort();
        for layer in layers {
            let arr = &self.cache[&layer];
            let slice = arr.as_slice().unwrap();
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4)
            };
            w.write_all(bytes)?;
        }
        w.flush()?;
        Ok(())
    }

    /// Load cache from a binary file.
    pub fn load(path: &Path) -> Result<Self, InferenceError> {
        let mut file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(&mut file);

        // Read header line
        let mut header_line = String::new();
        reader.read_line(&mut header_line)?;
        let header: serde_json::Value = serde_json::from_str(&header_line)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;

        let hidden_size = header["hidden_size"].as_u64().unwrap() as usize;
        let seq_len = header["seq_len"].as_u64().unwrap() as usize;
        let layers: Vec<usize> = header["layers"].as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();

        let floats_per_layer = seq_len * hidden_size;
        let bytes_per_layer = floats_per_layer * 4;
        let mut cache = HashMap::new();

        for layer in layers {
            let mut buf = vec![0u8; bytes_per_layer];
            std::io::Read::read_exact(&mut reader, &mut buf)?;
            let floats: Vec<f32> = buf.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let arr = ndarray::Array2::from_shape_vec((seq_len, hidden_size), floats)
                .map_err(|e| InferenceError::Parse(e.to_string()))?;
            cache.insert(layer, arr.into_shared());
        }

        Ok(CachedFfn { cache, hidden_size })
    }

    /// Number of cached layers.
    pub fn num_layers(&self) -> usize {
        self.cache.len()
    }
}

impl FfnBackend for CachedFfn {
    fn forward(&self, layer: usize, _x: &Array2<f32>) -> Array2<f32> {
        match self.cache.get(&layer) {
            // ArcArray clone = refcount bump (O(1)), then .into_owned() only copies
            // if there are other references. Since we hold the only Arc, this is
            // typically a no-op move. But even if it copies, it's just memcpy.
            Some(cached) => cached.clone().into_owned(),
            None => Array2::<f32>::zeros((_x.shape()[0], self.hidden_size)),
        }
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        (self.forward(layer, x), Array2::<f32>::zeros((x.shape()[0], 1)))
    }

    fn name(&self) -> &str {
        "cached"
    }
}

// ── Down-clustered FFN: select features by output direction, not gate scan ──

/// Per-layer down clusters: centroids of down-projection columns.
struct DownClusters {
    /// Centroid vectors: (num_clusters, hidden_size) — average down direction per cluster.
    centroids: ndarray::Array2<f32>,
    /// members[c] = feature indices whose down vectors belong to cluster c.
    members: Vec<Vec<usize>>,
}

/// Down-clustered gate index: features grouped by what they OUTPUT.
/// Runtime: residual → nearest down centroids → candidate features → sparse gate/up/down.
pub struct DownClusteredIndex {
    layers: HashMap<usize, DownClusters>,
    pub num_clusters: usize,
    pub top_c: usize,
}

impl DownClusteredIndex {
    /// Build by clustering the columns of w_down at each layer.
    pub fn build(
        weights: &ModelWeights,
        layers: &[usize],
        num_clusters: usize,
        top_c: usize,
        kmeans_iters: usize,
        mut on_layer: impl FnMut(usize, usize),
    ) -> Self {
        let mut layer_map = HashMap::new();
        let total = layers.len();
        for (idx, &layer) in layers.iter().enumerate() {
            on_layer(idx, total);
            let arch = &*weights.arch;
            let w_down = match weights.tensors.get(&arch.ffn_down_key(layer)) {
                Some(w) => w,
                None => continue,
            };
            // w_down is (hidden, intermediate). We need to cluster by columns (features).
            // Transpose to (intermediate, hidden) so each row is a feature's down vector.
            let down_t = w_down.t().to_owned();
            layer_map.insert(layer, Self::kmeans(&down_t, num_clusters, kmeans_iters));
        }
        DownClusteredIndex { layers: layer_map, num_clusters, top_c }
    }

    fn kmeans(features: &ndarray::Array2<f32>, k: usize, iters: usize) -> DownClusters {
        let n = features.shape()[0];
        let d = features.shape()[1];
        let k = k.min(n);

        let mut centroids = ndarray::Array2::<f32>::zeros((k, d));
        for c in 0..k { centroids.row_mut(c).assign(&features.row(c * n / k)); }

        let mut assignments = vec![0usize; n];
        for _iter in 0..iters {
            let scores = features.dot(&centroids.t());
            for i in 0..n {
                let row = scores.row(i);
                let (best, _) = row.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
                assignments[i] = best;
            }
            let mut sums = ndarray::Array2::<f32>::zeros((k, d));
            let mut counts = vec![0usize; k];
            for i in 0..n {
                let c = assignments[i];
                counts[c] += 1;
                for j in 0..d { sums[[c, j]] += features[[i, j]]; }
            }
            for c in 0..k {
                if counts[c] > 0 {
                    let cnt = counts[c] as f32;
                    for j in 0..d { centroids[[c, j]] = sums[[c, j]] / cnt; }
                }
                let norm: f32 = centroids.row(c).iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 1e-12 { for j in 0..d { centroids[[c, j]] /= norm; } }
            }
        }

        let mut members = vec![Vec::new(); k];
        for i in 0..n { members[assignments[i]].push(i); }
        DownClusters { centroids, members }
    }

    /// Look up features whose down vectors point in the residual's direction.
    pub fn lookup(&self, layer: usize, residual: &ndarray::ArrayView1<f32>) -> Vec<usize> {
        let dc = match self.layers.get(&layer) { Some(dc) => dc, None => return vec![] };
        let scores = dc.centroids.dot(residual);
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        let c = self.top_c.min(indexed.len());
        if c < indexed.len() {
            indexed.select_nth_unstable_by(c, |a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.truncate(c);
        }
        let mut features = Vec::new();
        for &(cid, _) in &indexed { features.extend_from_slice(&dc.members[cid]); }
        features.sort_unstable();
        features.dedup();
        features
    }

    pub fn num_layers(&self) -> usize { self.layers.len() }
    pub fn avg_cluster_size(&self) -> f64 {
        let (mut t, mut c) = (0usize, 0usize);
        for dc in self.layers.values() { for m in &dc.members { t += m.len(); c += 1; } }
        if c > 0 { t as f64 / c as f64 } else { 0.0 }
    }
}

/// Down-clustered FFN backend: selects features by output direction, then computes
/// actual gate/up/down for those features only. No gate scan.
pub struct DownClusteredFfn<'a> {
    pub weights: &'a ModelWeights,
    pub down_index: &'a DownClusteredIndex,
}

impl<'a> FfnBackend for DownClusteredFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> { self.forward_inner(layer, x).0 }
    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) { self.forward_inner(layer, x) }
    fn name(&self) -> &str { "down-clustered" }
}

impl<'a> DownClusteredFfn<'a> {
    fn forward_inner(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let arch = &*self.weights.arch;
        let w_gate = self.weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        let w_up = self.weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
        let hidden = x.shape()[1];
        let intermediate = w_up.shape()[0];
        let seq_len = x.shape()[0];
        let gate_raw = w_gate.as_slice().unwrap();
        let up_raw = w_up.as_slice().unwrap();

        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));
        let mut out = Array2::<f32>::zeros((seq_len, hidden));

        for s in 0..seq_len {
            let x_row = x.row(s);

            // Select features by down-vector direction (output-directed)
            let features = self.down_index.lookup(layer, &x_row);
            let k = features.len();
            if k == 0 { continue; }

            // Gather gate + up rows
            let mut gate_buf = vec![0.0f32; k * hidden];
            let mut up_buf = vec![0.0f32; k * hidden];
            for (i, &feat) in features.iter().enumerate() {
                let src = feat * hidden;
                gate_buf[i * hidden..(i + 1) * hidden].copy_from_slice(&gate_raw[src..src + hidden]);
                up_buf[i * hidden..(i + 1) * hidden].copy_from_slice(&up_raw[src..src + hidden]);
            }

            let gate_sub = ndarray::ArrayView2::from_shape((k, hidden), &gate_buf).unwrap();
            let up_sub = ndarray::ArrayView2::from_shape((k, hidden), &up_buf).unwrap();
            let gate_proj = gate_sub.dot(&x_row);
            let up_proj = up_sub.dot(&x_row);

            for (i, &feat) in features.iter().enumerate() {
                let g = gate_proj[i];
                full_activation[[s, feat]] = (g * sigmoid(g)) * up_proj[i];
            }

            // Down projection
            let act_row = full_activation.row(s);
            let out_vec = w_down.dot(&act_row);
            let mut out_row = out.row_mut(s);
            ndarray::Zip::from(&mut out_row).and(&out_vec).for_each(|o, &v| *o += v);
        }
        (out, full_activation)
    }
}

// ── Precomputed feature lists: calibrate once, sparse FFN at query time ──

/// Stores precomputed feature lists per layer from a calibration forward pass.
/// At query time: attention runs live, FFN uses these feature lists for sparse
/// gate/up/down — no gate matmul scan.
pub struct FeatureListFfn<'a> {
    pub weights: &'a ModelWeights,
    /// layer → sorted feature indices (the ~50 features the gate matmul would select)
    feature_lists: Vec<Vec<usize>>,
}

impl<'a> FeatureListFfn<'a> {
    /// Calibrate: run a dense forward pass, capture which features the gate selects at each layer.
    pub fn calibrate(
        weights: &'a ModelWeights,
        token_ids: &[u32],
        top_k: usize,
    ) -> Self {
        use crate::ffn::WeightFfn;

        let num_layers = weights.num_layers;
        let hidden = weights.hidden_size;
        let seq_len = token_ids.len();
        let embed_scale = weights.arch.embed_scale();

        let mut h = ndarray::Array2::<f32>::zeros((seq_len, hidden));
        for (i, &tok_id) in token_ids.iter().enumerate() {
            let row = weights.embed.row(tok_id as usize);
            for j in 0..hidden { h[[i, j]] = row[j] * embed_scale; }
        }

        let ffn = WeightFfn { weights };
        let norm_offset = weights.arch.norm_weight_offset();
        let mut feature_lists = vec![Vec::new(); num_layers];

        for layer in 0..num_layers {
            // Run attention
            let h_post_attn = match crate::forward::run_attention_public(weights, &h, layer) {
                Some(ha) => ha,
                None => { continue; }
            };

            // Get the pre-FFN normed residual (what the gate matmul sees)
            let arch = &*weights.arch;
            let pre_ffn_key = if arch.has_post_norms() {
                arch.pre_feedforward_layernorm_key(layer)
            } else {
                Some(arch.post_attention_layernorm_key(layer))
            };
            let h_ffn = crate::residual::rms_norm(
                &h_post_attn,
                pre_ffn_key.and_then(|k| weights.vectors.get(&k)),
                norm_offset,
            );

            // Gate matmul on last position → find top-K features
            let w_gate = weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
            let last_row = h_ffn.row(seq_len - 1);
            let scores = w_gate.dot(&last_row);
            let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate()
                .map(|(i, v)| (i, v * sigmoid(v)))
                .collect();
            let k = top_k.min(indexed.len());
            indexed.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            indexed.truncate(k);
            let mut feats: Vec<usize> = indexed.iter().map(|&(id, _)| id).collect();
            feats.sort_unstable();
            feature_lists[layer] = feats;

            // Run dense FFN to get correct residual for next layer
            let ffn_out = ffn.forward(layer, &h_ffn);
            h = if arch.has_post_norms() {
                let normed = crate::residual::rms_norm(
                    &ffn_out,
                    arch.post_feedforward_layernorm_key(layer)
                        .and_then(|k| weights.vectors.get(&k)),
                    norm_offset,
                );
                &h_post_attn + &normed
            } else {
                &h_post_attn + &ffn_out
            };
        }

        FeatureListFfn { weights, feature_lists }
    }

    /// Save feature lists to a compact binary file.
    /// Format: JSON header + one line per layer with feature IDs.
    pub fn save(&self, path: &std::path::Path) -> Result<(), crate::error::InferenceError> {
        use std::io::Write;
        let file = std::fs::File::create(path)?;
        let mut w = std::io::BufWriter::new(file);

        let header = serde_json::json!({
            "_type": "feature_lists",
            "num_layers": self.feature_lists.len(),
        });
        serde_json::to_writer(&mut w, &header)
            .map_err(|e| crate::error::InferenceError::Parse(e.to_string()))?;
        w.write_all(b"\n")?;

        for (layer, feats) in self.feature_lists.iter().enumerate() {
            let record = serde_json::json!({ "l": layer, "f": feats });
            serde_json::to_writer(&mut w, &record)
                .map_err(|e| crate::error::InferenceError::Parse(e.to_string()))?;
            w.write_all(b"\n")?;
        }
        w.flush()?;
        Ok(())
    }

    /// Load feature lists from file.
    pub fn load(
        weights: &'a ModelWeights,
        path: &std::path::Path,
    ) -> Result<Self, crate::error::InferenceError> {
        use std::io::BufRead;
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);

        let num_layers = weights.num_layers;
        let mut feature_lists = vec![Vec::new(); num_layers];

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() { continue; }
            let obj: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| crate::error::InferenceError::Parse(e.to_string()))?;
            if obj.get("_type").is_some() { continue; }

            let layer = obj["l"].as_u64().unwrap_or(0) as usize;
            let feats: Vec<usize> = obj["f"].as_array().unwrap()
                .iter().map(|v| v.as_u64().unwrap() as usize).collect();
            if layer < num_layers {
                feature_lists[layer] = feats;
            }
        }

        Ok(FeatureListFfn { weights, feature_lists })
    }

    pub fn total_features(&self) -> usize {
        self.feature_lists.iter().map(|f| f.len()).sum()
    }

    pub fn avg_features_per_layer(&self) -> f64 {
        let active: Vec<_> = self.feature_lists.iter().filter(|f| !f.is_empty()).collect();
        if active.is_empty() { 0.0 } else {
            active.iter().map(|f| f.len()).sum::<usize>() as f64 / active.len() as f64
        }
    }
}

impl<'a> FfnBackend for FeatureListFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        self.forward_inner(layer, x).0
    }
    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        self.forward_inner(layer, x)
    }
    fn name(&self) -> &str { "feature-list" }
}

impl<'a> FeatureListFfn<'a> {
    fn forward_inner(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let arch = &*self.weights.arch;
        let w_gate = self.weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        let w_up = self.weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
        let hidden = x.shape()[1];
        let intermediate = w_up.shape()[0];
        let seq_len = x.shape()[0];
        let gate_raw = w_gate.as_slice().unwrap();
        let up_raw = w_up.as_slice().unwrap();

        let features = &self.feature_lists[layer];
        let k = features.len();

        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));
        let mut out = Array2::<f32>::zeros((seq_len, hidden));

        if k == 0 { return (out, full_activation); }

        // Pre-gather gate and up rows for the preselected features
        let mut gate_buf = vec![0.0f32; k * hidden];
        let mut up_buf = vec![0.0f32; k * hidden];
        for (i, &feat) in features.iter().enumerate() {
            let src = feat * hidden;
            gate_buf[i * hidden..(i + 1) * hidden].copy_from_slice(&gate_raw[src..src + hidden]);
            up_buf[i * hidden..(i + 1) * hidden].copy_from_slice(&up_raw[src..src + hidden]);
        }

        let gate_sub = ndarray::ArrayView2::from_shape((k, hidden), &gate_buf).unwrap();
        let up_sub = ndarray::ArrayView2::from_shape((k, hidden), &up_buf).unwrap();

        for s in 0..seq_len {
            let x_row = x.row(s);
            let gate_proj = gate_sub.dot(&x_row);
            let up_proj = up_sub.dot(&x_row);

            for (i, &feat) in features.iter().enumerate() {
                let g = gate_proj[i];
                full_activation[[s, feat]] = (g * sigmoid(g)) * up_proj[i];
            }

            // Down projection via BLAS on sparse activation
            let act_row = full_activation.row(s);
            let out_vec = w_down.dot(&act_row);
            let mut out_row = out.row_mut(s);
            ndarray::Zip::from(&mut out_row).and(&out_vec).for_each(|o, &v| *o += v);
        }
        (out, full_activation)
    }
}
