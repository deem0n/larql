//! `larql compile` — AOT compilation of vindex patches to model weights.
//!
//! Takes a base model (safetensors) and a vindex with patches, produces a
//! modified safetensors checkpoint where the patches are baked into the
//! weights. The output runs in any inference engine without LARQL.

use std::collections::HashMap;
use std::path::PathBuf;

use clap::Args;
use ndarray::ArcArray2;

#[derive(Args)]
pub struct CompileArgs {
    /// Path to the base model (directory with safetensors, or HF model ID).
    #[arg(long)]
    base: PathBuf,

    /// Path to the vindex (with patches to compile).
    #[arg(long)]
    vindex: PathBuf,

    /// Output directory for the compiled model safetensors.
    #[arg(short, long)]
    output: PathBuf,

    /// Gate scale for compiled edges (default: 30.0).
    #[arg(long, default_value = "30.0")]
    gate_scale: f32,

    /// Alpha multiplier for write magnitude (default: 1.0).
    #[arg(long, default_value = "1.0")]
    alpha: f32,
}

pub fn run(args: CompileArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("LARQL AOT Compiler");
    eprintln!("  base model: {}", args.base.display());
    eprintln!("  vindex:     {}", args.vindex.display());
    eprintln!("  output:     {}", args.output.display());

    // ── Load base model ─────────────────────────────────────────
    eprintln!("\nLoading base model...");
    let weights = larql_models::loading::load_model_dir(&args.base)?;
    let arch = &weights.arch;
    let config = arch.config();
    eprintln!(
        "  {} layers, hidden={}, ffn={}",
        config.num_layers, config.hidden_size, config.intermediate_size
    );

    // Detect tensor naming convention
    let gate_pattern = detect_ffn_pattern(&weights.tensors, "gate");
    let up_pattern = detect_ffn_pattern(&weights.tensors, "up");
    let down_pattern = detect_ffn_pattern(&weights.tensors, "down");
    eprintln!("  gate pattern: {}", gate_pattern.replace("{}", "N"));
    eprintln!("  up pattern:   {}", up_pattern.replace("{}", "N"));
    eprintln!("  down pattern:  {}", down_pattern.replace("{}", "N"));

    // ── Load patches ─────────────────────────────────────────────
    eprintln!("\nLoading patches...");

    // The vindex path can be a directory (scan for .vlp files) or a single .vlp file
    let patch_files: Vec<PathBuf> = if args.vindex.is_file() {
        vec![args.vindex.clone()]
    } else {
        std::fs::read_dir(&args.vindex)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "vlp"))
        .collect()
    };

    let mut all_ops = Vec::new();
    for pf in &patch_files {
        let patch = larql_vindex::VindexPatch::load(pf)?;
        eprintln!("  patch: {} ({} ops)", pf.display(), patch.operations.len());
        all_ops.extend(patch.operations);
    }

    eprintln!("  total patch operations: {}", all_ops.len());
    if all_ops.is_empty() {
        eprintln!("  no patches found — nothing to compile");
        return Ok(());
    }

    // ── Apply patches to weight tensors ─────────────────────────
    eprintln!("\nCompiling patches into weights...");

    // Clone the tensors we need to modify (CoW via ArcArray)
    let mut modified_tensors: HashMap<String, ArcArray2<f32>> = HashMap::new();
    let mut n_compiled = 0;

    for op in &all_ops {
        match op {
            larql_vindex::PatchOp::Insert {
                layer, feature, gate_vector_b64, entity, target, down_meta, ..
            } => {
                let gate_vec = match gate_vector_b64 {
                    Some(b64) => decode_f32_b64(b64)?,
                    None => {
                        eprintln!("  skip: insert at L{}[{}] has no gate vector", layer, feature);
                        continue;
                    }
                };

                let gate_key = gate_pattern.replace("{}", &layer.to_string());
                let up_key = up_pattern.replace("{}", &layer.to_string());
                let down_key = down_pattern.replace("{}", &layer.to_string());

                let slot = *feature;
                let hidden = config.hidden_size;

                // Ensure all three tensors are cloned into modified_tensors
                ensure_cloned(&mut modified_tensors, &weights.tensors, &gate_key)?;
                ensure_cloned(&mut modified_tensors, &weights.tensors, &up_key)?;
                ensure_cloned(&mut modified_tensors, &weights.tensors, &down_key)?;

                // Read reference norms before modification
                let g_norm = row_norm(modified_tensors.get(&gate_key).unwrap(), slot);
                let u_norm = row_norm(modified_tensors.get(&up_key).unwrap(), slot);
                let d_norm = col_norm(modified_tensors.get(&down_key).unwrap(), slot);

                let gv_norm = vec_norm(&gate_vec);
                if gv_norm > 1e-8 {
                    // Write gate direction
                    let scale = g_norm * args.gate_scale / gv_norm;
                    let gt = modified_tensors.get_mut(&gate_key).unwrap();
                    for (j, &v) in gate_vec.iter().enumerate().take(hidden) {
                        gt.as_standard_layout();
                        gt[[slot, j]] = v * scale;
                    }

                    // Up gets the same direction
                    let u_scale = u_norm / gv_norm;
                    let ut = modified_tensors.get_mut(&up_key).unwrap();
                    for (j, &v) in gate_vec.iter().enumerate().take(hidden) {
                        ut[[slot, j]] = v * u_scale;
                    }
                }

                // Write down direction from embed table if target token known
                if let Some(ref dm) = down_meta {
                    let tid = dm.top_token_id as usize;
                    if tid < weights.embed.shape()[0] {
                        let emb_row: Vec<f32> = weights.embed.row(tid).to_vec();
                        let emb_norm = vec_norm(&emb_row);
                        let alpha = (d_norm / emb_norm.max(1e-8)) * args.alpha;
                        let dt = modified_tensors.get_mut(&down_key).unwrap();
                        for (j, &v) in emb_row.iter().enumerate().take(hidden) {
                            dt[[j, slot]] = v * alpha;
                        }
                    }
                }

                n_compiled += 1;
                eprintln!(
                    "  compiled: L{}[{}] {} → {} (gate ‖{:.3}‖, down ‖{:.3}‖)",
                    layer, slot, entity, target, gv_norm, d_norm
                );
            }
            _ => {
                // Delete and Update ops: handle in future
            }
        }
    }

    eprintln!("\n  {} edges compiled into weights", n_compiled);

    // ── Save modified safetensors ────────────────────────────────
    eprintln!("\nSaving compiled model...");
    std::fs::create_dir_all(&args.output)?;

    // Merge: start with all original tensors, override with modified ones
    let mut all_tensors = weights.tensors.clone();
    for (k, v) in modified_tensors {
        all_tensors.insert(k, v);
    }

    // Write as safetensors (may need to split into shards for large models)
    let output_file = args.output.join("model.safetensors");
    write_safetensors(&all_tensors, &output_file)?;

    let file_size = std::fs::metadata(&output_file)?.len();
    eprintln!(
        "  saved: {} ({:.1} GB)",
        output_file.display(),
        file_size as f64 / 1e9
    );

    // Copy config.json and tokenizer files from base
    copy_model_config(&args.base, &args.output);

    eprintln!("\nDone. The compiled model runs in any inference engine:");
    eprintln!("  transformers: AutoModelForCausalLM.from_pretrained(\"{}\")", args.output.display());
    eprintln!("  ollama:       convert to GGUF, then `ollama create`");

    Ok(())
}

// ─── Helpers ────────────────────────────────────────────────────

fn detect_ffn_pattern(tensors: &HashMap<String, ArcArray2<f32>>, component: &str) -> String {
    let patterns = match component {
        "gate" => &["model.layers.{}.mlp.gate_proj.weight",
                     "layers.{}.ffn.gate.weight",
                     "model.layers.{}.feed_forward.gate_proj.weight"][..],
        "up" => &["model.layers.{}.mlp.up_proj.weight",
                   "layers.{}.ffn.up.weight",
                   "model.layers.{}.feed_forward.up_proj.weight"][..],
        "down" => &["model.layers.{}.mlp.down_proj.weight",
                     "layers.{}.ffn.down.weight",
                     "model.layers.{}.feed_forward.down_proj.weight"][..],
        _ => &[][..],
    };

    for pat in patterns {
        let test = pat.replace("{}", "0");
        if tensors.contains_key(&test) {
            return pat.to_string();
        }
    }

    // Fallback: search all tensor names
    let search = match component {
        "gate" => "gate",
        "up" => "up",
        "down" => "down",
        _ => "",
    };
    for key in tensors.keys() {
        if key.contains(search) && key.contains(".0.") {
            let pattern = key.replace(".0.", ".{}.");
            return pattern;
        }
    }

    format!("model.layers.{{}}.mlp.{}_proj.weight", component)
}

fn decode_f32_b64(b64: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD.decode(b64)?;
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(floats)
}

fn ensure_cloned(
    modified: &mut HashMap<String, ArcArray2<f32>>,
    originals: &HashMap<String, ArcArray2<f32>>,
    key: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if !modified.contains_key(key) {
        let original = originals.get(key)
            .ok_or_else(|| format!("tensor not found: {}", key))?;
        modified.insert(key.to_string(), original.to_owned().into());
    }
    Ok(())
}

fn row_norm(tensor: &ArcArray2<f32>, row: usize) -> f32 {
    let r = tensor.row(row);
    r.dot(&r).sqrt()
}

fn col_norm(tensor: &ArcArray2<f32>, col: usize) -> f32 {
    let c = tensor.column(col);
    c.dot(&c).sqrt()
}

fn vec_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn write_safetensors(
    tensors: &HashMap<String, ArcArray2<f32>>,
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use safetensors::tensor::{serialize, TensorView};

    // Convert to byte buffers
    let mut byte_bufs: HashMap<String, Vec<u8>> = HashMap::new();
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

    for (name, arr) in tensors {
        let shape = arr.shape().to_vec();
        let bytes: Vec<u8> = arr.iter().flat_map(|f| f.to_le_bytes()).collect();
        byte_bufs.insert(name.clone(), bytes);
        shapes.insert(name.clone(), shape);
    }

    // Create TensorViews
    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    for (name, bytes) in &byte_bufs {
        let shape = &shapes[name];
        views.insert(
            name.clone(),
            TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)?,
        );
    }

    let serialized = serialize(&views, &None)?;
    std::fs::write(path, serialized)?;
    Ok(())
}

fn copy_model_config(base: &std::path::Path, output: &std::path::Path) {
    for name in &["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json"] {
        let src = base.join(name);
        if src.exists() {
            let _ = std::fs::copy(&src, output.join(name));
        }
    }
}
