//! Full transformer forward pass.
//!
//! Runs tokens through embedding → layers → final norm → logits.
//! Uses the ModelArchitecture trait for model-specific behavior
//! and FfnBackend trait for swappable FFN computation.

use ndarray::Array2;

use crate::attention::{apply_rope, gqa_attention_with_weights, AttentionWeights};
use crate::ffn::{FfnBackend, LayerFfnRouter, WeightFfn};
use crate::model::ModelWeights;
use larql_models::NormType;
use crate::residual::{rms_norm, rms_norm_heads};

/// Per-head attention pattern for the last token at one layer.
pub struct LayerAttentionCapture {
    pub layer: usize,
    /// Per-head attention weights for the last token.
    /// `heads[h][j]` = how much the last token attends to position j.
    pub weights: AttentionWeights,
}

/// Result of a forward trace — residuals and optional sparse activations.
pub struct TraceResult {
    /// (layer, residual_vector) for each capture layer.
    pub residuals: Vec<(usize, Vec<f32>)>,
    /// (layer, top-K (feature_index, activation_magnitude)) for each capture layer.
    /// Only populated if capture_activations=true.
    pub activations: Vec<(usize, Vec<(usize, f32)>)>,
    /// Per-layer attention weight captures. Only populated if capture_attention=true.
    pub attention: Vec<LayerAttentionCapture>,
}

/// Prediction result from a full forward pass.
pub struct PredictResult {
    /// Top-k predicted tokens as (token_string, probability).
    pub predictions: Vec<(String, f64)>,
}

/// Per-layer computation strategy.
pub enum LayerMode<'a> {
    /// Run full attention + FFN with the given backend.
    Compute(&'a dyn FfnBackend),
    /// Skip the layer entirely — just multiply the hidden state by a scalar gain.
    /// gain = norm[L+1] / norm[L] from calibration data.
    ScalarGain(f32),
    /// Run attention but skip FFN (return zeros from FFN).
    /// Attention still routes information; FFN contribution is dropped.
    AttentionOnly,
}

/// Compute x @ w.T with f64 accumulation for precision.
/// This matches MLX Metal kernel behavior which uses higher-precision accumulation.
/// Apply the appropriate norm (RMSNorm or LayerNorm) based on architecture.
fn apply_norm(
    weights: &ModelWeights,
    x: &Array2<f32>,
    weight_key: &str,
    norm_offset: f32,
) -> Array2<f32> {
    match weights.arch.norm_type() {
        NormType::LayerNorm => {
            let bias_key = weight_key.replace(".weight", ".bias");
            crate::residual::layer_norm(
                x,
                weights.vectors.get(weight_key),
                weights.vectors.get(&bias_key),
            )
        }
        _ => rms_norm(x, weights.vectors.get(weight_key), norm_offset),
    }
}

/// Compute x @ w.T using f32 BLAS.
pub fn dot_f64(x: &Array2<f32>, w: &Array2<f32>) -> Array2<f32> {
    x.dot(&w.t())
}

// Temporary debug: print per-layer norms when LARQL_DEBUG is set
#[allow(dead_code)]
pub(crate) fn debug_norm(label: &str, h: &Array2<f32>) {
    if std::env::var("LARQL_DEBUG").is_ok() {
        let last = h.row(h.shape()[0] - 1);
        let norm: f32 = last.iter().map(|v| v * v).sum::<f32>().sqrt();
        let s = last.as_slice().unwrap();
        eprintln!("[dbg] {label}: norm={norm:.6} last[:4]=[{:.8},{:.8},{:.8},{:.8}]", s[0], s[1], s[2], s[3]);
    }
}

/// Add a 1D bias vector to each row of a 2D matrix.
pub fn add_bias(x: &mut Array2<f32>, bias: &[f32]) {
    let cols = x.shape()[1];
    let n = cols.min(bias.len());
    for mut row in x.rows_mut() {
        for j in 0..n {
            row[j] += bias[j];
        }
    }
}

/// Embed token IDs with architecture-specific scaling.
fn embed_tokens(weights: &ModelWeights, token_ids: &[u32]) -> Array2<f32> {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let scale = weights.arch.embed_scale();

    let mut h = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &tok_id) in token_ids.iter().enumerate() {
        let row = weights.embed.row(tok_id as usize);
        for j in 0..hidden {
            h[[i, j]] = row[j] * scale;
        }
    }
    h
}

/// Public wrapper for run_attention (used by CachedFfn calibration).
pub fn run_attention_public(weights: &ModelWeights, h: &Array2<f32>, layer: usize) -> Option<Array2<f32>> {
    run_attention(weights, h, layer)
}

/// Run attention for a single layer. Returns the post-attention residual.
fn run_attention(weights: &ModelWeights, h: &Array2<f32>, layer: usize) -> Option<Array2<f32>> {
    let (h_post_attn, _) = run_attention_inner(weights, h, layer, false)?;
    Some(h_post_attn)
}

/// Run attention with optional per-head weight capture.
fn run_attention_inner(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
) -> Option<(Array2<f32>, Option<AttentionWeights>)> {
    let arch = &*weights.arch;
    let head_dim = weights.head_dim;
    let num_q = weights.num_q_heads;
    let num_kv = weights.num_kv_heads;
    let reps = num_q / num_kv;
    let scale = if arch.attention_multiplier() != 1.0 {
        // Granite: attention_multiplier replaces the default 1/sqrt(head_dim) scale
        arch.attention_multiplier() as f64
    } else {
        arch.attention_scale()
    };
    let seq_len = h.shape()[0];
    let norm_offset = arch.norm_weight_offset();

    let h_norm = apply_norm(weights, h, &arch.input_layernorm_key(layer), norm_offset);

    let w_q = weights.tensors.get(&arch.attn_q_key(layer))?;
    let w_k = weights.tensors.get(&arch.attn_k_key(layer)).unwrap();
    let w_v = weights.tensors.get(&arch.attn_v_key(layer)).unwrap();
    let w_o = weights.tensors.get(&arch.attn_o_key(layer)).unwrap();

    // f64 accumulation for linear projections to match MLX Metal precision
    let mut q_full = dot_f64(&h_norm, w_q);
    let mut k_full = dot_f64(&h_norm, w_k);
    let mut v_full = dot_f64(&h_norm, w_v);

    // Add attention bias if present (e.g., Qwen2/2.5)
    if let Some(bias) = arch.attn_q_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut q_full, bias);
    }
    if let Some(bias) = arch.attn_k_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut k_full, bias);
    }
    if let Some(bias) = arch.attn_v_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut v_full, bias);
    }

    let qk_offset = weights.arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 { qk_offset } else { norm_offset };
    let q_normed = match arch
        .attn_q_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, qk_norm_off),
        None => q_full,
    };
    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&k_full, norm_w, num_kv, head_dim, qk_norm_off),
        None => k_full,
    };

    let layer_rope_base = arch.rope_base_for_layer(layer);
    let q_rope = apply_rope(&q_normed, num_q, head_dim, layer_rope_base);
    let k_rope = apply_rope(&k_normed, num_kv, head_dim, layer_rope_base);

    let softcap = arch.attn_logit_softcapping();
    let (attn_out, attn_weights) = gqa_attention_with_weights(
        &q_rope, &k_rope, &v_full, num_q, head_dim, reps, scale, seq_len, capture_attention, softcap,
    );
    let mut attn_projected = dot_f64(&attn_out, w_o);
    if let Some(bias) = arch.attn_o_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut attn_projected, bias);
    }

    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = apply_norm(weights, &attn_projected, &arch.post_attention_layernorm_key(layer), norm_offset);
        if res_mult != 1.0 {
            h + &(&normed * res_mult)
        } else {
            h + &normed
        }
    } else if res_mult != 1.0 {
        h + &(&attn_projected * res_mult)
    } else {
        h + &attn_projected
    };

    Some((h_post_attn, attn_weights))
}

/// Run FFN for a single layer using the given backend. Returns the post-FFN residual.
fn run_ffn(
    weights: &ModelWeights,
    h_post_attn: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
) -> (Array2<f32>, Option<Array2<f32>>) {
    let norm_offset = weights.arch.norm_weight_offset();
    let arch = &*weights.arch;

    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };
    let h_ffn = match pre_ffn_key {
        Some(key) => apply_norm(weights, h_post_attn, &key, norm_offset),
        None => rms_norm(h_post_attn, None, norm_offset),
    };

    let (ffn_out, activation) = if capture_activation {
        let (out, act) = ffn.forward_with_activation(layer, &h_ffn);
        (out, Some(act))
    } else {
        (ffn.forward(layer, &h_ffn), None)
    };

    let res_mult = arch.residual_multiplier();
    let h_out = if arch.has_post_norms() {
        let normed = match arch.post_feedforward_layernorm_key(layer) {
            Some(key) => apply_norm(weights, &ffn_out, &key, norm_offset),
            None => rms_norm(&ffn_out, None, norm_offset),
        };
        if res_mult != 1.0 {
            h_post_attn + &(&normed * res_mult)
        } else {
            h_post_attn + &normed
        }
    } else if res_mult != 1.0 {
        h_post_attn + &(&ffn_out * res_mult)
    } else {
        h_post_attn + &ffn_out
    };

    (h_out, activation)
}

/// Run a single transformer layer with the given FFN backend.
fn run_layer_with_ffn(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
) -> Option<(Array2<f32>, Option<Array2<f32>>)> {
    let h_post_attn = run_attention(weights, h, layer)?;
    let (h_out, activation) = run_ffn(weights, &h_post_attn, layer, ffn, capture_activation);
    Some((h_out, activation))
}

/// Run a single transformer layer, optionally capturing attention weights.
fn run_layer_with_capture(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
    capture_attention: bool,
) -> Option<(Array2<f32>, Option<Array2<f32>>, Option<AttentionWeights>)> {
    let (h_post_attn, attn_weights) = run_attention_inner(weights, h, layer, capture_attention)?;
    let (h_out, activation) = run_ffn(weights, &h_post_attn, layer, ffn, capture_activation);
    Some((h_out, activation, attn_weights))
}

/// Project the final hidden state to logits and return top-k predictions.
fn logits_to_predictions(
    weights: &ModelWeights,
    h: &Array2<f32>,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
) -> PredictResult {
    let seq_len = h.shape()[0];
    let norm_offset = weights.arch.norm_weight_offset();

    let h_final = apply_norm(weights, h, weights.arch.final_norm_key(), norm_offset);

    let last = h_final.row(seq_len - 1);
    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let mut logits: Vec<f32> = Vec::with_capacity(weights.vocab_size);
    for tok_id in 0..weights.vocab_size {
        let lm_row = weights.lm_head.row(tok_id);
        let dot: f64 = last
            .iter()
            .zip(lm_row.iter())
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        let mut logit = (dot / logits_scale as f64) as f32;
        if let Some(cap) = final_softcap {
            logit = (logit / cap).tanh() * cap;
        }
        logits.push(logit);
    }

    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = logits
        .iter()
        .map(|l| ((l - max_logit) as f64).exp())
        .sum();
    let probs: Vec<f32> = logits
        .iter()
        .map(|l| (((l - max_logit) as f64).exp() / exp_sum) as f32)
        .collect();

    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    let k = top_k.min(indexed.len());
    indexed.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let predictions = indexed
        .into_iter()
        .filter_map(|(idx, prob)| {
            tokenizer
                .decode(&[idx as u32], true)
                .ok()
                .map(|s| (s.trim().to_string(), prob as f64))
        })
        .collect();

    PredictResult { predictions }
}

// ── Public API ──

/// Run a forward pass through layers 0..=stop_layer and return the full
/// hidden state matrix (seq_len, hidden_size) at that layer.
/// This is the complete residual stream — all positions, not just last token.
pub fn forward_to_layer(
    weights: &ModelWeights,
    token_ids: &[u32],
    stop_layer: usize,
) -> Array2<f32> {
    let ffn = WeightFfn { weights };
    let mut h = embed_tokens(weights, token_ids);

    for layer in 0..=stop_layer {
        h = match run_layer_with_ffn(weights, &h, layer, &ffn, false) {
            Some((h_new, _)) => h_new,
            None => continue,
        };
    }
    h
}

/// Run a forward pass through layers 0..=max_layer and return the
/// last-token residual at each requested capture layer.
pub fn capture_residuals(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
) -> Vec<(usize, Vec<f32>)> {
    let trace = trace_forward(weights, token_ids, capture_layers, false, 0);
    trace.residuals
}

/// Run a forward pass and capture both residuals and sparse activations.
pub fn trace_forward(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    capture_activations: bool,
    activation_top_k: usize,
) -> TraceResult {
    let ffn = WeightFfn { weights };
    trace_forward_with_ffn(
        weights,
        token_ids,
        capture_layers,
        capture_activations,
        activation_top_k,
        &ffn,
    )
}

/// Run a forward pass with a custom FFN backend.
pub fn trace_forward_with_ffn(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    capture_activations: bool,
    activation_top_k: usize,
    ffn: &dyn FfnBackend,
) -> TraceResult {
    trace_forward_full(
        weights, token_ids, capture_layers, capture_activations,
        activation_top_k, false, ffn,
    )
}

/// Run a forward pass capturing residuals, activations, and optionally attention weights.
pub fn trace_forward_full(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    capture_activations: bool,
    activation_top_k: usize,
    capture_attention: bool,
    ffn: &dyn FfnBackend,
) -> TraceResult {
    let seq_len = token_ids.len();
    let max_layer = *capture_layers.iter().max().unwrap_or(&0);

    let mut h = embed_tokens(weights, token_ids);
    let mut results = Vec::new();
    let mut activations: Vec<(usize, Vec<(usize, f32)>)> = Vec::new();
    let mut attention_captures: Vec<LayerAttentionCapture> = Vec::new();

    for layer in 0..=max_layer {
        let is_capture_layer = capture_layers.contains(&layer);
        let need_activation = capture_activations && is_capture_layer;
        let need_attention = capture_attention && is_capture_layer;

        let (h_new, activation, attn_weights) =
            match run_layer_with_capture(weights, &h, layer, ffn, need_activation, need_attention) {
                Some(result) => result,
                None => continue,
            };
        h = h_new;

        if is_capture_layer {
            let last_row = h.row(seq_len - 1);
            results.push((layer, last_row.to_vec()));

            if let Some(act) = activation {
                let act_row = act.row(seq_len - 1);
                let mut indexed: Vec<(usize, f32)> = act_row.iter().copied().enumerate().collect();
                indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                indexed.truncate(activation_top_k);
                activations.push((layer, indexed));
            }

            if let Some(weights) = attn_weights {
                attention_captures.push(LayerAttentionCapture {
                    layer,
                    weights,
                });
            }
        }
    }

    TraceResult {
        residuals: results,
        activations,
        attention: attention_captures,
    }
}

/// Run a full forward pass and return the top-k next token predictions.
/// Uses dense WeightFfn (ground truth).
pub fn predict(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
) -> PredictResult {
    let ffn = WeightFfn { weights };
    predict_with_ffn(weights, tokenizer, token_ids, top_k, &ffn)
}

/// Run a full forward pass with a custom FFN backend for all layers.
pub fn predict_with_ffn(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    ffn: &dyn FfnBackend,
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);

    for layer in 0..num_layers {
        h = match run_layer_with_ffn(weights, &h, layer, ffn, false) {
            Some((h_new, _)) => h_new,
            None => continue,
        };
    }

    logits_to_predictions(weights, &h, tokenizer, top_k)
}

/// Run a full forward pass with per-layer FFN backend selection.
pub fn predict_with_router(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    router: &LayerFfnRouter,
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);

    for layer in 0..num_layers {
        let ffn = router.get(layer);
        h = match run_layer_with_ffn(weights, &h, layer, ffn, false) {
            Some((h_new, _)) => h_new,
            None => continue,
        };
    }

    logits_to_predictions(weights, &h, tokenizer, top_k)
}

/// Run a forward pass with per-layer strategy: full compute or scalar gain bypass.
pub fn predict_with_strategy(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    strategy: &[LayerMode],
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);

    for layer in 0..num_layers {
        match &strategy[layer] {
            LayerMode::Compute(ffn) => {
                h = match run_layer_with_ffn(weights, &h, layer, *ffn, false) {
                    Some((h_new, _)) => h_new,
                    None => continue,
                };
            }
            LayerMode::ScalarGain(gain) => {
                h *= *gain;
            }
            LayerMode::AttentionOnly => {
                // Run attention but skip FFN — residual gets attention contribution only.
                if let Some(h_post_attn) = run_attention(weights, &h, layer) {
                    h = h_post_attn;
                }
            }
        }
    }

    logits_to_predictions(weights, &h, tokenizer, top_k)
}

/// Resume a forward pass from a pre-computed hidden state at a given start layer.
/// Runs layers start_layer..num_layers, then projects to logits.
/// The hidden state `h` should be shaped (seq_len, hidden_size).
pub fn predict_from_hidden(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    h_init: &Array2<f32>,
    start_layer: usize,
    top_k: usize,
) -> PredictResult {
    let ffn = WeightFfn { weights };
    predict_from_hidden_with_ffn(weights, tokenizer, h_init, start_layer, top_k, &ffn)
}

/// Resume a forward pass from a pre-computed hidden state with a custom FFN backend.
pub fn predict_from_hidden_with_ffn(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    h_init: &Array2<f32>,
    start_layer: usize,
    top_k: usize,
    ffn: &dyn FfnBackend,
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = h_init.clone();

    for layer in start_layer..num_layers {
        h = match run_layer_with_ffn(weights, &h, layer, ffn, false) {
            Some((h_new, _)) => h_new,
            None => continue,
        };
    }

    logits_to_predictions(weights, &h, tokenizer, top_k)
}

/// Calibrate scalar gains from a forward pass: compute norm[L+1] / norm[L] at each layer.
pub fn calibrate_scalar_gains(
    weights: &ModelWeights,
    token_ids: &[u32],
) -> Vec<f32> {
    let all_layers: Vec<usize> = (0..weights.num_layers).collect();
    let trace = trace_forward(weights, token_ids, &all_layers, false, 0);

    let mut gains = Vec::with_capacity(weights.num_layers);
    for i in 0..trace.residuals.len() {
        let norm_curr: f32 = trace.residuals[i].1.iter().map(|x| x * x).sum::<f32>().sqrt();
        if i + 1 < trace.residuals.len() {
            let norm_next: f32 = trace.residuals[i + 1].1.iter().map(|x| x * x).sum::<f32>().sqrt();
            gains.push(if norm_curr > 1e-12 { norm_next / norm_curr } else { 1.0 });
        } else {
            gains.push(1.0);
        }
    }
    gains
}
