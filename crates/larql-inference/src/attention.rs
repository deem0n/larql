//! Attention computation — GQA, RoPE, causal masking.

use ndarray::Array2;

/// Apply Rotary Position Embeddings to Q or K.
/// Uses split-half pairing: (x[i], x[i + half_dim]).
/// This matches MLX traditional=False and HuggingFace's default.
/// x: (seq_len, num_heads * head_dim)
pub fn apply_rope(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let mut out = x.clone();

    let half_dim = head_dim / 2;
    let inv_freq: Vec<f64> = (0..half_dim)
        .map(|i| 1.0 / rope_base.powf(2.0 * i as f64 / head_dim as f64))
        .collect();

    for pos in 0..seq_len {
        for h in 0..num_heads {
            let offset = h * head_dim;
            for i in 0..half_dim {
                let theta = pos as f64 * inv_freq[i];
                let cos_t = theta.cos() as f32;
                let sin_t = theta.sin() as f32;

                let x0 = x[[pos, offset + i]];
                let x1 = x[[pos, offset + half_dim + i]];

                out[[pos, offset + i]] = x0 * cos_t - x1 * sin_t;
                out[[pos, offset + half_dim + i]] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    out
}

/// Per-head attention weights for the last token position.
/// `weights[head]` = vec of attention scores over all positions.
pub struct AttentionWeights {
    /// Per-head attention distribution for the last sequence position.
    /// `heads[h][j]` = attention weight from last token to position j.
    pub heads: Vec<Vec<f32>>,
}

/// Grouped-query attention with causal masking.
///
/// q: (seq, num_q * head_dim), k: (seq, num_kv * head_dim), v: same as k
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
) -> Array2<f32> {
    let (out, _) = gqa_attention_with_weights(q, k, v, num_q, head_dim, reps, scale, seq_len, false, None);
    out
}

/// GQA attention that optionally captures per-head attention weights for the last token.
/// `softcap`: if Some(cap), apply tanh(scores/cap)*cap before softmax (Gemma2).
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_weights(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    capture: bool,
    softcap: Option<f32>,
) -> (Array2<f32>, Option<AttentionWeights>) {
    let mut out = Array2::<f32>::zeros((seq_len, num_q * head_dim));
    let mut captured_heads: Vec<Vec<f32>> = if capture {
        Vec::with_capacity(num_q)
    } else {
        Vec::new()
    };

    let last_pos = seq_len - 1;
    let _scale_f = scale as f32;

    for h in 0..num_q {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        // Extract per-head Q and K slices, compute Q @ K^T in f64 for precision
        let q_head = q.slice(ndarray::s![.., q_off..q_off + head_dim]);
        let k_head = k.slice(ndarray::s![.., kv_off..kv_off + head_dim]);

        let q_f64 = q_head.mapv(|v| v as f64);
        let k_f64 = k_head.mapv(|v| v as f64);
        let scores_f64 = q_f64.dot(&k_f64.t()) * scale;
        let mut scores = scores_f64.mapv(|v| v as f32);

        // Softcapping: tanh(scores / cap) * cap (Gemma2)
        if let Some(cap) = softcap {
            scores.mapv_inplace(|v| (v / cap).tanh() * cap);
        }

        // Causal mask + softmax (f64 accumulation for precision)
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[[i, j]] = -1e9;
            }
            let max_val = scores.row(i).iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            for j in 0..seq_len {
                let v = ((scores[[i, j]] - max_val) as f64).exp();
                scores[[i, j]] = v as f32;
                sum += v;
            }
            let inv_sum = (1.0 / sum) as f32;
            for j in 0..seq_len {
                scores[[i, j]] *= inv_sum;
            }
        }

        // Capture last-token attention weights
        if capture {
            captured_heads.push(scores.row(last_pos).to_vec());
        }

        // Weighted sum: scores @ V_head via BLAS
        let v_head = v.slice(ndarray::s![.., kv_off..kv_off + head_dim]).to_owned();
        let attn_v = scores.dot(&v_head); // (seq, seq) @ (seq, hd) → (seq, hd)

        // Write back to output
        for i in 0..seq_len {
            for d in 0..head_dim {
                out[[i, q_off + d]] = attn_v[[i, d]];
            }
        }
    }

    let weights = if capture {
        Some(AttentionWeights {
            heads: captured_heads,
        })
    } else {
        None
    };

    (out, weights)
}
