//! Feed-forward network computation — trait-based with dense and sparse backends.

use ndarray::Array2;

use crate::forward::{add_bias, dot_f64};
use crate::model::ModelWeights;

// ── Trait ──

/// FFN backend trait. Defines how a single layer's FFN is computed.
/// Same interface, different implementations: dense matmul vs sparse top-K.
pub trait FfnBackend {
    /// Run the FFN for a given layer on the pre-FFN-normed residual.
    /// Input: (seq_len, hidden_size). Output: (seq_len, hidden_size).
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32>;

    /// Run FFN and also return the pre-down activation (for capture).
    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>);

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}

// ── Dense backend (original) ──

/// Dense FFN: full matrix multiply across all features.
/// SiLU(x @ gate.T) * (x @ up.T) @ down.T
/// This is the ground truth — identical to model inference.
pub struct WeightFfn<'a> {
    pub weights: &'a ModelWeights,
}

impl<'a> FfnBackend for WeightFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let arch = &*self.weights.arch;
        let w_up = self.weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();

        let activation = if arch.ffn_type() == larql_models::FfnType::Gated {
            let w_gate = self.weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
            let gate = dot_f64(x, w_gate);
            let up = dot_f64(x, w_up);
            match arch.activation() {
                larql_models::Activation::GeluTanh => gelu_tanh_gate_up(&gate, &up),
                _ => silu_gate_up(&gate, &up),
            }
        } else {
            // Standard (non-gated) FFN: activation(x @ up.T + bias)
            let mut projected = dot_f64(x, w_up);
            if let Some(bias) = arch.ffn_up_bias_key(layer).and_then(|k| self.weights.vectors.get(&k)) {
                add_bias(&mut projected, bias);
            }
            match arch.activation() {
                larql_models::Activation::GeluTanh => projected.mapv(gelu_tanh),
                larql_models::Activation::Gelu => projected.mapv(gelu_tanh),
                _ => projected.mapv(|v| v * sigmoid(v)),
            }
        };
        let mut out = dot_f64(&activation, w_down);
        if let Some(bias) = arch.ffn_down_bias_key(layer).and_then(|k| self.weights.vectors.get(&k)) {
            add_bias(&mut out, bias);
        }
        out
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let arch = &*self.weights.arch;
        let w_up = self.weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();

        let activation = if arch.ffn_type() == larql_models::FfnType::Gated {
            let w_gate = self.weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
            let gate = dot_f64(x, w_gate);
            let up = dot_f64(x, w_up);
            match arch.activation() {
                larql_models::Activation::GeluTanh => gelu_tanh_gate_up(&gate, &up),
                _ => silu_gate_up(&gate, &up),
            }
        } else {
            let mut projected = dot_f64(x, w_up);
            if let Some(bias) = arch.ffn_up_bias_key(layer).and_then(|k| self.weights.vectors.get(&k)) {
                add_bias(&mut projected, bias);
            }
            match arch.activation() {
                larql_models::Activation::GeluTanh => projected.mapv(gelu_tanh),
                larql_models::Activation::Gelu => projected.mapv(gelu_tanh),
                _ => projected.mapv(|v| v * sigmoid(v)),
            }
        };
        let mut out = dot_f64(&activation, w_down);
        if let Some(bias) = arch.ffn_down_bias_key(layer).and_then(|k| self.weights.vectors.get(&k)) {
            add_bias(&mut out, bias);
        }
        (out, activation)
    }

    fn name(&self) -> &str {
        "weights"
    }
}

// ── Sparse backend ──

/// Sparse FFN: compute all gate activations, but only project the top-K
/// features through up and down. Same weights, sparse access.
///
/// K=10240 (all features) = identical to WeightFfn.
/// K=100 = ~60x faster, only computes features that matter.
pub struct SparseFfn<'a> {
    pub weights: &'a ModelWeights,
    pub top_k: usize,
}

impl<'a> FfnBackend for SparseFfn<'a> {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        let arch = &*self.weights.arch;
        let w_gate = self.weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        let w_up = self.weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
        ffn_forward_sparse(x, w_gate, w_up, w_down, self.top_k)
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let arch = &*self.weights.arch;
        let w_gate = self.weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        let w_up = self.weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let w_down = self.weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
        // For activation capture with sparse, return the sparse activation
        let (out, act) = ffn_forward_sparse_with_activation(x, w_gate, w_up, w_down, self.top_k);
        (out, act)
    }

    fn name(&self) -> &str {
        "sparse"
    }
}

// ── Per-layer backend selection ──

/// Selects which FFN backend to use for each layer.
/// Enables mixing: e.g., dense for layers 0-25, sparse for 26-33.
pub struct LayerFfnRouter<'a> {
    backends: Vec<&'a dyn FfnBackend>,
    num_layers: usize,
}

impl<'a> LayerFfnRouter<'a> {
    /// All layers use the same backend.
    pub fn uniform(backend: &'a dyn FfnBackend, num_layers: usize) -> Self {
        Self {
            backends: vec![backend; num_layers],
            num_layers,
        }
    }

    /// Each layer gets its own backend.
    pub fn per_layer(backends: Vec<&'a dyn FfnBackend>) -> Self {
        let num_layers = backends.len();
        Self {
            backends,
            num_layers,
        }
    }

    pub fn get(&self, layer: usize) -> &dyn FfnBackend {
        if layer < self.num_layers {
            self.backends[layer]
        } else {
            self.backends[self.num_layers - 1]
        }
    }
}

// ── Highway backend (skip FFN, return zeros) ──

/// Highway FFN: returns zeros. For layers where the residual stream barely
/// turns (angular displacement < 0.04 rad), the FFN contribution is negligible.
/// Attention still runs — this only skips the feed-forward computation.
pub struct HighwayFfn;

impl FfnBackend for HighwayFfn {
    fn forward(&self, _layer: usize, x: &Array2<f32>) -> Array2<f32> {
        Array2::<f32>::zeros((x.shape()[0], x.shape()[1]))
    }

    fn forward_with_activation(&self, _layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        (
            Array2::<f32>::zeros((x.shape()[0], x.shape()[1])),
            Array2::<f32>::zeros((x.shape()[0], 1)),
        )
    }

    fn name(&self) -> &str {
        "highway"
    }
}

// ── Dense implementations ──

/// Full FFN forward: SiLU(x @ gate.T) * (x @ up.T) @ down.T
pub fn ffn_forward_dense(
    x: &Array2<f32>,
    w_gate: &Array2<f32>,
    w_up: &Array2<f32>,
    w_down: &Array2<f32>,
) -> Array2<f32> {
    let gate = x.dot(&w_gate.t());
    let up = x.dot(&w_up.t());
    let activation = silu_gate_up(&gate, &up);
    activation.dot(&w_down.t())
}

/// Full FFN forward, also returning the pre-down activation for capture.
pub fn ffn_forward_dense_with_activation(
    x: &Array2<f32>,
    w_gate: &Array2<f32>,
    w_up: &Array2<f32>,
    w_down: &Array2<f32>,
) -> (Array2<f32>, Array2<f32>) {
    let gate = x.dot(&w_gate.t());
    let up = x.dot(&w_up.t());
    let activation = silu_gate_up(&gate, &up);
    let out = activation.dot(&w_down.t());
    (out, activation)
}

// ── Sparse implementations ──

/// Sparse FFN: compute gate activations for all features, select top-K,
/// only compute up and down for those K features.
fn ffn_forward_sparse(
    x: &Array2<f32>,
    w_gate: &Array2<f32>,
    w_up: &Array2<f32>,
    w_down: &Array2<f32>,
    top_k: usize,
) -> Array2<f32> {
    let (out, _) = ffn_forward_sparse_with_activation(x, w_gate, w_up, w_down, top_k);
    out
}

fn ffn_forward_sparse_with_activation(
    x: &Array2<f32>,
    w_gate: &Array2<f32>,
    w_up: &Array2<f32>,
    w_down: &Array2<f32>,
    top_k: usize,
) -> (Array2<f32>, Array2<f32>) {
    let seq_len = x.shape()[0];
    let hidden = x.shape()[1];
    let intermediate = w_gate.shape()[0];
    let k = top_k.min(intermediate);

    // If K >= 80% of features, dense BLAS is faster than gather + sparse BLAS
    if k * 5 >= intermediate * 4 {
        return ffn_forward_dense_with_activation(x, w_gate, w_up, w_down);
    }

    // Step 1: gate activations — dense BLAS (need all to find top-K)
    let gate_proj = x.dot(&w_gate.t()); // (seq, intermediate)
    let gate_act = gate_proj.mapv(|v| v * sigmoid(v));

    // w_up is (intermediate, hidden) — row gather is contiguous
    let up_raw = w_up.as_slice().unwrap();

    // w_down is (hidden, intermediate). For the down projection, we compute:
    //   out = w_down @ sparse_act   where sparse_act has only K non-zeros.
    // Instead of gathering columns, we use sparse accumulation:
    //   out[j] = sum_i(act[i] * w_down[j, feat_i])
    // This avoids any transpose or column gather entirely.

    let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));
    let mut out = Array2::<f32>::zeros((seq_len, hidden));

    // Pre-allocate gather buffer for up rows — reused across positions
    let mut up_buf = vec![0.0f32; k * hidden];

    for s in 0..seq_len {
        // Top-K selection by gate magnitude
        let mut indexed: Vec<(usize, f32)> = gate_act.row(s).iter().copied().enumerate().collect();
        indexed.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        indexed.truncate(k);

        // Step 2: gather top-K up rows (contiguous memcpy)
        for (i, &(feat, _)) in indexed.iter().enumerate() {
            let src = feat * hidden;
            up_buf[i * hidden..(i + 1) * hidden].copy_from_slice(&up_raw[src..src + hidden]);
        }

        // up_proj = up_sub @ x[s] -> (K,)   [BLAS gemv]
        let up_sub = ndarray::ArrayView2::from_shape((k, hidden), &up_buf[..k * hidden]).unwrap();
        let x_row = x.row(s);
        let up_proj = up_sub.dot(&x_row);

        // activation = silu(gate) * up
        let mut act_buf = vec![0.0f32; k];
        for (i, &(_, gate_val)) in indexed.iter().enumerate() {
            act_buf[i] = gate_val * up_proj[i];
        }

        // Store sparse activations
        for (i, &(feat, _)) in indexed.iter().enumerate() {
            full_activation[[s, feat]] = act_buf[i];
        }

        // Step 3: down projection via dense BLAS gemv on the sparse activation vector.
        // The full_activation row has K non-zeros out of intermediate.
        // w_down @ full_activation[s] is a standard gemv — BLAS reads w_down row-by-row
        // (cache-friendly) and the zero entries cost multiply-by-zero, not strided access.
        let act_row = full_activation.row(s);
        let out_vec = w_down.dot(&act_row); // BLAS gemv: (hidden, intermediate) @ (intermediate,) = (hidden,)
        let mut out_row = out.row_mut(s);
        ndarray::Zip::from(&mut out_row)
            .and(&out_vec)
            .for_each(|o, &v| *o = v);
    }

    (out, full_activation)
}

// ── Backward-compatible free functions ──

/// SiLU(gate) * up — the gated FFN activation used in Gemma/Llama.
pub fn silu_gate_up(gate: &Array2<f32>, up: &Array2<f32>) -> Array2<f32> {
    let activated = gate.mapv(|v| v * sigmoid(v));
    &activated * up
}

/// Gated FFN with GELU tanh approximation: GELU_tanh(gate) * up.
pub fn gelu_tanh_gate_up(gate: &Array2<f32>, up: &Array2<f32>) -> Array2<f32> {
    let activated = gate.mapv(gelu_tanh);
    &activated * up
}

/// GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu_tanh(x: f32) -> f32 {
    let c = 0.7978845608f32; // sqrt(2/pi)
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// Backward-compatible alias for dense FFN forward.
pub fn ffn_forward(
    x: &Array2<f32>,
    w_gate: &Array2<f32>,
    w_up: &Array2<f32>,
    w_down: &Array2<f32>,
) -> Array2<f32> {
    ffn_forward_dense(x, w_gate, w_up, w_down)
}

/// Backward-compatible alias for dense FFN forward with activation.
pub fn ffn_forward_with_activation(
    x: &Array2<f32>,
    w_gate: &Array2<f32>,
    w_up: &Array2<f32>,
    w_down: &Array2<f32>,
) -> (Array2<f32>, Array2<f32>) {
    ffn_forward_dense_with_activation(x, w_gate, w_up, w_down)
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
