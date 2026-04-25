//! Kernel-level bisect for the CPU/Metal LM-head divergence surfaced
//! by `test_logits_goldens` on tied-embedding models (Gemma 3 4B,
//! Gemma 4 31B).
//!
//! ## What we're testing
//!
//! The LM head goes through `index.lm_head_knn_backend` which has
//! three paths:
//!   1. `backend.q4_matvec` — Q4_0 weights × Q8 quantized query.
//!      Used when `lm_head_q4.bin` exists *or* `lm_head_q4_synth`
//!      was built from f16 embeddings (tied-embed Gemma path).
//!   2. `backend.f16_gemv` — f16 weights × f32 query (some vindexes).
//!   3. `backend.f32_gemv` / BLAS — f32 fallback.
//!
//! End-to-end goldens show CPU and Metal disagree on Gemma's top-5
//! next token, but agree on Llama 2 and Mistral. Per-stage parity
//! tests pass at `cos=1.0` through `down_out`, so the divergence is
//! in the LM-head step. Llama 2 / Mistral go through path 3 (f32
//! BLAS, kernel-equivalent on both backends — see
//! `f32_gemv_matches_ndarray_dot` and the vocab-scale test below);
//! Gemma's tied-embedding path goes through path 1 (Q4_0 + Q8),
//! which is where the divergence has to live.
//!
//! This file pins both paths at vocab scale:
//!
//! - `f32_gemv_cpu_vs_metal_at_vocab_scale` — confirms suspect (3)
//!   is **clean**: the f32 fallback agrees on top-5 + top-1 logit
//!   between CPU and Metal at K=262144 × hidden=2560.
//! - `q4_matvec_cpu_vs_metal_at_vocab_scale` — pins suspect (1):
//!   same Q4_0 weights + Q8 query on both backends. If this fails,
//!   the production Q4_0 matvec kernel disagrees between CPU NEON
//!   and Metal simdgroup shader at the LM-head shape, and that's
//!   the direct cause of the goldens divergence.
//!
//! Both allocate ~2.68 GB f32 + ~1.3 GB Q4_0; gated to keep casual
//! `cargo test` runs cheap.
//!
//! ```bash
//! LARQL_RUN_LM_HEAD_BISECT=1 \
//!   cargo test --release --features metal -p larql-compute \
//!     --test test_kernel_lm_head_gemv -- --nocapture
//! ```

extern crate blas_src;

#[path = "common/mod.rs"]
mod common;
use common::get_metal;

use larql_compute::{ComputeBackend, CpuBackend};
use ndarray::Array2;

fn run_enabled() -> bool {
    matches!(
        std::env::var("LARQL_RUN_LM_HEAD_BISECT").ok().as_deref(),
        Some("1") | Some("true")
    )
}

/// Synthesise a deterministic `[n, k]` matrix and a `[k]` query.
/// Values are scaled to land in the magnitude range f32_gemv sees in
/// production (LM-head logits typically run from ~10⁰ to 10³ depending
/// on the model and how tightly normalised its last hidden is).
fn synth_inputs(n: usize, k: usize) -> (Array2<f32>, Vec<f32>) {
    // Compact deterministic generator — no rand crate dependency.
    let mut w = Vec::with_capacity(n * k);
    for i in 0..n * k {
        let f = i as f32;
        w.push(((f * 0.0001).sin() + 0.3 * (f * 0.00037).cos()) * 0.05);
    }
    let w = Array2::from_shape_vec((n, k), w).unwrap();
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    (w, x)
}

fn top5(scores: &[f32]) -> [(u32, f32); 5] {
    let mut indexed: Vec<(u32, f32)> = scores.iter().copied().enumerate()
        .map(|(i, s)| (i as u32, s)).collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    std::array::from_fn(|i| indexed[i])
}

#[test]
fn f32_gemv_cpu_vs_metal_at_vocab_scale() {
    if !run_enabled() {
        eprintln!(
            "skip: LARQL_RUN_LM_HEAD_BISECT=1 not set. \
             This test allocates a ~2.68 GB f32 matrix; gated to keep \
             casual `cargo test` runs cheap."
        );
        return;
    }

    let metal = get_metal();
    metal.set_flop_threshold(1); // force GPU dispatch even for non-tiny

    // Gemma 3 4B tied-embedding LM head shape.
    let n = 262_144usize; // vocab
    let k = 2_560usize;   // hidden
    eprintln!("Synthesising W [{n}, {k}] = {:.2} GB and x [{k}]…",
        (n * k * 4) as f64 / 1e9);
    let (w, x) = synth_inputs(n, k);

    // CPU has no `f32_gemv` specialisation (returns `None`); production
    // `lm_head_topk` falls back to `matmul_transb` for the CPU path.
    // Mirror that fallback here so we're benching the *exact* code
    // each backend uses in production.
    let cpu_scores: Vec<f32> = match CpuBackend.f32_gemv(w.view(), &x) {
        Some(s) => s,
        None => {
            let q_row = ndarray::Array2::from_shape_vec((1, k), x.clone()).unwrap();
            CpuBackend.matmul_transb(q_row.view(), w.view()).row(0).to_vec()
        }
    };
    let metal_scores = metal.f32_gemv(w.view(), &x)
        .expect("Metal f32_gemv should dispatch above threshold");

    let cpu_top5 = top5(&cpu_scores);
    let metal_top5 = top5(&metal_scores);

    eprintln!("CPU   top-5: {:?}", cpu_top5);
    eprintln!("Metal top-5: {:?}", metal_top5);

    let cpu_top1 = cpu_top5[0];
    let metal_top1 = metal_top5[0];

    // Within-CPU vs within-Metal accumulation order can swap rank
    // within the top-5 by ULP noise — but the **set** must match,
    // and the top-1 logit value should match within 1e-3 absolute on
    // a 0.05-scale matrix. (Total dot-product range here is bounded
    // by Σ |w| * |x| ≈ 0.05 * 0.5 * 2560 ≈ 64.)
    let mut cpu_set: Vec<u32> = cpu_top5.iter().map(|t| t.0).collect();
    let mut metal_set: Vec<u32> = metal_top5.iter().map(|t| t.0).collect();
    cpu_set.sort_unstable();
    metal_set.sort_unstable();
    assert_eq!(
        cpu_set, metal_set,
        "f32_gemv top-5 sets diverge at vocab-scale K=262144 × hidden=2560 \
         (CPU vs Metal). This is the suspect for the open Gemma 3/4 \
         CPU/Metal LM-head divergence in `test_logits_goldens`. \
         If this fails, the Metal `f32_gemv` shader is the cause; if it \
         passes, the divergence is upstream (last-hidden-state differs)."
    );

    let logit_diff = (cpu_top1.1 - metal_top1.1).abs();
    let max_abs = cpu_scores.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    let rel = logit_diff / max_abs;
    assert!(
        rel < 1e-3,
        "top-1 logit diverges: cpu={:.6} metal={:.6} (rel={:.3e})",
        cpu_top1.1, metal_top1.1, rel,
    );

    eprintln!(
        "✓ f32_gemv vocab-scale CPU vs Metal: top-5 sets match, \
         top-1 logit Δ={:.3e} (rel {:.2e})",
        logit_diff, rel,
    );
}

/// Q4_0 + Q8 input matvec at the LM-head shape (vocab × hidden).
///
/// This is the path `lm_head_knn_backend` takes when the vindex has
/// either an `lm_head_q4.bin` file or a tied-embedding `lm_head_q4_synth`
/// built from f16 embeddings. CPU and Metal each implement
/// `q4_matvec(q4_data, q8_x, q8_scales, n, k)` independently — CPU
/// via the `larql-compute/src/csrc/q4_dot.c` ARM NEON kernel, Metal
/// via the `q4_matvec_v4` simdgroup shader. If the two kernels
/// disagree at vocab scale, every Q4_0 LM-head dispatch in
/// production will produce a different top-K on each backend.
#[test]
fn q4_matvec_cpu_vs_metal_at_vocab_scale() {
    if !run_enabled() {
        eprintln!(
            "skip: LARQL_RUN_LM_HEAD_BISECT=1 not set. \
             Allocates a ~2.68 GB f32 matrix + ~1.3 GB Q4_0; gated."
        );
        return;
    }

    let metal = get_metal();
    metal.set_flop_threshold(1);

    use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_to_q8};

    let n = 262_144usize;
    let k = 2_560usize;
    eprintln!("Synthesising W [{n}, {k}] f32 → Q4_0 + Q8 query…");
    let (w, x) = synth_inputs(n, k);

    let w_flat: &[f32] = w.as_slice().expect("synth produced contiguous Array2");
    let q4_data = quantize_q4_0(w_flat);
    let (q8_x_i8, q8_scales) = quantize_to_q8(&x);
    eprintln!(
        "  Q4 bytes: {:.2} GB, Q8 input: {} elements, scales: {} blocks",
        q4_data.len() as f64 / 1e9, q8_x_i8.len(), q8_scales.len(),
    );

    let cpu_scores = CpuBackend.q4_matvec(&q4_data, &q8_x_i8, &q8_scales, n, k)
        .expect("CpuBackend.q4_matvec should always return Some");
    let metal_scores = metal.q4_matvec(&q4_data, &q8_x_i8, &q8_scales, n, k)
        .expect("MetalBackend.q4_matvec should always return Some");

    let cpu_top5 = top5(&cpu_scores);
    let metal_top5 = top5(&metal_scores);
    eprintln!("CPU   top-5: {:?}", cpu_top5);
    eprintln!("Metal top-5: {:?}", metal_top5);

    let cpu_top1 = cpu_top5[0];
    let metal_top1 = metal_top5[0];

    let mut cpu_set: Vec<u32> = cpu_top5.iter().map(|t| t.0).collect();
    let mut metal_set: Vec<u32> = metal_top5.iter().map(|t| t.0).collect();
    cpu_set.sort_unstable();
    metal_set.sort_unstable();

    if cpu_set != metal_set {
        // Annotate with the per-token score on the *other* backend so
        // we can see how close the rankings actually are.
        let cpu_score_at = |id: u32| cpu_scores[id as usize];
        let metal_score_at = |id: u32| metal_scores[id as usize];
        eprintln!("\n  Score on CPU at IDs Metal returned:");
        for &(id, _s) in metal_top5.iter() {
            eprintln!("    id {id}: cpu={:.4} metal={:.4}", cpu_score_at(id), metal_score_at(id));
        }
        eprintln!("  Score on Metal at IDs CPU returned:");
        for &(id, _s) in cpu_top5.iter() {
            eprintln!("    id {id}: cpu={:.4} metal={:.4}", cpu_score_at(id), metal_score_at(id));
        }
    }

    assert_eq!(
        cpu_set, metal_set,
        "Q4_0 matvec top-5 sets diverge at vocab-scale (N=262144 × K=2560). \
         This is the DIRECT cause of the open Gemma 3/4 CPU/Metal LM-head \
         divergence in `test_logits_goldens`. CPU NEON kernel and Metal \
         simdgroup shader produce different top-5 token IDs for the same \
         Q4_0 weights × Q8 query."
    );

    let logit_diff = (cpu_top1.1 - metal_top1.1).abs();
    let max_abs = cpu_scores.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
    let rel = logit_diff / max_abs;
    assert!(
        rel < 1e-2,
        "Q4 top-1 logit diverges: cpu={:.6} metal={:.6} (rel={:.3e})",
        cpu_top1.1, metal_top1.1, rel,
    );

    eprintln!(
        "✓ Q4 matvec vocab-scale CPU vs Metal: top-5 sets match, \
         top-1 logit Δ={:.3e} (rel {:.2e})",
        logit_diff, rel,
    );
}
