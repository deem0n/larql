//! Autoregressive generation via a sharded expert grid.
//!
//! Uses the Metal pipeline for attention + dense FFN (same as normal `generate`),
//! but intercepts the MoE expert block per layer via a callback that dispatches
//! to remote shards over HTTP instead of calling `cpu_moe_forward` locally.
//!
//! The hook: `ComputeBackend::decode_token_with_moe(layers, x, ..., moe_fn)`
//! where `moe_fn(layer, h_post_attn) -> Vec<f32>` calls
//! `RemoteMoeBackend::forward_moe`.
//!
//! # Diagnostics
//!
//! Set `SKIP_MOE=1` to zero out the expert block on every decode step.
//! This isolates whether errors come from remote dispatch vs. dense FFN.

use larql_compute::prelude::*;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;

use crate::ffn::moe_remote::{MoeRouterWeights, RemoteMoeError};
use crate::ffn::RemoteMoeBackend;
use crate::forward::{apply_norm, embed_tokens_pub};
use crate::layer_graph::generate::lm_head_topk as lm_topk;
use crate::layer_graph::pipeline_layer::build_pipeline_layers;

/// Build `MoeRouterWeights` for one layer from the model's vector store.
/// Returns None if the required router projection is absent.
fn build_router<'a>(
    weights: &'a ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Option<MoeRouterWeights<'a>> {
    let router_proj_key = arch.moe_router_key(layer)?;
    let router_proj = weights.vectors.get(&router_proj_key)?.as_slice();
    let sl = |k: Option<String>| -> &'a [f32] {
        k.and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    };
    Some(MoeRouterWeights {
        router_proj,
        router_scale:            sl(arch.moe_router_scale_key(layer)),
        router_per_expert_scale: sl(arch.moe_router_per_expert_scale_key(layer)),
        router_norm:             sl(arch.moe_router_norm_key(layer)),
        router_norm_parameter_free: arch.moe_router_norm_parameter_free(),
        router_input_scalar: arch.moe_router_input_scalar().unwrap_or(1.0),
        pre_experts_norm:  sl(arch.moe_pre_experts_norm_key(layer)),
        post_experts_norm: sl(arch.moe_post_experts_norm_key(layer)),
        num_experts: arch.num_experts(),
        top_k:       arch.num_experts_per_token(),
    })
}

#[derive(Debug)]
pub struct GridGenerateResult {
    pub tokens: Vec<String>,
    pub decode_ms: Vec<f64>,
}

/// Greedy autoregressive generation through a remote-expert grid.
///
/// Requires a Metal (or Q4-capable) backend — attention and dense FFN run on
/// the GPU exactly as in the normal `generate()` path.  Expert blocks are
/// dispatched to `remote` instead of running locally.
pub fn generate_with_remote_moe(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    index: &VectorIndex,
    remote: &RemoteMoeBackend,
    backend: &dyn ComputeBackend,
) -> Result<GridGenerateResult, RemoteMoeError> {
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let eps = arch.norm_eps();
    let hidden = weights.hidden_size;
    let num_layers = weights.num_layers;

    // ── Build pipeline layers (same as generate()) ────────────────────────────
    let gate_index: &dyn larql_vindex::GateIndex = index;
    let q4_ffn = gate_index
        .interleaved_q4k_mmap_ref()
        .or_else(|| gate_index.interleaved_q4_mmap_ref())
        .ok_or_else(|| {
            RemoteMoeError::BadResponse("no interleaved Q4 FFN mmap in vindex".into())
        })?;
    let ffn_is_q4k = gate_index.interleaved_q4k_mmap_ref().is_some();

    let intermediate = gate_index.num_features(0);
    let q4_ffn_per_matrix = if ffn_is_q4k {
        (intermediate * hidden).div_ceil(256) * 144
    } else {
        intermediate * hidden / 32 * 18
    };
    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };

    let layers = build_pipeline_layers(
        weights,
        index,
        0..num_layers,
        q4_ffn,
        q4_ffn_per_matrix,
        ffn_format,
    );

    let q_dim = weights.num_q_heads * weights.head_dim;
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    let rope = arch.rope_base_for_layer(0) as f32;

    // ── Open gRPC streams (one pair for the entire generation) ───────────────
    //
    // For gRPC shards (`grpc://` URLs), we open one bidirectional stream per
    // shard and reuse it for every layer of every token (prefill + decode).
    // This eliminates the per-layer connection setup cost: each layer pays only
    // the cost of one proto frame exchange on an existing HTTP/2 connection
    // (~0.5ms) instead of ~12ms for a new unary call.
    //
    // For HTTP shards, `open_streams` returns an empty vec and we fall back to
    // `forward_moe` (per-layer HTTP calls, as before).
    let mut streams: Vec<crate::ffn::moe_remote::ShardStream> =
        if remote.has_grpc_shards() {
            remote.open_streams().unwrap_or_default()
        } else {
            vec![]
        };

    // ── Prefill ───────────────────────────────────────────────────────────────
    //
    // Run one `decode_token_with_moe` per prompt token rather than `prefill_q4`.
    // `prefill_q4` does not correctly apply MoE experts for hybrid-MoE post-norm
    // models (Gemma 4 26B-A4B), so the first-token prediction and subsequent KV
    // cache entries are wrong.  Sequential decode builds the KV cache correctly
    // — each token processes with the proper remote expert contribution.
    backend.reset_kv_cache();
    {
        let kv_shapes: Vec<(usize, usize)> = (0..num_layers)
            .map(|l| (arch.num_kv_heads_for_layer(l), arch.head_dim_for_layer(l)))
            .collect();
        backend.preallocate_kv_cache_per_layer(&kv_shapes, 4096);
    }

    let skip_moe = std::env::var("SKIP_MOE").is_ok();
    let mut last_hidden_vec: Vec<f32> = vec![0.0f32; hidden];
    let mut current_ids = prompt_ids.clone();

    for &tok_id in &prompt_ids {
        let tok_embed = embed_tokens_pub(weights, &[tok_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();

        let mut step_error: Option<RemoteMoeError> = None;
        let mut moe_fn = |layer: usize, h_post_attn: &[f32]| -> Vec<f32> {
            if skip_moe { return vec![0.0f32; hidden]; }
            if step_error.is_some() { return vec![0.0f32; hidden]; }
            let router = match build_router(weights, arch, layer) {
                Some(r) => r,
                None => return vec![0.0f32; hidden],
            };
            let result = if streams.is_empty() {
                remote.forward_moe(layer, h_post_attn, &router, norm_offset, eps)
            } else {
                remote.forward_moe_stream(layer, h_post_attn, &router, &mut streams, norm_offset, eps)
            };
            match result {
                Ok(out) => out,
                Err(e) => { step_error = Some(e); vec![0.0f32; hidden] }
            }
        };

        let h = backend.decode_token_with_moe(
            &layers, &x_tok, hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope, &mut moe_fn,
        );
        if let Some(err) = step_error { return Err(err); }
        last_hidden_vec = h.ok_or_else(|| {
            RemoteMoeError::BadResponse("decode_token_with_moe returned None during prefill".into())
        })?;
    }

    // ── Decode loop ───────────────────────────────────────────────────────────
    let mut tokens = Vec::new();
    let mut decode_ms = Vec::new();

    // First token from the (correct) prefill output.
    let prefill_h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
        .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
    let h_norm0 = apply_norm(weights, &prefill_h_arr, arch.final_norm_key(), norm_offset);
    let last0 = h_norm0.row(0).to_owned();
    let first_id = lm_topk(index, weights, &last0, 1, backend)
        .into_iter()
        .next()
        .map(|(id, _)| id)
        .unwrap_or(0);

    let first_tok = crate::tokenizer::decode_token(tokenizer, first_id)
        .unwrap_or_else(|| format!("<{first_id}>"));
    tokens.push(first_tok);
    current_ids.push(first_id);
    let first_is_eos = crate::vindex::is_end_of_turn(
        crate::tokenizer::decode_token(tokenizer, first_id)
            .unwrap_or_default()
            .trim(),
    );
    if first_is_eos || tokens.len() >= max_tokens {
        return Ok(GridGenerateResult {
            tokens,
            decode_ms: vec![0.0],
        });
    }

    for _step in 0..max_tokens.saturating_sub(1) {
        let t0 = std::time::Instant::now();
        let next_input_id = *current_ids.last().unwrap();

        // Embed next token.
        let tok_embed = embed_tokens_pub(weights, &[next_input_id]);
        let x_tok: Vec<f32> = tok_embed.as_slice().unwrap_or(&[]).to_vec();

        let mut step_error: Option<RemoteMoeError> = None;
        let mut moe_fn = |layer: usize, h_post_attn: &[f32]| -> Vec<f32> {
            if skip_moe { return vec![0.0f32; hidden]; }
            if step_error.is_some() { return vec![0.0f32; hidden]; }
            let router = match build_router(weights, arch, layer) {
                Some(r) => r,
                None => return vec![0.0f32; hidden],
            };
            let result = if streams.is_empty() {
                remote.forward_moe(layer, h_post_attn, &router, norm_offset, eps)
            } else {
                remote.forward_moe_stream(layer, h_post_attn, &router, &mut streams, norm_offset, eps)
            };
            match result {
                Ok(out) => out,
                Err(e) => { step_error = Some(e); vec![0.0f32; hidden] }
            }
        };

        let result = backend.decode_token_with_moe(
            &layers,
            &x_tok,
            hidden,
            intermediate,
            q_dim,
            kv_dim,
            weights.num_q_heads,
            weights.num_kv_heads,
            weights.head_dim,
            rope,
            &mut moe_fn,
        );

        if let Some(err) = step_error {
            return Err(err);
        }

        let h_vec = result.ok_or_else(|| {
            RemoteMoeError::BadResponse("decode_token_with_moe returned None".into())
        })?;

        last_hidden_vec = h_vec;

        let h_arr = ndarray::Array2::from_shape_vec((1, hidden), last_hidden_vec.clone())
            .map_err(|e| RemoteMoeError::BadResponse(e.to_string()))?;
        let h_normed = apply_norm(weights, &h_arr, arch.final_norm_key(), norm_offset);
        let last_hidden = h_normed.row(0).to_owned();
        let next_id = lm_topk(index, weights, &last_hidden, 1, backend)
            .into_iter()
            .next()
            .map(|(id, _)| id)
            .unwrap_or(0);

        decode_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        let tok_str = crate::tokenizer::decode_token(tokenizer, next_id)
            .unwrap_or_else(|| format!("<{next_id}>"));
        let is_eos = crate::vindex::is_end_of_turn(tok_str.trim());
        tokens.push(tok_str);
        current_ids.push(next_id);
        if is_eos {
            break;
        }
    }

    Ok(GridGenerateResult { tokens, decode_ms })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::test_utils::{make_test_tokenizer, make_test_vindex, make_test_weights};
    use crate::ffn::moe_remote::RemoteMoeBackend;
    use larql_compute::CpuBackend;

    // ── generate_with_remote_moe — error path ────────────────────────────────

    #[test]
    fn errors_when_vindex_has_no_q4k_mmap() {
        let weights = make_test_weights();
        let idx = make_test_vindex(&weights);
        let tokenizer = make_test_tokenizer(weights.vocab_size);

        // make_test_vindex has no interleaved Q4K or Q4 mmap.
        // The function should fail at the mmap guard, before any GPU or shard call.
        let remote = RemoteMoeBackend::new_disconnected();
        let result = generate_with_remote_moe(
            &weights,
            &tokenizer,
            vec![0u32],
            1,
            &idx,
            &remote,
            &CpuBackend,
        );
        match result {
            Err(RemoteMoeError::BadResponse(msg)) => {
                assert!(
                    msg.contains("no interleaved Q4 FFN mmap"),
                    "unexpected error message: {msg}"
                );
            }
            other => panic!("expected BadResponse, got: {other:?}"),
        }
    }
}
