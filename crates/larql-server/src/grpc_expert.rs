//! gRPC `ExpertService` implementation.
//!
//! Exposes two RPCs:
//!
//! `ExpertBatch` — unary, processes a flat list of (layer, expert_id, residual) items.
//! Good for correctness testing and small batches.
//!
//! `ExpertStream` — bidirectional streaming, one frame per MoE layer per decode step.
//! Client sends `ExpertLayerInput` for each layer as it becomes available; server
//! streams back `ExpertLayerOutput` after computing the weighted expert sum.
//! ONE stream per shard per token eliminates the per-call connection overhead of
//! 30 unary calls — measured improvement: ~360ms overhead → ~18ms.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use futures::Stream;
use tonic::{Request, Response, Status, Streaming};

use larql_router_protocol::{
    ExpertBatchRequest, ExpertBatchResponse, ExpertBatchResult, ExpertLayerInput,
    ExpertLayerOutput, ExpertService,
};

use crate::state::AppState;

pub struct ExpertGrpcService {
    pub state: Arc<AppState>,
}

#[tonic::async_trait]
impl ExpertService for ExpertGrpcService {
    // ── Unary batch ──────────────────────────────────────────────────────────

    async fn expert_batch(
        &self,
        request: Request<ExpertBatchRequest>,
    ) -> Result<Response<ExpertBatchResponse>, Status> {
        self.state.bump_requests();
        let start = Instant::now();
        let req = request.into_inner();
        let state = Arc::clone(&self.state);

        let results = tokio::task::spawn_blocking(move || {
            req.items
                .iter()
                .map(|item| {
                    let layer = item.layer as usize;
                    let expert_id = item.expert_id as usize;

                    if item.residual.len() % 4 != 0 {
                        return Err(Status::invalid_argument("residual not 4-byte aligned"));
                    }
                    let residual: Vec<f32> = item
                        .residual
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();

                    let output =
                        crate::routes::expert::run_expert(&state, layer, expert_id, &residual)
                            .map_err(|e| Status::internal(e.to_string()))?;

                    let output_bytes: Vec<u8> =
                        output.iter().flat_map(|v| v.to_le_bytes()).collect();

                    Ok(ExpertBatchResult {
                        layer: item.layer,
                        expert_id: item.expert_id,
                        output: output_bytes,
                    })
                })
                .collect::<Result<Vec<_>, Status>>()
        })
        .await
        .map_err(|e| Status::internal(e.to_string()))??;

        let latency_ms = start.elapsed().as_secs_f32() * 1000.0;
        Ok(Response::new(ExpertBatchResponse {
            results,
            latency_ms,
        }))
    }

    // ── Bidirectional streaming ──────────────────────────────────────────────
    //
    // Each incoming ExpertLayerInput carries:
    //   layer, expert_ids[], expert_weights[], residual (h_post_attn), post_experts_norm
    //
    // For each message, the server:
    //   1. Runs each selected expert: run_single_expert_with_norm(residual, ...)
    //   2. Weighted sum: h2 = sum(w_k * expert_k_output)
    //   3. Post-experts norm: h2 = rms_norm(h2, post_experts_norm)
    //   4. Streams back ExpertLayerOutput { layer, h2 }

    type ExpertStreamStream =
        Pin<Box<dyn Stream<Item = Result<ExpertLayerOutput, Status>> + Send + 'static>>;

    async fn expert_stream(
        &self,
        request: Request<Streaming<ExpertLayerInput>>,
    ) -> Result<Response<Self::ExpertStreamStream>, Status> {
        self.state.bump_requests();
        let state = Arc::clone(&self.state);
        let mut in_stream = request.into_inner();

        let out_stream = async_stream::try_stream! {
            while let Some(msg) = {
                use futures::StreamExt;
                in_stream.next().await
            } {
                let input = msg?;
                let layer = input.layer as usize;

                // Decode bytes on the async thread, then do blocking expert compute.
                if input.residual.len() % 4 != 0 {
                    Err(Status::invalid_argument("residual not 4-byte aligned"))?;
                }
                let residual: Vec<f32> = input
                    .residual
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect();

                let post_norm: Vec<f32> = if input.post_experts_norm.is_empty() {
                    vec![]
                } else {
                    input.post_experts_norm
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect()
                };
                let norm_offset = input.norm_offset;
                let eps = input.eps;

                let expert_ids: Vec<usize> =
                    input.expert_ids.iter().map(|&e| e as usize).collect();
                let expert_weights: Vec<f32> = input.expert_weights.clone();

                let state2 = Arc::clone(&state);

                // Run on the blocking pool — expert matmuls are CPU-bound.
                let h2 = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, Status> {
                    let hidden = residual.len();
                    let mut out = vec![0.0f32; hidden];

                    for (&expert_id, &weight) in
                        expert_ids.iter().zip(expert_weights.iter())
                    {
                        if weight == 0.0 {
                            continue;
                        }
                        let expert_out =
                            crate::routes::expert::run_expert(&state2, layer, expert_id, &residual)
                                .map_err(|e| Status::internal(e.to_string()))?;
                        for (acc, &v) in out.iter_mut().zip(expert_out.iter()) {
                            *acc += weight * v;
                        }
                    }

                    // Post-experts norm is applied by the CLIENT after combining
                    // all shards' partial sums.  Applying it here (on a partial
                    // sum) then summing would be wrong:
                    //   norm(A) + norm(B) ≠ norm(A + B)
                    // The server returns the raw partial weighted sum; the client
                    // does the final post_experts_norm over the combined result.
                    Ok(out)
                })
                .await
                .map_err(|e| Status::internal(e.to_string()))??;

                let h2_bytes: Vec<u8> = h2.iter().flat_map(|v| v.to_le_bytes()).collect();
                yield ExpertLayerOutput {
                    layer: input.layer,
                    h2: h2_bytes,
                };
            }
        };

        Ok(Response::new(Box::pin(out_stream)))
    }
}
