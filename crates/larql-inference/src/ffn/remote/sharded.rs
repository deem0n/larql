//! Layer-sharded FFN backend.
//!
//! Routes each layer's FFN call to whichever shard owns that layer range.
//! A single-URL `--ffn URL` is the degenerate case (one shard, all layers).
//! A multi-shard `--ffn "0-14=URL1,15-29=URL2"` fans out by layer.
//!
//! Each shard may itself have `--moe-shards` configured server-side, making
//! expert dispatch transparent to the client.

use std::time::Duration;

use ndarray::Array2;

use super::http::{RemoteFfnConfig, RemoteFfnError, RemoteWalkBackend};
use crate::ffn::FfnBackend;

struct LayerShard {
    start: usize,
    end: usize, // inclusive
    backend: RemoteWalkBackend,
}

/// FFN backend that routes each layer to the owning shard.
///
/// Build with [`LayerShardedBackend::connect`]. Parses either:
/// - A bare URL `"http://host:8080"` → single shard, all layers.
/// - A shard map `"0-14=http://a:8091,15-29=http://b:8092"` → routed by layer.
pub struct LayerShardedBackend {
    shards: Vec<LayerShard>,
}

impl LayerShardedBackend {
    /// Build from a spec string and connect (health-check) each shard.
    pub fn connect(spec: &str, timeout: Duration) -> Result<Self, RemoteFfnError> {
        let shards = if spec.contains('=') {
            parse_shard_map(spec, timeout)?
        } else {
            let config = RemoteFfnConfig::new(spec).with_timeout(timeout);
            let backend = RemoteWalkBackend::connect(config)?;
            vec![LayerShard { start: 0, end: usize::MAX, backend }]
        };
        Ok(Self { shards })
    }

    pub fn hidden_size(&self) -> usize {
        self.shards.first().map(|s| s.backend.hidden_size()).unwrap_or(0)
    }

    /// URL of the first shard (for logging/display).
    pub fn primary_url(&self) -> &str {
        self.shards.first().map(|s| s.backend.base_url()).unwrap_or("")
    }

    fn shard_for(&self, layer: usize) -> Option<&RemoteWalkBackend> {
        self.shards
            .iter()
            .find(|s| layer >= s.start && layer <= s.end)
            .map(|s| &s.backend)
    }
}

impl FfnBackend for LayerShardedBackend {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        match self.shard_for(layer) {
            Some(shard) => shard.forward(layer, x),
            None => Array2::zeros(x.raw_dim()),
        }
    }

    fn forward_with_activation(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        match self.shard_for(layer) {
            Some(shard) => shard.forward_with_activation(layer, x),
            None => {
                let z = Array2::zeros(x.raw_dim());
                (z.clone(), z)
            }
        }
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Option<Array2<f32>> {
        self.shard_for(layer)?.forward_moe_full_layer(layer, h_post_attn)
    }

    fn name(&self) -> &str {
        "layer-sharded-remote"
    }
}

// ── Parse "START-END=URL,..." ─────────────────────────────────────────────────

fn parse_shard_map(spec: &str, timeout: Duration) -> Result<Vec<LayerShard>, RemoteFfnError> {
    let mut shards = Vec::new();
    for segment in spec.split(',') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        let mut parts = segment.splitn(2, '=');
        let range_str = parts.next().ok_or_else(|| {
            RemoteFfnError::Client(format!("malformed --ffn segment: {segment:?}"))
        })?;
        let url = parts.next().ok_or_else(|| {
            RemoteFfnError::Client(format!("missing URL in --ffn segment: {segment:?}"))
        })?;
        let (start, end) = parse_layer_range(range_str).ok_or_else(|| {
            RemoteFfnError::Client(format!("bad layer range {range_str:?} in --ffn"))
        })?;
        let config = RemoteFfnConfig::new(url).with_timeout(timeout);
        let backend = RemoteWalkBackend::connect(config)?;
        shards.push(LayerShard { start, end, backend });
    }
    if shards.is_empty() {
        return Err(RemoteFfnError::Client("--ffn: no valid shard segments".into()));
    }
    Ok(shards)
}

fn parse_layer_range(s: &str) -> Option<(usize, usize)> {
    let mut parts = s.splitn(2, '-');
    let start: usize = parts.next()?.trim().parse().ok()?;
    let end: usize = parts.next()?.trim().parse().ok()?;
    if start <= end { Some((start, end)) } else { None }
}
