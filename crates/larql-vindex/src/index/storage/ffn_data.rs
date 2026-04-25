//! `FfnStore` — owns FFN-side mmap handles, manifests, and the Q4_K
//! dequant cache.
//!
//! Carved out of the monolithic `VectorIndex` in the 2026-04-25
//! reorg. Field names mirror the legacy flat ones so call sites can
//! migrate mechanically; future PRs can drop redundant prefixes.
//!
//! The accessor / loader methods live next door in `ffn_store.rs`
//! (they need the full `VectorIndex` for `num_features(layer)`,
//! `hidden_size`, etc.). This file only carries the data shape +
//! `Clone` / `empty` constructors so `core.rs` can compose it.

use std::sync::{Arc, Mutex};

#[allow(clippy::type_complexity)]
pub struct FfnStore {
    /// Feature-major down projections (f32 mmap).
    pub down_features_mmap: Option<Arc<memmap2::Mmap>>,
    /// Feature-major up projections (f32 mmap).
    pub up_features_mmap: Option<Arc<memmap2::Mmap>>,
    /// Interleaved [gate|up|down] FFN data (f32, packed per layer).
    pub interleaved_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_0 quantized interleaved FFN.
    pub interleaved_q4_mmap: Option<Arc<memmap2::Mmap>>,
    /// Q4_K / Q6_K quantized interleaved FFN (Ollama-compatible).
    pub interleaved_q4k_mmap: Option<Arc<memmap2::Mmap>>,
    /// Per-matrix (offset, length, format) entries — 3 per layer in
    /// `[gate, up, down]` order.
    pub interleaved_q4k_manifest: Option<Vec<(usize, usize, String)>>,
    /// Per-layer lazy dequant cache for Q4_K/Q6_K FFN tensors.
    /// `q4k_ffn_cache[layer][c]` is the dequantised
    /// `[intermediate × hidden]` matrix for component `c`
    /// (0=gate, 1=up, 2=down). LRU-bounded by
    /// `q4k_ffn_cache_max_layers`.
    pub q4k_ffn_cache: Mutex<Vec<[Option<Arc<Vec<f32>>>; 3]>>,
    /// LRU of layers held in `q4k_ffn_cache`. Front = newest.
    pub q4k_ffn_cache_lru: Mutex<std::collections::VecDeque<usize>>,
    /// Cap on `q4k_ffn_cache`. 0 = unlimited (default).
    pub q4k_ffn_cache_max_layers: std::sync::atomic::AtomicUsize,
    /// FP4 / FP8 FFN storage (exp 26).
    pub fp4_storage: Option<Arc<crate::index::fp4_storage::Fp4Storage>>,
}

impl FfnStore {
    pub fn empty(num_layers: usize) -> Self {
        Self {
            down_features_mmap: None,
            up_features_mmap: None,
            interleaved_mmap: None,
            interleaved_q4_mmap: None,
            interleaved_q4k_mmap: None,
            interleaved_q4k_manifest: None,
            q4k_ffn_cache: Mutex::new(
                (0..num_layers).map(|_| [None, None, None]).collect(),
            ),
            q4k_ffn_cache_lru: Mutex::new(std::collections::VecDeque::new()),
            q4k_ffn_cache_max_layers: std::sync::atomic::AtomicUsize::new(0),
            fp4_storage: None,
        }
    }
}

impl Clone for FfnStore {
    fn clone(&self) -> Self {
        use std::sync::atomic::Ordering;
        let nl = self
            .q4k_ffn_cache
            .lock()
            .map(|c| c.len())
            .unwrap_or(0);
        Self {
            down_features_mmap: self.down_features_mmap.clone(),
            up_features_mmap: self.up_features_mmap.clone(),
            interleaved_mmap: self.interleaved_mmap.clone(),
            interleaved_q4_mmap: self.interleaved_q4_mmap.clone(),
            interleaved_q4k_mmap: self.interleaved_q4k_mmap.clone(),
            interleaved_q4k_manifest: self.interleaved_q4k_manifest.clone(),
            q4k_ffn_cache: Mutex::new(
                (0..nl).map(|_| [None, None, None]).collect(),
            ),
            q4k_ffn_cache_lru: Mutex::new(std::collections::VecDeque::new()),
            q4k_ffn_cache_max_layers: std::sync::atomic::AtomicUsize::new(
                self.q4k_ffn_cache_max_layers.load(Ordering::Relaxed),
            ),
            fp4_storage: self.fp4_storage.clone(),
        }
    }
}
