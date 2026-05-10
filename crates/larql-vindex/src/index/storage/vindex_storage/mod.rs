//! `VindexStorage` — sealed mmap-agnostic byte-handle trait.
//!
//! Goal: every backend (today's mmap; eventual Redis-cached, S3-buffered,
//! GPU-resident) satisfies the same surface. Walk kernels, Metal
//! dispatch, and KNN consume this trait — they don't reach into
//! `Arc<Mmap>` fields directly.
//!
//! ## Status
//!
//! Step 1 of the migration plan in `ROADMAP.md` (P0 active, promoted
//! from P2 on 2026-05-10): trait skeleton only, no impls, no callsite
//! changes. The `MmapStorage` parity wrapper lands in step 2; the
//! Criterion bench gate lands in step 3 before any substore migration.
//!
//! ## Returns: `bytes::Bytes`
//!
//! All byte-yielding methods return `bytes::Bytes` (or array/tuple
//! shapes built on top of it). Zero-copy from `Arc<Mmap>` via
//! `Bytes::from_owner`, slicing is O(1) refcounted, and Redis-style
//! backends can return owned bytes without lifetime contortions. The
//! cost is one atomic increment per *layer* fetch, which amortises
//! across rows in the inner decode loop.
//!
//! `&[u8]` was rejected because it locks remote backends out (no
//! anchor on the substore). `Cow<'_, [u8]>` was rejected because every
//! signature would carry a lifetime, rippling through array-of-tuple
//! returns. `Arc<[u8]>` was rejected because slicing isn't cheap.
//!
//! ## Sealing
//!
//! Sealed via the standard private supertrait pattern: out-of-crate
//! types cannot implement `VindexStorage`. We need this so we can add
//! methods (with defaults) without a major bump as Redis / S3 backends
//! arrive. The cost is that downstream tests can't write a stub impl —
//! acceptable here because the capability traits
//! (`QuantizedFfnAccess`, `NativeFfnAccess`, `Fp4FfnAccess`) already
//! cover the dispatch logic above this layer with their own stubs.
//!
//! ## Method shape
//!
//! Every byte-yielding method returns `Option<...>`. `None` means
//! "this backend doesn't carry that file kind" (e.g., FP4 vindexes
//! don't have Q4_K FFN data; client-only slices have no gate
//! vectors). Callers must treat absence as a fall-through, not an
//! error.
//!
//! ## Out of scope (for now)
//!
//! `Fp4Storage` and `DownMetaMmap` are deliberately not behind this
//! trait. Both carry richer per-feature decoders (FP4/FP8 dequant
//! tables, per-layer offsets + tokenizer for down_meta) that are not
//! a clean fit for the "give me bytes" surface. They keep their own
//! mmap fields on substores and stay reachable as
//! `Arc<Fp4Storage>` / `Arc<DownMetaMmap>` directly. If a Redis-backed
//! FP4 vindex ever lands, the path is to either provide a parallel
//! `Fp4Storage` impl or have `Fp4Storage` consume `VindexStorage`
//! internally — but that's a separate decision from this trait.

use bytes::Bytes;

use crate::config::dtype::StorageDtype;
use crate::index::storage::attn::ATTN_TENSORS_PER_LAYER;
use crate::index::storage::ffn_store::FFN_COMPONENTS_PER_LAYER;
use crate::index::types::GateLayerSlice;

mod mmap_storage;
pub use mmap_storage::MmapStorage;

/// Bundled view of one layer's gate vectors. Replaces three
/// independent substore reaches (`gate.gate_mmap_bytes` +
/// `gate_mmap_slices[layer]` + `gate_mmap_dtype`) that always travel
/// together.
#[derive(Clone)]
pub struct GateLayerView {
    /// Whole gate-vectors mmap as a refcounted handle. Zero-copy from
    /// `Arc<Mmap>` via `Bytes::from_owner`. Callers slice into this
    /// using `slice.float_offset` × dtype byte width.
    pub bytes: Bytes,
    /// Storage dtype of the gate matrix (`F16` or `F32` in production
    /// vindexes; legacy paths may report `F32` for synthesized gates).
    pub dtype: StorageDtype,
    /// Per-layer offset + feature count inside the gate buffer.
    pub slice: GateLayerSlice,
}

/// Sealed mmap-agnostic byte-handle for vindex storage backends.
///
/// See module docs for the design rationale and the migration plan
/// this fits into.
pub trait VindexStorage: sealed::Sealed + Send + Sync {
    // ── FFN ─────────────────────────────────────────────────────────────

    /// Q4_K / Q6_K interleaved FFN slices for one layer:
    /// `[(gate_bytes, gate_fmt), (up_bytes, up_fmt), (down_bytes, down_fmt)]`.
    ///
    /// `None` when this backend has no Q4_K interleaved FFN file or the
    /// layer is out of range. `fmt` is the quant tag (`"Q4_K"` /
    /// `"Q6_K"`) routed through `quant::registry`.
    fn interleaved_q4k_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(Bytes, &str); FFN_COMPONENTS_PER_LAYER]>;

    /// Whole-file Q4_K interleaved FFN buffer. Used by Metal
    /// `q4k_matmul_transb` for full-K decode without per-layer
    /// gathering.
    fn interleaved_q4k_whole_buffer(&self) -> Option<Bytes>;

    /// Whole-file Q4_0 interleaved FFN buffer. The Q4_0 path doesn't
    /// have a per-layer manifest; consumers compute layer offsets
    /// from `num_features`.
    fn interleaved_q4_whole_buffer(&self) -> Option<Bytes>;

    /// W2 feature-major Q4_K down for one layer:
    /// `(bytes, fmt, padded_width)`. `padded_width` is the row stride
    /// after `pad_rows_to_block` — usually equal to `hidden_size`.
    fn down_features_q4k_layer_data(&self, layer: usize) -> Option<(Bytes, &str, usize)>;

    /// Q4_0 gate vectors for one layer (KNN side-channel — feature
    /// retrieval without dequantising the full layer).
    fn gate_q4_layer_data(&self, layer: usize) -> Option<Bytes>;

    // ── Attention ───────────────────────────────────────────────────────

    /// Q4_K / Q6_K attention projections for one layer:
    /// `[(Q, fmt), (K, fmt), (V, fmt), (O, fmt)]`. `None` when no Q4_K
    /// attention manifest is loaded or the layer is out of range.
    fn attn_q4k_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(Bytes, &str); ATTN_TENSORS_PER_LAYER]>;

    /// Whole-file Q4_0 attention buffer.
    fn attn_q4_whole_buffer(&self) -> Option<Bytes>;

    /// Q4_0 attention projections for one layer: `[Q, K, V, O]` byte
    /// slices.
    fn attn_q4_layer_slices(&self, layer: usize) -> Option<[Bytes; ATTN_TENSORS_PER_LAYER]>;

    /// Q8 attention projections for one layer:
    /// `[(vals, scales), (vals, scales), (vals, scales), (vals, scales)]`
    /// for Q, K, V, O. Scales are returned as `Bytes` — the caller
    /// reinterprets as `&[f32]` (today's accessor does the same via
    /// `slice::from_raw_parts`).
    fn attn_q8_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(Bytes, Bytes); ATTN_TENSORS_PER_LAYER]>;

    // ── lm_head ─────────────────────────────────────────────────────────

    /// Q4_0 lm_head buffer (`lm_head_q4.bin`).
    fn lm_head_q4_bytes(&self) -> Option<Bytes>;
    /// f16 lm_head buffer (`lm_head.bin` when the source dtype is f16).
    fn lm_head_f16_bytes(&self) -> Option<Bytes>;
    /// f32 lm_head buffer (`lm_head.bin` when the source dtype is f32).
    fn lm_head_f32_bytes(&self) -> Option<Bytes>;

    // ── Gate vectors (KNN) ──────────────────────────────────────────────

    /// Bundled view of one layer's gate vectors: bytes + dtype + slice.
    /// Replaces the three-field reach
    /// (`gate_mmap_bytes` + `gate_mmap_slices[layer]` + `gate_mmap_dtype`)
    /// that always travels together.
    fn gate_layer_view(&self, layer: usize) -> Option<GateLayerView>;
}

mod sealed {
    /// Crate-private supertrait that prevents out-of-crate impls of
    /// `VindexStorage`. Every type that implements `VindexStorage`
    /// inside this crate must also implement `Sealed`. Lives in a
    /// private module so it can't be named by downstream code.
    pub trait Sealed {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `VindexStorage` must be object-safe — the migration plan holds
    /// it as `Arc<dyn VindexStorage>` on `VectorIndex`.
    #[test]
    fn trait_is_object_safe() {
        // Compile-time check: if the trait gains a non-object-safe
        // method (generics, `Self` by value, associated consts), this
        // line fails to compile.
        fn _assert_object_safe(_: &dyn VindexStorage) {}
    }

    /// `Arc<dyn VindexStorage>` must be `Send + Sync` so it can be
    /// shared across the rayon worker pool the way today's
    /// `Arc<VectorIndex>` is.
    #[test]
    fn trait_object_is_send_sync() {
        fn _assert_send_sync<T: Send + Sync>() {}
        _assert_send_sync::<std::sync::Arc<dyn VindexStorage>>();
    }

    /// `GateLayerView` clones cheaply — `Bytes` is refcounted, the
    /// other two fields are `Copy`-equivalents. Callers will hold this
    /// across kernel dispatches; an expensive clone would be a
    /// surprise.
    #[test]
    fn gate_layer_view_clones_via_refcount() {
        let bytes = Bytes::from_static(b"abc");
        let view = GateLayerView {
            bytes: bytes.clone(),
            dtype: StorageDtype::F16,
            slice: GateLayerSlice {
                float_offset: 0,
                num_features: 1,
            },
        };
        let cloned = view.clone();
        // Same underlying buffer: ptr equality on the slice the Bytes
        // is pointing at.
        assert_eq!(view.bytes.as_ptr(), cloned.bytes.as_ptr());
    }
}
