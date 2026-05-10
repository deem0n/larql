//! `MmapStorage` — production `VindexStorage` impl backed by the
//! existing `Arc<Mmap>` substore fields.
//!
//! Step 2 of the migration plan: a parity wrapper that returns the
//! same byte ranges the substore accessors do today, in `Bytes`
//! shape. No behavior change. Substores still own their
//! `Arc<Mmap>` fields; `MmapStorage` holds clones (cheap — one Arc
//! refcount bump per file at construction) and a reused `Bytes`
//! handle per whole-file mmap so per-layer slices are O(1)
//! refcounted.
//!
//! In step 4 the substore byte-yielding accessors get rewritten to
//! forward through `MmapStorage`. In step 5 the substore mmap fields
//! drop entirely. Until then this is purely additive.

use std::sync::Arc;

use bytes::Bytes;

use crate::config::dtype::StorageDtype;
use crate::index::storage::attn::ATTN_TENSORS_PER_LAYER;
use crate::index::storage::ffn_store::{DownFeaturesQ4kEntry, FfnStore, FFN_COMPONENTS_PER_LAYER};
use crate::index::storage::gate_store::GateStore;
use crate::index::storage::projection_store::ProjectionStore;
use crate::index::types::{GateLayerSlice, GateQ4Slice};

use super::sealed::Sealed;
use super::{BytesView, GateLayerView, VindexStorage};

/// Parity wrapper over today's substore mmaps. Implements
/// `VindexStorage` by cloning each substore's `Arc<Mmap>` (or
/// `Arc<Vec<u8>>` for the synth lm_head) and converting once into a
/// reusable `Bytes` whole-file handle. Per-layer accessors slice the
/// whole-file `Bytes` via `Bytes::slice` — O(1) refcounted, no copy.
#[derive(Clone)]
pub struct MmapStorage {
    // Fields are `pub(crate)` so tests inside the crate can construct
    // corrupt-manifest fixtures directly. External callers must go
    // through the trait or the inherent setters/views.

    // ── FFN ──────────────────────────────────────────────────────────
    pub(crate) interleaved_q4k: Option<Bytes>,
    pub(crate) interleaved_q4k_manifest: Option<Vec<(usize, usize, String)>>,
    pub(crate) interleaved_q4: Option<Bytes>,
    pub(crate) down_features_q4k: Option<Bytes>,
    pub(crate) down_features_q4k_manifest: Option<Vec<DownFeaturesQ4kEntry>>,

    // ── Attention ────────────────────────────────────────────────────
    pub(crate) attn_q4k: Option<Bytes>,
    pub(crate) attn_q4k_manifest: Option<Vec<(usize, usize, String)>>,
    pub(crate) attn_q4: Option<Bytes>,
    pub(crate) attn_q4_manifest: Option<Vec<(usize, usize)>>,
    pub(crate) attn_q8: Option<Bytes>,
    pub(crate) attn_q8_manifest: Option<Vec<(usize, usize, usize)>>,

    // ── lm_head ──────────────────────────────────────────────────────
    pub(crate) lm_head_f32: Option<Bytes>,
    pub(crate) lm_head_f16: Option<Bytes>,
    pub(crate) lm_head_q4: Option<Bytes>,

    // ── Gate ─────────────────────────────────────────────────────────
    pub(crate) gate_bytes: Option<Bytes>,
    pub(crate) gate_dtype: StorageDtype,
    pub(crate) gate_slices: Vec<GateLayerSlice>,
    pub(crate) gate_q4_bytes: Option<Bytes>,
    pub(crate) gate_q4_slices: Vec<GateQ4Slice>,
    /// Hidden dim — gate `GateLayerView::slice.float_offset` is in
    /// floats, but a Redis-backed impl will need `hidden_size` to
    /// slice from a flat buffer. Carry it here so the trait surface
    /// doesn't have to.
    #[allow(dead_code)]
    hidden_size: usize,
}

impl Sealed for MmapStorage {}

impl MmapStorage {
    /// Build a parity wrapper from today's substores. Cheap — every
    /// `Arc<Mmap>` clone is a refcount bump; the only allocation is
    /// `Bytes::from_owner` once per whole-file mmap.
    ///
    /// `lm_head_q4_synth` (in-RAM Q4 synthesised from f16 embeddings)
    /// is folded into `lm_head_q4`: callers don't see the difference.
    pub fn from_substores(
        ffn: &FfnStore,
        gate: &GateStore,
        projections: &ProjectionStore,
        hidden_size: usize,
    ) -> Self {
        // lm_head_q4 unifies the mmap and the in-RAM synth fallback.
        // The synth path is `Arc<Vec<u8>>`; the mmap path is
        // `Arc<Mmap>`. Both convert to `Bytes::from_owner` cleanly.
        let lm_head_q4 = projections
            .lm_head_q4_mmap
            .as_ref()
            .map(arc_mmap_to_bytes)
            .or_else(|| {
                projections
                    .lm_head_q4_synth
                    .as_ref()
                    .map(|v| Bytes::from_owner(ArcAsBytes(v.clone())))
            });

        Self {
            // FFN
            interleaved_q4k: ffn.interleaved_q4k_mmap.as_ref().map(arc_mmap_to_bytes),
            interleaved_q4k_manifest: ffn.interleaved_q4k_manifest.clone(),
            interleaved_q4: ffn.interleaved_q4_mmap.as_ref().map(arc_mmap_to_bytes),
            down_features_q4k: ffn.down_features_q4k_mmap.as_ref().map(arc_mmap_to_bytes),
            down_features_q4k_manifest: ffn.down_features_q4k_manifest.clone(),

            // Attention
            attn_q4k: projections.attn_q4k_mmap.as_ref().map(arc_mmap_to_bytes),
            attn_q4k_manifest: projections.attn_q4k_manifest.clone(),
            attn_q4: projections.attn_q4_mmap.as_ref().map(arc_mmap_to_bytes),
            attn_q4_manifest: projections.attn_q4_manifest.clone(),
            attn_q8: projections.attn_q8_mmap.as_ref().map(arc_mmap_to_bytes),
            attn_q8_manifest: projections.attn_q8_manifest.clone(),

            // lm_head
            lm_head_f32: projections.lm_head_mmap.as_ref().map(arc_mmap_to_bytes),
            lm_head_f16: projections.lm_head_f16_mmap.as_ref().map(arc_mmap_to_bytes),
            lm_head_q4,

            // Gate
            gate_bytes: gate.gate_mmap_bytes.as_ref().map(arc_mmap_to_bytes),
            gate_dtype: gate.gate_mmap_dtype,
            gate_slices: gate.gate_mmap_slices.clone(),
            gate_q4_bytes: gate.gate_q4_mmap.as_ref().map(arc_mmap_to_bytes),
            gate_q4_slices: gate.gate_q4_slices.clone(),
            hidden_size,
        }
    }

    // ── Per-field setters (loader-side mutation) ───────────────────────
    //
    // Loaders that used to write `self.<substore>.<field> = Some(...)`
    // now go through these setters. `VectorIndex::storage` is
    // `Arc<MmapStorage>`; `Arc::make_mut(&mut self.storage)` clones if
    // the Arc is shared, then returns `&mut MmapStorage` for the
    // setter to mutate. That preserves the Arc-clone semantics on
    // `VectorIndex::clone` (cheap refcount bump until a loader
    // mutates a clone, at which point the clone gets its own copy).

    /// Set the FFN interleaved Q4_K mmap + manifest. Each
    /// (offset, length, format_tag) triple in the manifest is one
    /// component (gate / up / down).
    pub fn set_interleaved_q4k(
        &mut self,
        mmap: Arc<memmap2::Mmap>,
        manifest: Option<Vec<(usize, usize, String)>>,
    ) {
        self.interleaved_q4k = Some(arc_mmap_to_bytes(&mmap));
        self.interleaved_q4k_manifest = manifest;
    }

    /// Set the FFN interleaved Q4_0 mmap (no manifest — uniform stride).
    pub fn set_interleaved_q4(&mut self, mmap: Arc<memmap2::Mmap>) {
        self.interleaved_q4 = Some(arc_mmap_to_bytes(&mmap));
    }

    /// Set the W2 feature-major Q4_K down mmap + manifest.
    pub fn set_down_features_q4k(
        &mut self,
        mmap: Arc<memmap2::Mmap>,
        manifest: Vec<DownFeaturesQ4kEntry>,
    ) {
        self.down_features_q4k = Some(arc_mmap_to_bytes(&mmap));
        self.down_features_q4k_manifest = Some(manifest);
    }

    /// Set the attention Q4_K mmap + manifest.
    pub fn set_attn_q4k(
        &mut self,
        mmap: Arc<memmap2::Mmap>,
        manifest: Option<Vec<(usize, usize, String)>>,
    ) {
        self.attn_q4k = Some(arc_mmap_to_bytes(&mmap));
        self.attn_q4k_manifest = manifest;
    }

    /// Set the attention Q4_0 mmap + manifest.
    pub fn set_attn_q4(&mut self, mmap: Arc<memmap2::Mmap>, manifest: Option<Vec<(usize, usize)>>) {
        self.attn_q4 = Some(arc_mmap_to_bytes(&mmap));
        self.attn_q4_manifest = manifest;
    }

    /// Set the attention Q8 mmap + manifest.
    pub fn set_attn_q8(
        &mut self,
        mmap: Arc<memmap2::Mmap>,
        manifest: Option<Vec<(usize, usize, usize)>>,
    ) {
        self.attn_q8 = Some(arc_mmap_to_bytes(&mmap));
        self.attn_q8_manifest = manifest;
    }

    /// Set the lm_head f32 mmap.
    pub fn set_lm_head_f32(&mut self, mmap: Arc<memmap2::Mmap>) {
        self.lm_head_f32 = Some(arc_mmap_to_bytes(&mmap));
    }

    /// Set the lm_head f16 mmap (tied-embedding case).
    pub fn set_lm_head_f16(&mut self, mmap: Arc<memmap2::Mmap>) {
        self.lm_head_f16 = Some(arc_mmap_to_bytes(&mmap));
    }

    /// Set the lm_head Q4 from an mmap'd file.
    pub fn set_lm_head_q4_mmap(&mut self, mmap: Arc<memmap2::Mmap>) {
        self.lm_head_q4 = Some(arc_mmap_to_bytes(&mmap));
    }

    /// Set the lm_head Q4 from in-RAM synthesised bytes (the f16
    /// embeddings → Q4 fallback path).
    pub fn set_lm_head_q4_synth(&mut self, bytes: Arc<Vec<u8>>) {
        self.lm_head_q4 = Some(Bytes::from_owner(ArcAsBytes(bytes)));
    }

    /// Set the gate vectors mmap + dtype + per-layer slices.
    pub fn set_gate_vectors(
        &mut self,
        mmap: Arc<memmap2::Mmap>,
        dtype: StorageDtype,
        slices: Vec<GateLayerSlice>,
    ) {
        self.gate_bytes = Some(arc_mmap_to_bytes(&mmap));
        self.gate_dtype = dtype;
        self.gate_slices = slices;
    }

    /// Set the Q4_0 gate vectors mmap + per-layer Q4 slices.
    pub fn set_gate_q4(&mut self, mmap: Arc<memmap2::Mmap>, slices: Vec<GateQ4Slice>) {
        self.gate_q4_bytes = Some(arc_mmap_to_bytes(&mmap));
        self.gate_q4_slices = slices;
    }

    // ── Boolean capability checks (consumed by `is_some()` migrations) ─

    pub fn has_interleaved_q4k(&self) -> bool {
        self.interleaved_q4k.is_some()
    }
    pub fn has_interleaved_q4(&self) -> bool {
        self.interleaved_q4.is_some()
    }
    pub fn has_down_features_q4k(&self) -> bool {
        self.down_features_q4k.is_some()
    }
    pub fn has_attn_q4k(&self) -> bool {
        self.attn_q4k.is_some()
    }
    pub fn has_attn_q4(&self) -> bool {
        self.attn_q4.is_some()
    }
    pub fn has_attn_q8(&self) -> bool {
        self.attn_q8.is_some()
    }
    pub fn has_lm_head_q4(&self) -> bool {
        self.lm_head_q4.is_some()
    }
    pub fn has_lm_head_f16(&self) -> bool {
        self.lm_head_f16.is_some()
    }
    pub fn has_lm_head_f32(&self) -> bool {
        self.lm_head_f32.is_some()
    }
    pub fn has_gate_vectors(&self) -> bool {
        self.gate_bytes.is_some()
    }
    pub fn has_gate_q4(&self) -> bool {
        self.gate_q4_bytes.is_some()
    }

    // ── Whole-buffer view accessors (zero-atomic borrows) ──────────────

    /// Borrow the FFN Q4_K interleaved buffer without paying a
    /// refcount bump. Use when the consumer needs the bytes for the
    /// duration of `&self` only (e.g., madvise, kernel dispatch).
    pub fn interleaved_q4k_whole_buffer_view(&self) -> Option<&Bytes> {
        self.interleaved_q4k.as_ref()
    }
    pub fn interleaved_q4_whole_buffer_view(&self) -> Option<&Bytes> {
        self.interleaved_q4.as_ref()
    }
    pub fn attn_q4_whole_buffer_view(&self) -> Option<&Bytes> {
        self.attn_q4.as_ref()
    }
    pub fn lm_head_f32_view(&self) -> Option<&Bytes> {
        self.lm_head_f32.as_ref()
    }
    pub fn lm_head_f16_view(&self) -> Option<&Bytes> {
        self.lm_head_f16.as_ref()
    }
    pub fn lm_head_q4_view(&self) -> Option<&Bytes> {
        self.lm_head_q4.as_ref()
    }

    /// Inert empty wrapper — every `Option` is `None`. Used by
    /// `VectorIndex::empty()` and tests. Constructed without any of
    /// the substore types so callers don't have to fabricate empty
    /// substores just to get a storage handle.
    pub fn empty(hidden_size: usize) -> Self {
        Self {
            interleaved_q4k: None,
            interleaved_q4k_manifest: None,
            interleaved_q4: None,
            down_features_q4k: None,
            down_features_q4k_manifest: None,
            attn_q4k: None,
            attn_q4k_manifest: None,
            attn_q4: None,
            attn_q4_manifest: None,
            attn_q8: None,
            attn_q8_manifest: None,
            lm_head_f32: None,
            lm_head_f16: None,
            lm_head_q4: None,
            gate_bytes: None,
            gate_dtype: StorageDtype::F32,
            gate_slices: Vec::new(),
            gate_q4_bytes: None,
            gate_q4_slices: Vec::new(),
            hidden_size,
        }
    }
}

/// `Arc<Mmap>` → `Bytes` via `Bytes::from_owner`. Zero-copy: the
/// `Bytes` keeps the `Arc<Mmap>` alive for the lifetime of any
/// outstanding slices.
fn arc_mmap_to_bytes(arc: &Arc<memmap2::Mmap>) -> Bytes {
    Bytes::from_owner(ArcAsBytes(arc.clone()))
}

/// Owner wrapper so `bytes::Bytes::from_owner` (which requires
/// `AsRef<[u8]>` on the owner) accepts an `Arc<T>` where `T` already
/// implements `AsRef<[u8]>`. Both `memmap2::Mmap` and `Vec<u8>` (the
/// in-RAM synth lm_head) qualify; without this wrapper Rust looks for
/// `AsRef<[u8]> for Arc<T>` directly and only finds `AsRef<T>`.
struct ArcAsBytes<T: AsRef<[u8]> + Send + Sync + 'static>(Arc<T>);

impl<T: AsRef<[u8]> + Send + Sync + 'static> AsRef<[u8]> for ArcAsBytes<T> {
    fn as_ref(&self) -> &[u8] {
        (*self.0).as_ref()
    }
}

/// Bounds-check (`offset + length <= bytes.len()`) and build a
/// borrowed `BytesView`. Matches the defensive behavior of every
/// substore accessor that consults a stale-or-corrupt manifest.
fn checked_view<'a>(bytes: &'a Bytes, offset: usize, length: usize) -> Option<BytesView<'a>> {
    let end = offset.checked_add(length)?;
    if end > bytes.len() {
        return None;
    }
    Some(BytesView::new(bytes, offset, length))
}

impl VindexStorage for MmapStorage {
    // ── FFN ───────────────────────────────────────────────────────

    fn interleaved_q4k_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(BytesView<'_>, &str); FFN_COMPONENTS_PER_LAYER]> {
        let bytes = self.interleaved_q4k.as_ref()?;
        let manifest = self.interleaved_q4k_manifest.as_ref()?;
        let base = layer * FFN_COMPONENTS_PER_LAYER;
        if base + FFN_COMPONENTS_PER_LAYER > manifest.len() {
            return None;
        }
        // Validate every entry's range before forming the array.
        for i in 0..FFN_COMPONENTS_PER_LAYER {
            let (offset, length, _) = &manifest[base + i];
            checked_view(bytes, *offset, *length)?;
        }
        let out: [(BytesView<'_>, &str); FFN_COMPONENTS_PER_LAYER] = std::array::from_fn(|i| {
            let (offset, length, format) = &manifest[base + i];
            (BytesView::new(bytes, *offset, *length), format.as_str())
        });
        Some(out)
    }

    fn interleaved_q4k_whole_buffer(&self) -> Option<Bytes> {
        self.interleaved_q4k.clone()
    }

    fn interleaved_q4_whole_buffer(&self) -> Option<Bytes> {
        self.interleaved_q4.clone()
    }

    fn down_features_q4k_layer_data(&self, layer: usize) -> Option<(BytesView<'_>, &str, usize)> {
        let bytes = self.down_features_q4k.as_ref()?;
        let manifest = self.down_features_q4k_manifest.as_ref()?;
        let entry = manifest.get(layer)?;
        let view = checked_view(bytes, entry.offset, entry.length)?;
        Some((view, entry.format.as_str(), entry.padded_width))
    }

    fn gate_q4_layer_data(&self, layer: usize) -> Option<BytesView<'_>> {
        let bytes = self.gate_q4_bytes.as_ref()?;
        let entry = self.gate_q4_slices.get(layer)?;
        if entry.byte_len == 0 {
            return None;
        }
        checked_view(bytes, entry.byte_offset, entry.byte_len)
    }

    // ── Attention ─────────────────────────────────────────────────

    fn attn_q4k_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(BytesView<'_>, &str); ATTN_TENSORS_PER_LAYER]> {
        let bytes = self.attn_q4k.as_ref()?;
        let manifest = self.attn_q4k_manifest.as_ref()?;
        let base = layer * ATTN_TENSORS_PER_LAYER;
        if base + ATTN_TENSORS_PER_LAYER > manifest.len() {
            return None;
        }
        for i in 0..ATTN_TENSORS_PER_LAYER {
            let (offset, length, _) = &manifest[base + i];
            checked_view(bytes, *offset, *length)?;
        }
        let out: [(BytesView<'_>, &str); ATTN_TENSORS_PER_LAYER] = std::array::from_fn(|i| {
            let (offset, length, format) = &manifest[base + i];
            (BytesView::new(bytes, *offset, *length), format.as_str())
        });
        Some(out)
    }

    fn attn_q4_whole_buffer(&self) -> Option<Bytes> {
        self.attn_q4.clone()
    }

    fn attn_q4_layer_slices(
        &self,
        layer: usize,
    ) -> Option<[BytesView<'_>; ATTN_TENSORS_PER_LAYER]> {
        let bytes = self.attn_q4.as_ref()?;
        let manifest = self.attn_q4_manifest.as_ref()?;
        let base = layer * ATTN_TENSORS_PER_LAYER;
        if base + ATTN_TENSORS_PER_LAYER > manifest.len() {
            return None;
        }
        for i in 0..ATTN_TENSORS_PER_LAYER {
            let (offset, length) = &manifest[base + i];
            checked_view(bytes, *offset, *length)?;
        }
        let out: [BytesView<'_>; ATTN_TENSORS_PER_LAYER] = std::array::from_fn(|i| {
            let (offset, length) = &manifest[base + i];
            BytesView::new(bytes, *offset, *length)
        });
        Some(out)
    }

    fn attn_q8_layer_data(
        &self,
        layer: usize,
    ) -> Option<[(BytesView<'_>, BytesView<'_>); ATTN_TENSORS_PER_LAYER]> {
        let bytes = self.attn_q8.as_ref()?;
        let manifest = self.attn_q8_manifest.as_ref()?;
        let base = layer * ATTN_TENSORS_PER_LAYER;
        if base + ATTN_TENSORS_PER_LAYER > manifest.len() {
            return None;
        }
        for i in 0..ATTN_TENSORS_PER_LAYER {
            let (offset, vals_len, scales_len) = manifest[base + i];
            let vals_end = offset.checked_add(vals_len)?;
            let scales_end = vals_end.checked_add(scales_len)?;
            if scales_end > bytes.len() {
                return None;
            }
        }
        let out: [(BytesView<'_>, BytesView<'_>); ATTN_TENSORS_PER_LAYER] =
            std::array::from_fn(|i| {
                let (offset, vals_len, scales_len) = manifest[base + i];
                let vals = BytesView::new(bytes, offset, vals_len);
                let scales = BytesView::new(bytes, offset + vals_len, scales_len);
                (vals, scales)
            });
        Some(out)
    }

    // ── lm_head ───────────────────────────────────────────────────

    fn lm_head_q4_bytes(&self) -> Option<Bytes> {
        self.lm_head_q4.clone()
    }

    fn lm_head_f16_bytes(&self) -> Option<Bytes> {
        self.lm_head_f16.clone()
    }

    fn lm_head_f32_bytes(&self) -> Option<Bytes> {
        self.lm_head_f32.clone()
    }

    // ── Gate ──────────────────────────────────────────────────────

    fn gate_layer_view(&self, layer: usize) -> Option<GateLayerView<'_>> {
        let bytes = self.gate_bytes.as_ref()?;
        let slice = *self.gate_slices.get(layer)?;
        if slice.num_features == 0 {
            return None;
        }
        Some(GateLayerView {
            bytes,
            dtype: self.gate_dtype,
            slice,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::types::GateLayerSlice;

    /// Empty wrapper has every accessor returning `None`.
    #[test]
    fn empty_storage_returns_none_everywhere() {
        let s = MmapStorage::empty(2560);
        assert!(s.interleaved_q4k_layer_data(0).is_none());
        assert!(s.interleaved_q4k_whole_buffer().is_none());
        assert!(s.interleaved_q4_whole_buffer().is_none());
        assert!(s.down_features_q4k_layer_data(0).is_none());
        assert!(s.gate_q4_layer_data(0).is_none());
        assert!(s.attn_q4k_layer_data(0).is_none());
        assert!(s.attn_q4_whole_buffer().is_none());
        assert!(s.attn_q4_layer_slices(0).is_none());
        assert!(s.attn_q8_layer_data(0).is_none());
        assert!(s.lm_head_q4_bytes().is_none());
        assert!(s.lm_head_f16_bytes().is_none());
        assert!(s.lm_head_f32_bytes().is_none());
        assert!(s.gate_layer_view(0).is_none());
    }

    /// A `Bytes`-backed `MmapStorage` with a fabricated FFN Q4_K
    /// manifest must hand back the same byte ranges the manifest
    /// describes.
    #[test]
    fn ffn_q4k_layer_data_matches_manifest() {
        let mut s = MmapStorage::empty(8);
        // 3 layers × 3 components × 16 bytes = 144 bytes.
        let payload: Vec<u8> = (0u8..144).collect();
        s.interleaved_q4k = Some(Bytes::from(payload.clone()));
        s.interleaved_q4k_manifest = Some(
            (0..3 * FFN_COMPONENTS_PER_LAYER)
                .map(|i| (i * 16, 16, "Q4_K".to_string()))
                .collect(),
        );

        for layer in 0..3 {
            let arr = s.interleaved_q4k_layer_data(layer).expect("layer present");
            for (c, (view, fmt)) in arr.iter().enumerate() {
                let global = layer * FFN_COMPONENTS_PER_LAYER + c;
                let expected: &[u8] = &payload[global * 16..(global + 1) * 16];
                assert_eq!(view.as_slice(), expected, "layer {layer} comp {c}");
                assert_eq!(*fmt, "Q4_K");
            }
        }
    }

    /// A stale FFN Q4_K manifest entry that runs past the buffer
    /// must produce `None`, not a slice-bounds panic.
    #[test]
    fn ffn_q4k_layer_data_rejects_out_of_bounds_manifest() {
        let mut s = MmapStorage::empty(8);
        let payload: Vec<u8> = vec![0u8; 32];
        s.interleaved_q4k = Some(Bytes::from(payload));
        // gate fits, up fits, down points past the end.
        s.interleaved_q4k_manifest = Some(vec![
            (0, 8, "Q4_K".to_string()),
            (8, 8, "Q4_K".to_string()),
            (16, 32, "Q4_K".to_string()), // 16 + 32 = 48 > 32
        ]);
        assert!(s.interleaved_q4k_layer_data(0).is_none());
    }

    /// Attention Q8 layer data carries vals + scales spans; both must
    /// fit before any tuple is formed.
    #[test]
    fn attn_q8_layer_data_validates_combined_span() {
        let mut s = MmapStorage::empty(8);
        s.attn_q8 = Some(Bytes::from(vec![0u8; 1024]));
        // Q, K, V fit; O's scales run past 1024.
        s.attn_q8_manifest = Some(vec![
            (0, 64, 16),
            (100, 64, 16),
            (200, 64, 16),
            (1000, 64, 16), // 1000 + 64 + 16 = 1080 > 1024
        ]);
        assert!(s.attn_q8_layer_data(0).is_none());
    }

    /// `GateLayerView<'_>` borrows the dtype + slice + bytes
    /// together. The view is `Copy`, so multiple holders share the
    /// same borrow without refcount touches.
    #[test]
    fn gate_layer_view_round_trip() {
        let mut s = MmapStorage::empty(4);
        s.gate_bytes = Some(Bytes::from(vec![1u8, 2, 3, 4, 5, 6, 7, 8]));
        s.gate_dtype = StorageDtype::F16;
        s.gate_slices = vec![
            GateLayerSlice {
                float_offset: 0,
                num_features: 1,
            },
            GateLayerSlice {
                float_offset: 4,
                num_features: 1,
            },
        ];
        let v0 = s.gate_layer_view(0).expect("layer 0 present");
        assert_eq!(v0.dtype, StorageDtype::F16);
        assert_eq!(v0.slice.num_features, 1);
        let v0_copy = v0; // `Copy`, no clone needed.
        assert_eq!(v0.bytes.as_ptr(), v0_copy.bytes.as_ptr());
    }

    /// `gate_layer_view` returns `None` when the layer's
    /// `num_features` is zero — matches the substore convention for
    /// unowned layers in a sharded `--layers` slice.
    #[test]
    fn gate_layer_view_none_when_layer_unowned() {
        let mut s = MmapStorage::empty(4);
        s.gate_bytes = Some(Bytes::from(vec![0u8; 8]));
        s.gate_slices = vec![GateLayerSlice {
            float_offset: 0,
            num_features: 0,
        }];
        assert!(s.gate_layer_view(0).is_none());
    }

    /// `MmapStorage` clones cheaply — every field is `Bytes` /
    /// `Vec<...>` / `Copy`, so clone is a refcount bump per
    /// whole-file `Bytes`.
    #[test]
    fn mmap_storage_clones_via_refcount() {
        let mut s = MmapStorage::empty(4);
        s.lm_head_f16 = Some(Bytes::from(vec![1u8, 2, 3, 4]));
        let cloned = s.clone();
        assert_eq!(
            s.lm_head_f16.as_ref().unwrap().as_ptr(),
            cloned.lm_head_f16.as_ref().unwrap().as_ptr(),
        );
    }

    // ── Setter coverage ──────────────────────────────────────────────
    //
    // All `set_*` methods take an `Arc<Mmap>` (or `Arc<Vec<u8>>`).
    // Building real `Arc<Mmap>` instances from anonymous mmap is the
    // closest synthetic analogue of what loaders do; helper below
    // produces one with a known byte payload.

    fn arc_mmap_from(payload: &[u8]) -> Arc<memmap2::Mmap> {
        let mut anon = memmap2::MmapMut::map_anon(payload.len()).expect("anon mmap");
        anon.copy_from_slice(payload);
        let mmap = anon.make_read_only().expect("freeze");
        Arc::new(mmap)
    }

    #[test]
    fn set_interleaved_q4k_with_manifest_then_layer_data() {
        let payload: Vec<u8> = (0u8..96).collect();
        let mut s = MmapStorage::empty(8);
        s.set_interleaved_q4k(
            arc_mmap_from(&payload),
            Some(vec![
                (0, 16, "Q4_K".to_string()),
                (16, 16, "Q4_K".to_string()),
                (32, 16, "Q4_K".to_string()),
            ]),
        );
        assert!(s.has_interleaved_q4k());
        assert!(s.interleaved_q4k_whole_buffer().is_some());
        assert!(s.interleaved_q4k_whole_buffer_view().is_some());
        let arr = s.interleaved_q4k_layer_data(0).expect("layer 0");
        assert_eq!(arr[0].0.len(), 16);
        assert_eq!(arr[0].1, "Q4_K");
        // Layer 1 is past the 3 manifest entries → None.
        assert!(s.interleaved_q4k_layer_data(1).is_none());
    }

    #[test]
    fn set_interleaved_q4_whole_buffer_round_trip() {
        let payload = vec![7u8; 32];
        let mut s = MmapStorage::empty(8);
        s.set_interleaved_q4(arc_mmap_from(&payload));
        assert!(s.has_interleaved_q4());
        let buf = s.interleaved_q4_whole_buffer().expect("whole buffer");
        assert_eq!(buf.as_ref(), payload.as_slice());
        let view = s.interleaved_q4_whole_buffer_view().expect("view");
        assert_eq!(view.as_ref(), payload.as_slice());
    }

    #[test]
    fn set_down_features_q4k_then_layer_data() {
        let payload = vec![0u8; 64];
        let mut s = MmapStorage::empty(8);
        s.set_down_features_q4k(
            arc_mmap_from(&payload),
            vec![DownFeaturesQ4kEntry {
                offset: 0,
                length: 32,
                format: "Q4_K".to_string(),
                padded_width: 8,
            }],
        );
        assert!(s.has_down_features_q4k());
        let (view, fmt, padded) = s.down_features_q4k_layer_data(0).expect("layer 0");
        assert_eq!(view.len(), 32);
        assert_eq!(fmt, "Q4_K");
        assert_eq!(padded, 8);
    }

    #[test]
    fn set_attn_q4k_q4_q8_round_trips() {
        let payload = vec![0u8; 256];
        let mut s = MmapStorage::empty(8);

        s.set_attn_q4k(
            arc_mmap_from(&payload),
            Some(vec![
                (0, 16, "Q4_K".to_string()),
                (16, 16, "Q4_K".to_string()),
                (32, 16, "Q4_K".to_string()),
                (48, 16, "Q4_K".to_string()),
            ]),
        );
        assert!(s.has_attn_q4k());
        let q4k_arr = s.attn_q4k_layer_data(0).expect("attn q4k");
        assert_eq!(q4k_arr[0].0.len(), 16);

        s.set_attn_q4(
            arc_mmap_from(&payload),
            Some(vec![(0, 16), (16, 16), (32, 16), (48, 16)]),
        );
        assert!(s.has_attn_q4());
        let q4_arr = s.attn_q4_layer_slices(0).expect("attn q4");
        assert_eq!(q4_arr[0].len(), 16);
        assert!(s.attn_q4_whole_buffer().is_some());
        assert!(s.attn_q4_whole_buffer_view().is_some());

        s.set_attn_q8(
            arc_mmap_from(&payload),
            Some(vec![(0, 16, 4), (32, 16, 4), (64, 16, 4), (96, 16, 4)]),
        );
        assert!(s.has_attn_q8());
        let q8_arr = s.attn_q8_layer_data(0).expect("attn q8");
        assert_eq!(q8_arr[0].0.len(), 16);
        assert_eq!(q8_arr[0].1.len(), 4);
    }

    #[test]
    fn set_lm_head_variants_and_views() {
        let payload = vec![0u8; 32];
        let mut s = MmapStorage::empty(8);

        s.set_lm_head_f32(arc_mmap_from(&payload));
        assert!(s.has_lm_head_f32());
        assert!(s.lm_head_f32_bytes().is_some());
        assert!(s.lm_head_f32_view().is_some());

        s.set_lm_head_f16(arc_mmap_from(&payload));
        assert!(s.has_lm_head_f16());
        assert!(s.lm_head_f16_bytes().is_some());
        assert!(s.lm_head_f16_view().is_some());

        s.set_lm_head_q4_mmap(arc_mmap_from(&payload));
        assert!(s.has_lm_head_q4());
        assert!(s.lm_head_q4_bytes().is_some());
        assert!(s.lm_head_q4_view().is_some());
    }

    #[test]
    fn set_lm_head_q4_synth_round_trip() {
        let bytes = Arc::new(vec![1u8, 2, 3, 4, 5, 6, 7, 8]);
        let mut s = MmapStorage::empty(4);
        s.set_lm_head_q4_synth(bytes.clone());
        assert!(s.has_lm_head_q4());
        let view = s.lm_head_q4_view().expect("synth view");
        assert_eq!(view.as_ref(), bytes.as_slice());
    }

    #[test]
    fn set_gate_vectors_then_layer_view() {
        let payload = vec![0u8; 64];
        let mut s = MmapStorage::empty(4);
        s.set_gate_vectors(
            arc_mmap_from(&payload),
            StorageDtype::F16,
            vec![
                GateLayerSlice {
                    float_offset: 0,
                    num_features: 2,
                },
                GateLayerSlice {
                    float_offset: 8,
                    num_features: 2,
                },
            ],
        );
        assert!(s.has_gate_vectors());
        let view = s.gate_layer_view(0).expect("layer 0");
        assert_eq!(view.dtype, StorageDtype::F16);
        assert_eq!(view.slice.num_features, 2);
    }

    #[test]
    fn set_gate_q4_then_layer_data() {
        let payload = vec![0u8; 64];
        let mut s = MmapStorage::empty(4);
        s.set_gate_q4(
            arc_mmap_from(&payload),
            vec![GateQ4Slice {
                byte_offset: 0,
                byte_len: 32,
                num_features: 4,
            }],
        );
        assert!(s.has_gate_q4());
        let view = s.gate_q4_layer_data(0).expect("layer 0");
        assert_eq!(view.len(), 32);
    }

    /// Sweep test — exercise every trait method + has_* helper on a
    /// fully-populated `MmapStorage` so the trait `impl` block lights
    /// up under coverage.
    #[test]
    fn full_sweep_through_trait_and_helpers() {
        use crate::index::storage::vindex_storage::VindexStorage;
        let payload: Vec<u8> = (0u8..=255).collect();
        let mut s = MmapStorage::empty(8);
        s.set_interleaved_q4k(
            arc_mmap_from(&payload),
            Some(vec![
                (0, 16, "Q4_K".into()),
                (16, 16, "Q4_K".into()),
                (32, 16, "Q4_K".into()),
            ]),
        );
        s.set_interleaved_q4(arc_mmap_from(&payload));
        s.set_down_features_q4k(
            arc_mmap_from(&payload),
            vec![DownFeaturesQ4kEntry {
                offset: 0,
                length: 32,
                format: "Q4_K".into(),
                padded_width: 8,
            }],
        );
        s.set_gate_q4(
            arc_mmap_from(&payload),
            vec![GateQ4Slice {
                byte_offset: 0,
                byte_len: 32,
                num_features: 4,
            }],
        );
        s.set_attn_q4k(
            arc_mmap_from(&payload),
            Some(vec![
                (0, 16, "Q4_K".into()),
                (16, 16, "Q4_K".into()),
                (32, 16, "Q4_K".into()),
                (48, 16, "Q4_K".into()),
            ]),
        );
        s.set_attn_q4(
            arc_mmap_from(&payload),
            Some(vec![(0, 16), (16, 16), (32, 16), (48, 16)]),
        );
        s.set_attn_q8(
            arc_mmap_from(&payload),
            Some(vec![(0, 16, 4), (32, 16, 4), (64, 16, 4), (96, 16, 4)]),
        );
        s.set_lm_head_f32(arc_mmap_from(&payload));
        s.set_lm_head_f16(arc_mmap_from(&payload));
        s.set_lm_head_q4_mmap(arc_mmap_from(&payload));
        s.set_gate_vectors(
            arc_mmap_from(&payload),
            StorageDtype::F16,
            vec![GateLayerSlice {
                float_offset: 0,
                num_features: 2,
            }],
        );

        // Trait surface — owned-Bytes whole-buffer methods.
        assert!(s.interleaved_q4k_whole_buffer().is_some());
        assert!(s.interleaved_q4_whole_buffer().is_some());
        assert!(s.attn_q4_whole_buffer().is_some());
        assert!(s.lm_head_q4_bytes().is_some());
        assert!(s.lm_head_f16_bytes().is_some());
        assert!(s.lm_head_f32_bytes().is_some());

        // has_* helpers — both the populated and unpopulated
        // branches via a fresh empty.
        assert!(s.has_interleaved_q4k());
        assert!(s.has_interleaved_q4());
        assert!(s.has_down_features_q4k());
        assert!(s.has_gate_q4());
        assert!(s.has_gate_vectors());
        assert!(s.has_attn_q4k());
        assert!(s.has_attn_q4());
        assert!(s.has_attn_q8());
        assert!(s.has_lm_head_q4());
        assert!(s.has_lm_head_f16());
        assert!(s.has_lm_head_f32());

        let empty = MmapStorage::empty(8);
        assert!(!empty.has_interleaved_q4k());
        assert!(!empty.has_interleaved_q4());
        assert!(!empty.has_down_features_q4k());
        assert!(!empty.has_gate_q4());
        assert!(!empty.has_gate_vectors());
        assert!(!empty.has_attn_q4k());
        assert!(!empty.has_attn_q4());
        assert!(!empty.has_attn_q8());
        assert!(!empty.has_lm_head_q4());
        assert!(!empty.has_lm_head_f16());
        assert!(!empty.has_lm_head_f32());

        // Trait dispatch via `Arc<dyn VindexStorage>`.
        let dyn_storage: Arc<dyn VindexStorage> = Arc::new(s);
        assert!(dyn_storage.gate_q4_layer_data(0).is_some());
        assert!(dyn_storage.attn_q4k_layer_data(0).is_some());
        assert!(dyn_storage.attn_q4_layer_slices(0).is_some());
        assert!(dyn_storage.attn_q8_layer_data(0).is_some());
        assert!(dyn_storage.gate_layer_view(0).is_some());
        assert!(dyn_storage.down_features_q4k_layer_data(0).is_some());
    }

    /// `attn_q4_layer_slices` rejects an out-of-bounds manifest slice
    /// for the same reason `attn_q4k_layer_data` does — exercising the
    /// per-tensor checked_view branch.
    #[test]
    fn attn_q4_layer_slices_rejects_out_of_bounds() {
        let payload = vec![0u8; 64];
        let mut s = MmapStorage::empty(8);
        s.set_attn_q4(
            arc_mmap_from(&payload),
            Some(vec![(0, 16), (16, 16), (32, 16), (60, 32)]), // last spans past 64
        );
        assert!(s.attn_q4_layer_slices(0).is_none());
    }

    /// `down_features_q4k_layer_data` rejects an out-of-bounds entry.
    #[test]
    fn down_features_q4k_layer_data_rejects_out_of_bounds() {
        let payload = vec![0u8; 32];
        let mut s = MmapStorage::empty(8);
        s.set_down_features_q4k(
            arc_mmap_from(&payload),
            vec![DownFeaturesQ4kEntry {
                offset: 16,
                length: 32, // 16+32 = 48 > 32
                format: "Q4_K".into(),
                padded_width: 8,
            }],
        );
        assert!(s.down_features_q4k_layer_data(0).is_none());
    }

    /// `gate_q4_layer_data` rejects a slice whose `byte_len` is 0
    /// (typical for an unowned layer) or whose range overflows.
    #[test]
    fn gate_q4_layer_data_rejects_zero_or_overflow() {
        let payload = vec![0u8; 32];
        let mut s = MmapStorage::empty(8);
        // Zero-length slice.
        s.set_gate_q4(
            arc_mmap_from(&payload),
            vec![GateQ4Slice {
                byte_offset: 0,
                byte_len: 0,
                num_features: 0,
            }],
        );
        assert!(s.gate_q4_layer_data(0).is_none());

        // Overflow.
        s.set_gate_q4(
            arc_mmap_from(&payload),
            vec![GateQ4Slice {
                byte_offset: 16,
                byte_len: 32, // 16+32 = 48 > 32
                num_features: 4,
            }],
        );
        assert!(s.gate_q4_layer_data(0).is_none());
    }

    /// `from_substores` clones the underlying `Arc<Mmap>` handles,
    /// not the bytes themselves — verify by constructing substores
    /// with one mmap each and checking the storage's whole-buffer
    /// view points at the same memory region.
    #[test]
    fn from_substores_shares_bytes_with_substore_mmaps() {
        use crate::index::storage::ffn_store::FfnStore;
        use crate::index::storage::gate_store::GateStore;
        use crate::index::storage::projection_store::ProjectionStore;
        let payload = vec![0u8; 32];
        let mmap_arc = arc_mmap_from(&payload);

        let mut ffn = FfnStore::empty(1);
        ffn.interleaved_q4k_mmap = Some(mmap_arc.clone());
        let gate = GateStore::empty(1);
        let proj = ProjectionStore::empty();

        let s = MmapStorage::from_substores(&ffn, &gate, &proj, 8);
        assert!(s.has_interleaved_q4k());
        let view = s.interleaved_q4k_whole_buffer_view().expect("buf");
        // `Bytes::from_owner(arc.clone())` is zero-copy; the view's
        // pointer should point inside the mmap'd region.
        let mmap_ref: &[u8] = mmap_arc.as_ref();
        assert_eq!(view.as_ref().as_ptr(), mmap_ref.as_ptr());
    }
}
