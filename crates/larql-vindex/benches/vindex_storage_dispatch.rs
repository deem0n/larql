//! `VindexStorage` dispatch perf gate — Step 3 of the migration.
//!
//! Compares three shapes of FFN Q4_K layer-bytes access on the same
//! synthetic vindex:
//!
//! 1. **Direct**: today's `VectorIndex::interleaved_q4k_layer_data`
//!    (returns `Option<[(&[u8], &str); 3]>`). Borrows directly from
//!    the `Arc<Mmap>` held on `FfnStore`.
//! 2. **MmapStorage concrete**: `MmapStorage::interleaved_q4k_layer_data`
//!    (returns `Option<[(Bytes, &str); 3]>`). Same byte ranges; the
//!    `Bytes::slice` is O(1) refcounted but adds three atomic
//!    increments per call (one per component).
//! 3. **MmapStorage via `Arc<dyn VindexStorage>`**: same logic, behind a
//!    vtable pointer. This is the shape `VectorIndex.storage` will
//!    eventually hold.
//!
//! ## Acceptance bar
//!
//! `dyn` should be **within ~20% of direct** on this synthetic shape.
//! Production hot paths fetch one layer of bytes and reuse the slice
//! for every row in the inner decode loop, so any per-row trait cost
//! is amortised away — the bench measures the per-layer-fetch
//! overhead, which is the real cost. A blowout here (>2× of direct)
//! says step 4 needs `impl VindexStorage` (generic) on hot paths
//! instead of `dyn`.
//!
//! Run: `cargo bench -p larql-vindex --bench vindex_storage_dispatch`

use std::sync::Arc;

use bytes::Bytes;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use memmap2::{MmapMut, MmapOptions};

use larql_vindex::index::storage::vindex_storage::{MmapStorage, VindexStorage};

/// Build a synthetic `VectorIndex` with a populated FFN Q4_K manifest +
/// matching anonymous mmap. Shape mimics a small real model:
/// `num_layers` × 3 components × `intermediate × hidden` Q4_K bytes
/// (144 bytes per 256-element block).
///
/// Returns the index plus the `MmapStorage` snapshot (steady state for
/// the benchmark — both share the same underlying `Arc<Mmap>`).
fn build_fixture(
    num_layers: usize,
    intermediate: usize,
    hidden: usize,
) -> (larql_vindex::index::VectorIndex, MmapStorage) {
    use larql_models::quant::ggml::{K_QUANT_BLOCK_ELEMS, Q4_K_BLOCK_BYTES};

    let blocks_per_matrix = (intermediate * hidden) / K_QUANT_BLOCK_ELEMS;
    let bytes_per_matrix = blocks_per_matrix * Q4_K_BLOCK_BYTES;
    let bytes_per_layer = bytes_per_matrix * 3; // gate + up + down
    let total_bytes = bytes_per_layer * num_layers;

    // Anonymous mmap so this bench has no disk dependency.
    let mut mmap_mut = MmapOptions::new()
        .len(total_bytes)
        .map_anon()
        .expect("anon mmap");
    // Touch every page so we measure steady-state, not first-fault.
    for b in mmap_mut.iter_mut() {
        *b = 0;
    }
    let mmap: memmap2::Mmap = mmap_mut.make_read_only().expect("freeze");

    // Build the manifest: 3 entries per layer, each pointing at its
    // matrix's byte range. Layout: [L0 gate, L0 up, L0 down, L1 gate, ...].
    let mut manifest: Vec<(usize, usize, String)> = Vec::with_capacity(num_layers * 3);
    let mut offset = 0usize;
    for _layer in 0..num_layers {
        for _component in 0..3 {
            manifest.push((offset, bytes_per_matrix, "Q4_K".to_string()));
            offset += bytes_per_matrix;
        }
    }

    // Construct an inert `VectorIndex` then poke the FFN substore
    // fields — same pattern the production loader uses (see
    // `ffn_store/interleaved_q4k.rs::load_interleaved_q4k`), minus
    // the on-disk read.
    let mut index = larql_vindex::index::VectorIndex::empty(num_layers, hidden);
    index.ffn.interleaved_q4k_mmap = Some(Arc::new(mmap));
    index.ffn.interleaved_q4k_manifest = Some(manifest);

    let storage = MmapStorage::from_substores(
        &index.ffn,
        &index.gate,
        &index.projections,
        index.hidden_size,
    );

    (index, storage)
}

/// Zeroed-anon-mmap construction is fast but allocator-touchy; build
/// once per benchmark group, not per iteration.
fn bench_layer_data_dispatch(c: &mut Criterion) {
    // 3 layers is enough — the bench cycles through them inside the
    // measurement loop so the inner cost is layer fetch + slice
    // build, not page-fault spread.
    let num_layers = 3;
    // Intermediate / hidden close to a small real model
    // (Gemma 3 4B has hidden=2560, intermediate=15360 — too big for
    // a per-call bench; scale down so the fixture allocs in <100 ms
    // but the per-call shape is still 3 byte slices per layer).
    let intermediate = 1024;
    let hidden = 256;

    let (index, storage_concrete) = build_fixture(num_layers, intermediate, hidden);
    let storage_dyn: Arc<dyn VindexStorage> = Arc::new(storage_concrete.clone());

    let mut group = c.benchmark_group("interleaved_q4k_layer_data");
    group.bench_function(BenchmarkId::new("direct", num_layers), |b| {
        b.iter(|| {
            let mut sink = 0usize;
            for layer in 0..num_layers {
                let arr = index
                    .interleaved_q4k_layer_data(layer)
                    .expect("layer present");
                for (bytes, fmt) in arr.iter() {
                    sink ^= bytes.len() ^ fmt.len();
                }
            }
            black_box(sink);
        })
    });

    group.bench_function(BenchmarkId::new("mmap_storage_concrete", num_layers), |b| {
        b.iter(|| {
            let mut sink = 0usize;
            for layer in 0..num_layers {
                let arr = storage_concrete
                    .interleaved_q4k_layer_data(layer)
                    .expect("layer present");
                for (bytes, fmt) in arr.iter() {
                    sink ^= bytes.len() ^ fmt.len();
                }
            }
            black_box(sink);
        })
    });

    group.bench_function(BenchmarkId::new("mmap_storage_dyn", num_layers), |b| {
        b.iter(|| {
            let mut sink = 0usize;
            for layer in 0..num_layers {
                let arr = storage_dyn
                    .interleaved_q4k_layer_data(layer)
                    .expect("layer present");
                for (bytes, fmt) in arr.iter() {
                    sink ^= bytes.len() ^ fmt.len();
                }
            }
            black_box(sink);
        })
    });

    group.finish();
}

/// Same shape, but the inner per-row decode work amortises the
/// per-layer fetch — the realistic shape for production use. A
/// `Bytes::slice` per row would be visible here; per-layer fetches
/// shouldn't be.
fn bench_per_row_amortisation(c: &mut Criterion) {
    let num_layers = 3;
    let intermediate = 1024;
    let hidden = 256;
    let rows_per_layer = 64;

    let (index, storage_concrete) = build_fixture(num_layers, intermediate, hidden);
    let storage_dyn: Arc<dyn VindexStorage> = Arc::new(storage_concrete.clone());

    let mut group = c.benchmark_group("per_row_amortisation");
    group.bench_function("direct_per_row_byte_count", |b| {
        b.iter(|| {
            let mut sink = 0usize;
            for layer in 0..num_layers {
                let arr = index
                    .interleaved_q4k_layer_data(layer)
                    .expect("layer present");
                let (gate, _) = arr[0];
                // Inner row loop — what the actual decode kernel
                // does, just stripped to a byte read so we measure
                // dispatch, not decode.
                for row in 0..rows_per_layer {
                    sink = sink.wrapping_add(gate[row * 16] as usize);
                }
            }
            black_box(sink);
        })
    });
    group.bench_function("dyn_per_row_byte_count", |b| {
        b.iter(|| {
            let mut sink = 0usize;
            for layer in 0..num_layers {
                let arr = storage_dyn
                    .interleaved_q4k_layer_data(layer)
                    .expect("layer present");
                let gate: &Bytes = &arr[0].0;
                for row in 0..rows_per_layer {
                    sink = sink.wrapping_add(gate[row * 16] as usize);
                }
            }
            black_box(sink);
        })
    });
    group.finish();
}

criterion_group!(benches, bench_layer_data_dispatch, bench_per_row_amortisation);
criterion_main!(benches);
