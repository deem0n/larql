# Roadmap — larql-vindex

## Current State

- 173 unit tests + 148 integration tests passing on `larql-vindex`
  (321 total, all green); 211 on `larql-models`
- Folder layout: `index/{storage,compute,mutate}/`,
  `format/{huggingface,weights}/` decomposed; no .rs file > 750 lines
- Quant dispatch via `quant::registry` — adding the next format is one
  table entry, not eight match-arm edits
- Filename literals centralised in `format::filenames`
  (244 occurrences → one constant module)
- 3 storage formats: f32, Q8, Q4_K/Q6_K (Ollama-compatible)
- Mmap zero-copy with adaptive residency
- HNSW graph index wired into `gate_knn` (opt-in via `--hnsw`)
- Q4_K dequant cache LRU-bounded via `--max-q4k-cache-layers`
- Patch system for editable knowledge
- `make coverage` + `make coverage-summary` ready (`cargo-llvm-cov`
  install required)

## Round 2 cleanup — landed 2026-04-25

Most of the second-audit punch list is done in this session. Headlines:

| Item | Status |
|---|---|
| Add 8 missing filename constants | ✅ Done |
| Migrate 20 unmigrated `Q4_K`/`Q6_K` dispatch sites | ✅ Done |
| Replace 2× `unwrap_or("Q4_K")` silent fallbacks | ✅ Done |
| Rename top-level `vindex/src/storage/` → `engine/` | ✅ Done (back-compat alias kept) |
| Rename duplicate `fp4_storage.rs` files | ✅ Done — `format/fp4_codec.rs` + `index/storage/fp4_store.rs` |
| Merge `ffn_data.rs` into `ffn_store.rs` | ✅ Done |
| Inline `gate_trait.rs` (198 L pass-through) | ✅ Done — moved into `index/core.rs` |
| Rename `accessors.rs` → `gate_accessors.rs` | ✅ Done |
| Split `config/types.rs` (624 L) | ⏸ **Deferred to next session** — needs careful inter-type reference mapping |

321 vindex tests + 232 inference tests pass; whole workspace builds.

## P0: Round 2 cleanup (2026-04-25 second audit)

The first audit shipped (registry, filenames module, substores, file
splits, golden tests, coverage). A second audit on the post-refactor
state caught residue from that work plus paths the first scan missed.

### Add 8 missing filename constants
**Impact**: Closes the "wrong filename → silent fallback" class for the
files the first audit didn't grep for
**Effort**: Low
**Status**: Not started

The first migration covered the 19 names in the original list but
missed:

| Constant | Occurrences | Why missed |
|---|---|---|
| `LM_HEAD_BIN` | **10×** | not in first grep — used in extract, walk, build_lm_head_q4, convert_q4k, load, checksums, huggingface, write_f32, lm_head |
| `GATE_VECTORS_FP4_BIN` | 7× | FP4 family (exp 26) landed after baseline |
| `DOWN_FEATURES_FP8_BIN` | 5× | same |
| `UP_FEATURES_FP4_BIN` | 4× | same |
| `ATTN_WEIGHTS_Q4_BIN` + `ATTN_WEIGHTS_Q4_MANIFEST_JSON` | 1× each | low-traffic sibling of Q4K manifest |
| `ATTN_WEIGHTS_Q8_BIN` + `ATTN_WEIGHTS_Q8_MANIFEST_JSON` | 1× each | same |

Add to `format::filenames`, migrate the 28 sites.

### Migrate ~20 unmigrated `"Q4_K"`/`"Q6_K"` dispatch sites
**Impact**: Eliminates the dispatch-by-string-literal class the
registry was meant to subsume
**Effort**: Low–Medium
**Status**: Not started

Of 50 surviving format-tag literals, ~20 are still **dispatch sites**
in `match` arms / `if format == "Q4_K"` conditionals — the registry
covers the call shape, but these specific sites weren't migrated.
Each should become a `registry::lookup(tag)?` lookup with explicit
error on unknown tags.

### Replace `unwrap_or("Q4_K")` silent fallbacks
**Impact**: Malformed manifest no longer silently assumes Q4_K
**Effort**: Tiny
**Status**: Not started

`ffn_store.rs:276` and `attn.rs:93` both contain
`unwrap_or("Q4_K")` reads off manifest JSON. A bad / missing
`format` field today silently defaults to Q4_K, which is exactly the
silent-fallback class the registry was supposed to kill. Replace with
`registry::lookup(...)?` returning a parse error.

## P1: Folder + file layout polish (round 2)

### Rename top-level `vindex/src/storage/` → `engine/`
**Impact**: Removes the `storage/` clash with `index/storage/`
**Effort**: Low (pure rename)
**Status**: Not started

Two `storage/` directories at different levels of the tree confuse
navigation:
- `vindex/src/storage/` — `engine.rs`, `epoch.rs`, `memit_store.rs`,
  `status.rs` — that's **L0/L1/L2 lifecycle**, not data layout.
- `vindex/src/index/storage/` — gate / ffn / projection / metadata
  substores — actual data access.

The top-level dir's contents are about the `StorageEngine` lifecycle
(epoch, compaction, MEMIT solver). Rename to `engine/` so the path
becomes `crate::engine::StorageEngine`. `index/storage/` keeps its
name (correct for what it holds).

### Rename the duplicate `fp4_storage.rs` files
**Impact**: Removes the same-filename-different-concerns confusion
**Effort**: Low (pure rename)
**Status**: Not started

- `format/fp4_storage.rs` → `format/fp4_codec.rs` (write/read codec
  + layout math; *encoding* concern)
- `index/storage/fp4_storage.rs` → `index/storage/fp4_store.rs`
  (runtime `Fp4Storage` struct + row accessors; matches `gate_store`,
  `ffn_store` convention)

### Merge `ffn_data.rs` into `ffn_store.rs`
**Impact**: Removes the awkward data/impl split inside `index/storage/`
**Effort**: Low
**Status**: Not started

`ffn_data.rs` (~80 L) carries the `FfnStore` struct + `Clone` impl;
`ffn_store.rs` (~720 L) carries the `impl VectorIndex` accessor /
loader methods that touch FfnStore fields. They cite each other in
every method. Merge — same shape as `gate_store.rs` (which lives in
one file).

### Inline `gate_trait.rs` (198 L of one-liner pass-through)
**Impact**: One source of truth for `GateIndex` impl; less file
juggling when searching for a method
**Effort**: Low
**Status**: Not started

Every method in `gate_trait.rs` is `fn foo(...) { self.foo(...) }` —
identity forwarding because `impl GateIndex for VectorIndex` lives in
a separate file from the methods themselves. After the refactor the
ceremony has zero benefit. Move the impl block back next to the
methods (in `core.rs` or per-concern in `compute/`) and delete the
file. `PatchedVindex`'s `overlay_gate_trait.rs` stays — its methods
do real overlay-vs-base lookup work.

### Rename `accessors.rs` → `gate_accessors.rs`
**Impact**: Generic name disambiguated; future `ffn_accessors.rs` etc.
follow the same pattern
**Effort**: Tiny
**Status**: Not started

`index/storage/accessors.rs` is gate-specific (gate_vector,
gate_vectors_at, warmup, describe_ffn_backend) but the name implies a
catch-all accessor module.

## P2: Config split + forward scalability

### Split `config/types.rs` (624 L, 15 unrelated types)
**Impact**: Future quant/MoE additions scoped to one file
**Effort**: Medium (move-only)
**Status**: Not started

Split into:
- `config/index.rs` — `VindexConfig`, `VindexLayerInfo`, `DownMeta*`
- `config/quantization.rs` — `QuantFormat`, `Precision`,
  `ProjectionFormat`, `Projections`, `Fp4Config`
- `config/model.rs` — `VindexModelConfig` (model family, MoE, rope, …)
- `config/compliance.rs` — `ComplianceGate`, `LayerBands`

`mod.rs` re-exports the previous flat surface for back-compat.

### Parallelize gate KNN for batch inference
**Impact**: 2–4× prefill throughput on multi-token batches
**Effort**: Medium
**Status**: Forward-looking

`gate_matmul` already runs across all positions in one BLAS call but
the per-position top-K selection is sequential. Rayon-shard the
selection across rows (or fold into a single batched argpartial). Not
urgent — Metal kernel work (Q6_K dequant + 8-rows/TG) is the bigger
throughput lever.

### `VindexStorage` trait abstraction
**Impact**: Lets Redis / S3 / GPU-residency backends plug in
**Effort**: Medium
**Status**: Forward-looking

The substore extraction got most of the way there. Formalise a
sealed `VindexStorage` trait (mmap-agnostic row accessor) so Q4K row
reads can route through Redis-cached or S3-buffered backends without
walk-kernel changes.

### Expert-level sharding protocol
**Impact**: Unlocks > 256-expert MoE sharding-within-layer
**Effort**: Medium
**Status**: Forward-looking

Today `larql-router` shards by layer, not by expert ID within a
layer. For DeepSeek-V4-class models (1K+ experts) experts need to
shard across servers. Add an `ExpertRoute` message type to
`larql-router-protocol` and wire `GridState` dispatch.

### Won't-fix for now

- **`detect.rs` (1391 L) split** — cohesive; single entry point
  dispatching to 12 architectures. Splitting fragments without
  modularity gain. Wait for a second detection system before
  revisiting.

## P0: Code-quality cleanup (2026-04-25 audit)

Findings from the codebase-wide audit (six parallel agents covering
quant extensibility, magic strings, modularity, folder layout, test
coverage, and docs). Verdict: well-engineered crate with three
concentrated structural debts.

### `quant::registry` — single dispatch table for all GGML formats
**Impact**: Adding the next quant (Q5_K / Q3_K / …) drops from 8 files
to 3; deletes ~12 silent-fallback `_ => None` match arms in walk.rs
**Effort**: Medium
**Status**: Not started

Today three separate format enums coexist (`QuantFormat` in
`config/types.rs`, `QuantBlockFormat` in `format/weights/write.rs`, a
third in `larql-compute/pipeline.rs`). Block-byte sizes (144 for Q4_K,
210 for Q6_K) appear inline as magic numbers across `walk.rs`. 25+
bare `"Q4_K"` / `"Q6_K"` literals across the workspace.

Build a `crates/larql-vindex/src/quant/registry.rs` carrying a
`QuantFormatInfo` table: `tag`, `block_elements`, `bytes_per_block`,
function pointers for `dequantize` / `row_dot` / `row_scaled_add`.
`walk.rs` match arms collapse to `registry::lookup(tag)?` calls.
Adding Q5_K = one new entry plus the codec functions.

### `format::filenames` — one home for the 244 filename literals
**Impact**: Eliminates the "wrong filename → silent fallback" class
**Effort**: Low
**Status**: Not started

`"index.json"` (77 occurrences), `"tokenizer.json"` (56),
`"gate_vectors.bin"` (49), and friends are scattered across vindex,
cli, server, inference. A typo today silently triggers a fallback
codepath. Consolidate into `crates/larql-vindex/src/format/filenames.rs`
and migrate callers.

### Doc + bench freshness
**Impact**: README / PERFORMANCE / SPEC currently lag code by ~3 weeks
**Effort**: Low
**Status**: Not started

- README: test counts say "106 / 104"; actual is **304** (167 unit +
  137 integration)
- PERFORMANCE.md: still cites 51.9 tok/s; current `larql bench` is
  **68.7 tok/s** Gemma 3 4B Metal Q4K
- FFN_VINDEX_UNIFICATION_SPEC.md: aspirational, not flagged as such
  (KnnStore is still in `lib.rs`)
- Inline rustdoc + ADRs are current (no action needed)

## P1: Modularity + test depth

### Split `index/` along storage / compute / mutate seams — DONE
**Impact**: Unblocks the god-struct extraction; no behaviour change
**Effort**: Medium total (file moves + impl-block surgery)
**Status**: ✅ Complete (2026-04-25)

What landed:
- `storage/` (mmap loaders, decode caches, residency, FFN store, gate
  store, attn, lm_head, FP4 storage)
- `compute/` (gate KNN dispatch, HNSW, MoE router, Q4_K codec dispatch)
- `mutate/` (INSERT/DELETE, NDJSON loaders, persistence)
- 11 files moved + 4 net new (`gate_store`, `ffn_store`,
  `q4k_dispatch`, plus the existing `gate_knn`)
- gate.rs (992) → `compute/gate_knn.rs` (615) + `storage/gate_store.rs`
  (446)
- walk.rs (862) → `storage/ffn_store.rs` (720) +
  `compute/q4k_dispatch.rs` (168)
- All 321 tests pass; backwards-compatible aliases on `index/mod.rs`
  keep external paths resolving

`index/` is partitioned by *operation* (`gate.rs`, `walk.rs`, `attn.rs`,
`lm_head.rs`) but those files mix mmap slicing, KNN compute, and
caching. `gate.rs` is 992 lines covering all three concerns; `walk.rs`
is 912 the same way. Proposed layout:

```
index/
├── core.rs            — slimmed VectorIndex (composes substores)
├── types.rs / gate_trait.rs / mod.rs
├── storage/           — mmap + slicing + caches + LRU bookkeeping
│   ├── mmap_util.rs   (moved from src/)
│   ├── gate_store.rs
│   ├── ffn_store.rs
│   ├── projection_store.rs   (lm_head + attn)
│   └── caches.rs
├── compute/           — pure dispatch
│   ├── gate_knn.rs
│   ├── gate_walk.rs
│   ├── hnsw_dispatch.rs
│   └── lm_head_knn.rs
└── mutate/            — INSERT / DELETE / heap promotion
```

### `VectorIndex` god struct → composed substores — DONE
**Impact**: 35+ flat fields collapsed to four typed stores
**Effort**: Large
**Status**: ✅ Complete (2026-04-25)

What landed:
- `GateStore` (storage/gate_store.rs) — gate matrix mmap, decode caches,
  HNSW index. Owns 13 fields.
- `FfnStore` (storage/ffn_data.rs) — FFN mmaps, Q4_K dequant cache,
  FP4 storage. Owns 10 fields.
- `ProjectionStore` (storage/projection_store.rs) — lm_head + attention
  weight mmaps. Owns 10 fields.
- `MetadataStore` (storage/metadata_store.rs) — down_meta, overrides.
  Owns 4 fields.
- `VectorIndex` itself now holds 5 shape fields + 4 substores. Each
  store owns its own `Clone` impl (Arc-shares mmaps, resets caches).
- 321 tests pass; field names preserved within stores so a future PR
  can drop redundant `gate_` / `q4k_ffn_` prefixes if desired.

```rust
pub struct VectorIndex {
    config:      VindexConfigCore,
    gate:        GateStore,
    ffn:         FfnStore,
    projections: ProjectionStore,
    metadata:    MetadataStore,
    fp4_storage: Option<Arc<Fp4Storage>>,
}
```

`gate_trait.rs` stops being a thin pass-through over field accesses;
each store owns its caches and LRU.

### GGML quant round-trip tests
**Impact**: Catches the silent-fallback class via codec checks
**Effort**: Small
**Status**: Not started

Today there are zero round-trip tests for Q4_0 / Q4_K / Q6_K / Q8.
FP4 / FP8 have them via `larql-models`. Add
`crates/larql-vindex/tests/quant_roundtrip.rs`: quantize → dequantize
→ assert close-enough per format with frozen tolerance bounds.

### End-to-end golden pipeline test
**Impact**: One assertion catches all serialization regressions
**Effort**: Medium
**Status**: Not started

Fixture under `crates/larql-vindex/tests/golden/`: 3-layer synthetic
safetensors → extract → save → load (mmap) → KNN → patch → save →
reload → re-run KNN. Frozen SHA256 of bytes + bit-exact KNN result.
Also add: mmap-zero-copy regression (`assert_eq!(gate_heap_bytes(),
0)` after f16 mmap load), LRU-eviction-under-load (1000 random
queries, cap=4, 60 layers, observe never > 4).

### Benches for the 2026-04-25 work
**Impact**: Numbers behind ROADMAP claims become measurable
**Effort**: Small
**Status**: Not started

- `benches/hnsw_decode.rs` — brute vs HNSW at 10K / 28K / 131K
  features, recall %, build cost
- `benches/q4k_cache.rs` — cold dequant vs cached hit per layer, LRU
  eviction overhead (validates the "30× win" amortisation claim)
- `benches/q4k_prefetch.rs` — first-token cold-page latency with /
  without `prefetch_interleaved_q4k_layer`

## P2: Ergonomics + cosmetics

### Split oversized files — DONE
- ✅ `format/huggingface.rs` (1366) → `huggingface/{mod,download,publish,discovery}.rs`
- ✅ `format/weights/write.rs` (1249) → `weights/{write_f32,write_q4k}.rs`
- ✅ `larql-models/src/quant/ggml.rs` (1352) → `quant/ggml/{mod,legacy,q4_k,q6_k,quantize}.rs`

### Naming pass — one referent per format concept — DONE
- ✅ Rust types: `Q4K` (was 8 × `Q4k` before, all renamed)
- ✅ Snake-case identifiers: `q4k`
- ✅ Serialized strings: `"Q4_K"` (only in registry)

### Coverage tooling — DONE
- ✅ `make coverage` — HTML report under `coverage/`
- ✅ `make coverage-summary` — terminal-only digest
- ✅ Both fail-fast with install hint when `cargo-llvm-cov` is missing
- Override scope with `make coverage CRATE=larql-models`

## P0: Decode-path performance

Items raised by the 2026-04-25 perf audit (see PERFORMANCE.md and the
`gpu_forward_gap` memo). Vindex-side only — Metal kernel work lives in
larql-compute's roadmap.

### Bound the Q4_K dequant cache (LRU like gate cache) — DONE
**Impact**: Caps CPU-fallback RAM at a configurable budget (worst-case
today: 10.7 GB on 4B / ~110 GB on 31B if all layers cache fully)
**Effort**: Low
**Status**: ✅ Complete (2026-04-25)
- `set_q4k_ffn_cache_max_layers` API + LRU eviction in `walk.rs`
- `q4k_ffn_cache_stats` diagnostic, surfaced via `larql bench -v`
- `--max-q4k-cache-layers N` flag on `larql serve`
- Confirmed empirically: Metal full-K decode never populates the cache
  (`q4k_ffn_cache after larql-metal: 0 populated slots, 0.0 MB`)

**Finding from 2026-04-25 audit**: the Metal hot path never populates
`q4k_ffn_cache` (`larql bench --backends metal -v` reports
`q4k_ffn_cache after larql-metal: 0 populated slots, 0.0 MB`). The
full-K Metal branch in `walk_ffn/sparse.rs:84-117` streams Q4_K bytes
through `q4k_matmul_transb` and bypasses `q4k_ffn_layer` entirely. The
dequant cache only fires in the CPU per-position fallback at
`walk_ffn/sparse.rs:145` (`hits.len() >= 512 && down_native.is_none()`)
— and there it's a 30× win because one 614 ms layer-dequant is
amortised across thousands of feature reads per token.

So the cache is correct, not pathological. What's missing is an upper
bound: a long-running CPU-only server can grow it to all 34 layers ×
105 MB on Gemma 3 4B (10.7 GB) or 60 layers × 1.85 GB on 31B (~110 GB).
Mirror the existing gate-cache pattern (`gate_cache_max_layers`,
`gate_cache_lru` in `index/core.rs` / `gate.rs:80`) for the Q4_K FFN
cache:

1. Add `q4k_ffn_cache_max_layers` (atomic) + `q4k_ffn_cache_lru`
   (Mutex<VecDeque<usize>>) to `VectorIndex`.
2. On insert in `q4k_ffn_layer`, push the layer to the LRU and evict
   from the front when the cap is exceeded; clear the evicted layer's
   slot triple.
3. Expose `set_q4k_ffn_cache_max_layers(n)` + a `--max-q4k-cache-layers
   N` flag on `larql serve` and any other long-running CLI.
4. Default cap = 0 (unbounded — keeps current behaviour). Recommend 8
   for a CPU-only Gemma 3 4B server (≈ 840 MB ceiling for the down
   leg; gate/up dequant aren't on the hot path).

### Q4_K interleaved madvise + per-layer prefetch — DONE
**Impact**: Free win on cold-page first-token latency; small steady-state
**Effort**: Low
**Status**: ✅ Complete (2026-04-25)
- `prefetch_interleaved_q4k_layer` added to `walk.rs` (manifest-aware
  for mixed Q4_K/Q6_K layouts; uniform-stride fallback otherwise)
- Wired into `walk_ffn/sparse.rs` (hot path) and
  `walk_ffn/interleaved_q4k.rs` (dequant fallback)
- Trait surface: `GateIndex::prefetch_interleaved_q4k_layer`

### Audit `save_gate_vectors` 1.4 → 2.0 ms regression — DONE (false alarm)
**Status**: ✅ Resolved (2026-04-25) — not a regression
- Criterion's own change report flagged `p = 0.21 > 0.05` ("No change
  in performance detected"); the eyeballed 40% drift was inside the CI
- `git log` shows no functional changes to the save path since
  2026-04-07 (only sibling additions: `set_up_vector`, etc.)

### Lift gate KNN out of brute-force on the decode hot path — DONE
**Impact**: 64-expert MoE 230 → ~60 ms gate KNN/layer (search + re-rank)
**Effort**: Medium
**Status**: ✅ Complete (2026-04-25)
- `gate_knn_hnsw` was already routed in `gate_knn` behind
  `hnsw_enabled`. Two production fixes landed:
  1. **Zero-copy view** for f32-mmap layers — was cloning the entire
     gate matrix per query (~100 MB on Gemma 3 4B) defeating mmap
  2. **Abs-magnitude ranking parity** — brute uses `|dot|`, HNSW
     ranked by signed dot, systematically dropping large-negative
     features. Now oversamples 4× and re-ranks at the seam to match
- New end-to-end smoke test (`gate_knn_hnsw_smoke`) verifies
  enable/disable cycle restores brute results bit-for-bit
- `--hnsw` + `--hnsw-ef-search` flags on `larql serve`
- **Caveat**: HNSW is approximate (recall 80–95%). Default off; opt-in
  for high-feature MoE where brute gemv dominates

### Bench rig hygiene — fail fast under host contention — DONE
**Impact**: Makes regression detection meaningful again
**Effort**: Low
**Status**: ✅ Complete (2026-04-25)
- `vindex_scaling` calls `refuse_under_contention()` at every bench
  group entry; refuses with non-zero exit if `pgrep -fl
  'larql-(server|router)'` matches
- `LARQL_BENCH_ALLOW_DAEMONS=1` env override for intentional in-flight
  benching
- `make bench-vindex` (synthetic, safe) and `make bench-vindex-scaling`
  (production-dim, daemon-checked) split as separate targets

## P0: Support Cached Layer Decode

### Store pre-computed residuals for template-fixed layers (L0-12)
**Impact**: Enables 155+ tok/s decode (skip 13 of 21 layers)  
**Effort**: Medium  
**Status**: Not started (infrastructure ready — CachedLayerGraph in larql-inference)

The vindex needs to store cached residuals per template. During extraction, run one forward pass per template through L0-12 and save the output residual. At decode time, look up the cached residual instead of computing 13 layers.

### Wire Q4_K FFN consumption (interleaved_q4k.bin) — DONE
**Impact**: Match Ollama's exact FFN quantization  
**Effort**: Medium  
**Status**: ✅ Complete (2026-04-07)

Added `load_interleaved_q4k()`, `has_interleaved_q4k()`, `interleaved_q4k_mmap_ref()` to vindex.
Inference `predict_honest` now prefers Q4_K FFN (`interleaved_q4k.bin`) over Q4_0.
Format tag (`ffn_format`) passed through `FullPipelineLayer` to compute for shader dispatch.

### GGUF Q4_K format option (144 bytes vs 148 bytes)
**Impact**: Direct compatibility with llama.cpp weight files  
**Effort**: Low  
**Status**: Quantizer ready in larql-compute (`quantize_q4_k_gguf`)

Add option to store attention weights in GGUF-canonical 144-byte Q4_K format (packed scales+mins in 12 bytes) instead of our 148-byte format.

## P1: Production Hardening

### HuggingFace resolution in Vindexfile
**Effort**: Medium  
**Status**: TODO in `vindexfile/mod.rs:162`

FROM directive in Vindexfile should resolve `hf://user/repo` paths.

### Streaming extraction checkpoints
**Effort**: Medium  
**Status**: Not started

Save extraction progress between layers so interrupted builds can resume.

### Q4_K FFN in vindex
**Effort**: Low  
**Status**: Not started (Q4_0 interleaved exists)

Currently FFN gate/up/down stored as Q4_0. Switch to Q4_K (matching Ollama) for better precision at similar size.

## P2: Research

### Multi-model vindex
Store features from multiple models in one vindex. Compare representations across architectures.

### Incremental extraction
Add new layers/features to an existing vindex without full rebuild.

## Completed

| Item | Date | Impact |
|------|------|--------|
| Core VectorIndex with mmap | 2026-03 | Foundation |
| Gate KNN (brute-force + BLAS) | 2026-03 | Walk engine |
| Walk FFN (per-feature down/up vectors) | 2026-03 | Sparse inference |
| Binary down_meta format | 2026-03 | 5x compression vs JSONL |
| F16 storage + decode cache | 2026-03 | 2x smaller gate vectors |
| Interleaved layout (gate\|up\|down packed) | 2026-04 | Reduced TLB thrash |
| Q4_0 gate vectors + interleaved | 2026-04 | 7x smaller gates |
| HNSW graph index | 2026-04 | Sub-linear KNN |
| Adaptive residency (pin/evict) | 2026-04 | Memory budget management |
| Patch system (PatchedVindex) | 2026-04 | Editable knowledge |
| MoE expert routing | 2026-04 | Mixtral/DeepSeek support |
| Q4_K/Q6_K attention weights | 2026-04 | Ollama-compatible |
| Q8 attention weights | 2026-04 | Higher precision option |
| Streaming extraction (mmap, per-layer) | 2026-04 | ~2 GB peak RAM |
| Safety doc for mmap_optimized | 2026-04-07 | Clippy compliance |
| VindexPatch::is_empty() | 2026-04-07 | API completeness |
| Q4_K FFN loader + wiring | 2026-04-07 | `interleaved_q4k.bin` end-to-end |
| Quantizer single source of truth | 2026-04-07 | Builder uses larql-compute (ADR-008) |
| Example cleanup (13→11) | 2026-04-07 | Removed Q4_0 attn + Q4_0 interleaved |
| 8 ADRs documented | 2026-04-07 | All major decisions recorded |
| PERFORMANCE.md + format alignment | 2026-04-07 | Fresh benchmarks, verified pipeline |
