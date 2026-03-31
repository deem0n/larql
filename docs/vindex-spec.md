# Vindex — The Queryable Model Format

**Version:** 0.2  
**Author:** Chris Hay  
**Date:** 2026-03-31  
**Status:** Draft  
**Implementation:** `larql-vindex` crate (Rust)  
**Companion specs:** LQL Language Specification, larql-knowledge Specification

---

## 1. What is a Vindex?

A vindex (vector index) is a directory containing a neural network's weights reorganised for queryability. The model IS the database — each weight matrix is stored once in its optimal format for the operations it supports.

**Key principle:** `gate_vectors.bin` IS W_gate. `embeddings.bin` IS W_embed. They are the canonical storage, not copies. COMPILE reads `gate_vectors.bin` to reconstruct W_gate in safetensors format. No data is stored twice.

### Weights separated by function, not by file size

Existing model formats — safetensors, GGUF, ONNX — store all weights together. Sharded formats split by file size, not by purpose. Every format assumes you want all the weights or none of them.

Vindex separates weights by what you want to do with the model:

| Intent | Weights needed | Size (4B, f16) | What you skip |
|--------|---------------|-----------------|---------------|
| Browse knowledge | Gate + embeddings + down metadata | ~3 GB | Attention, up projections, norms, LM head |
| Run inference | + attention weights | ~6 GB | Up projections, norms, LM head |
| Edit and recompile | All weights | ~10 GB | Nothing |

This separation exists because these operations touch fundamentally different weight matrices. Querying what a model knows about France uses gate vectors and embeddings — a dot product against pre-extracted rows. It never touches attention weights. Running inference additionally needs attention for token routing. Recompiling needs everything to reconstruct the full safetensors.

The consequences are significant:

- **A 70B model's knowledge is browsable on a machine with 32GB RAM.** The browse-only core is ~25GB. Running that same model for inference requires multi-GPU.
- **Knowledge hosting costs scale like a CDN, not like a GPU cluster.** The browse files are static and immutable. $240/year instead of $50,000/year.
- **Distributed architectures become natural.** The knowledge server hosts gate vectors. The inference server hosts attention weights. Each tier loads only what it needs. No server pays for weights it doesn't use.
- **Patching is lightweight.** A vindex patch modifies gate vectors and down metadata. It never touches attention weights. A 10KB patch file can change what a model knows without touching 97% of its parameters.

No other model format makes this possible because no other format separates weights by function.

**Current limitation:** `model_weights.bin` duplicates gate and down weights that are already in the query index. This is a known issue — the deduplicated format (Section 10.2) splits model_weights into component files that exclude already-extracted data.

---

## 1.1 Architecture Support

The vindex format is model-agnostic. It stores weights by function (gate, down, embed, attention) not by architecture name. Any transformer model with a gated FFN (gate + up + down projections) and standard attention (Q, K, V, O) can be extracted into a vindex.

**Supported model families:**

| Family | Models | FFN Type | Notes |
|--------|--------|----------|-------|
| Gemma | Gemma 2/3 (2B-27B) | Gated (gate + up + down) | GeGLU activation, GQA attention |
| Llama | Llama 2/3 (7B-405B) | Gated (gate + up + down) | SiLU activation, GQA attention |
| Mistral | Mistral 7B | Gated (gate + up + down) | Sliding window attention |
| Mixtral | Mixtral 8x7B, 8x22B | MoE (8 experts × gate + up + down) | Sparse MoE, top-2 routing |
| Qwen | Qwen 2/2.5 (0.5B-72B) | Gated (gate + up + down) | SiLU activation |
| Phi | Phi 2/3 (2.7B-14B) | Gated (gate + up + down) | Partial attention |
| DeepSeek | DeepSeek V2/V3 | MoE (shared + routed experts) | Fine-grained MoE, shared expert |
| GPT-OSS | GPT-OSS-20B | MoE (8 experts × gate + up + down) | Test model for MoE validation |
| GPT-2 | GPT-2 (117M-1.5B) | Dense (W_in + W_out) | Soft gating via GELU — W_in rows as gate vectors |

### MoE Support

Mixture-of-Experts models have multiple FFN "experts" per layer, each with its own gate + up + down weight matrices. A router network selects which top-K experts process each token.

**In the vindex, each expert's features are separate entries in the same gate_vectors.bin.** A dense model has `intermediate_size` features per layer. An MoE model with N experts has `N × intermediate_size` features per layer. The gate KNN naturally selects features across all experts — no router needed for browse operations.

```
Dense (Gemma 4B):    10,240 features per layer → 348,160 total
MoE (Mixtral 8x7B):  8 × 14,336 = 114,688 features per layer → 3,670,016 total
MoE (GPT-OSS-20B):   8 × intermediate_size features per layer
```

**Vindex layout for MoE:**

```
gate_vectors.bin:
  [Layer 0, Expert 0: intermediate_size × hidden_size]
  [Layer 0, Expert 1: intermediate_size × hidden_size]
  ...
  [Layer 0, Expert 7: intermediate_size × hidden_size]
  [Layer 1, Expert 0: intermediate_size × hidden_size]
  ...
```

The `index.json` layer info includes expert metadata:

```json
{
  "layers": [
    {
      "layer": 0,
      "num_experts": 8,
      "num_features_per_expert": 14336,
      "num_features": 114688,
      "offset": 0,
      "length": 1174405120
    }
  ],
  "moe_config": {
    "num_experts": 8,
    "top_k_experts": 2,
    "has_shared_expert": false,
    "router_type": "top_k_softmax"
  }
}
```

**Feature addressing for MoE:** Each feature has a (layer, expert, feature) triple instead of (layer, feature). DESCRIBE shows which expert a feature belongs to:

```
larql> DESCRIBE "France";
France
  Edges (L14-27):
    capital     → Paris        gate=1436.9  L27 E3  (probe)
    language    → French       gate=35.2    L24 E5  (probe)
    continent   → Europe       gate=14.4    L25 E1  (probe)
```

The `E3` tag shows this feature lives in Expert 3 at Layer 27. This validates the mechanistic finding: if capital features cluster in specific experts, the model has domain-specialised experts. If they're spread across all experts, the experts are context-specialists (which GPT-OSS-20B research showed).

**Operations by mode:**

| Operation | Dense Model | MoE Model |
|-----------|-------------|-----------|
| WALK / DESCRIBE | gate KNN across all features | gate KNN across all experts' features |
| INFER | attention + FFN | attention + router + selected expert FFNs |
| COMPILE | reconstruct gate/up/down | reconstruct per-expert gate/up/down + router |
| SELECT | filter by layer, feature | filter by layer, expert, feature |
| SHOW FEATURES | per layer | per layer per expert |

**The router weights** (small network that selects top-K experts) are stored in `model_weights.bin` alongside attention weights. They're only needed for INFER — browse operations query all experts simultaneously via gate KNN and don't need routing.

**Test model: GPT-OSS-20B.** This is the validation target for MoE support because:
1. Chris built it and has complete mechanistic understanding of its routing
2. Prior research proved experts are context-specialists (trigram-based), not domain-specialists
3. 96% of routing signal comes from attention output — validatable via EXPLAIN INFER
4. The RLHF-hardened gates (soft→hard transition) should show as sharper feature clusters
5. The virtual expert architecture (Python functions as MoE experts) can be compared against the weight-based experts

**Architecture variations handled:**

- **GQA (Grouped Query Attention):** Fewer KV heads than Q heads. Stored as-is in attention weights.
- **Sliding window attention:** Window size stored in `model_config`. Walk/DESCRIBE unaffected (FFN-only).
- **Tied embeddings:** LM head shares embedding matrix. `lm_head.bin` omitted, `embeddings.bin` used for both.
- **RoPE variants:** Base frequency and scaling stored in `model_config`. Used by INFER, irrelevant for browse.
- **Vocabulary size:** Varies (32K for Llama, 256K for Gemma). Stored in config, `embeddings.bin` sized accordingly.
- **MoE shared expert (DeepSeek):** One expert is always active alongside routed experts. Stored as Expert 0 with a `shared: true` flag.
- **Fine-grained MoE (DeepSeek V3):** 256 experts with top-8 routing. Same structure, just more features per layer.

### Dense FFN Support (GPT-2, GPT-J, GPT-NeoX, OPT)

Dense FFN models have no explicit gate matrix — the FFN is `W_out @ GELU(W_in @ x)`. However, `W_in` rows serve the same functional role as gate vectors: each row defines a direction in residual space that activates a feature. GELU provides soft selection instead of the hard gating of SiLU(gate) × up.

**Extraction:** Use `W_in` rows as `gate_vectors.bin` and `W_out` columns as down projections. The KNN is identical: `W_in @ residual` gives per-feature activation scores. Top-K selection works the same way.

**Differences from gated FFN:**
- **Less sparse.** Gated FFNs zero out ~80% of features via the gate. Dense FFNs have all features contributing (just scaled by GELU). The top-K boundary is softer — more features have non-trivial scores.
- **No up projection.** Dense FFNs have W_in (gate equivalent) and W_out (down equivalent) but no W_up. COMPILE is simpler — only two matrices per layer.
- **Noisier DESCRIBE.** More features fire for any entity, so DESCRIBE shows more edges. Higher top-K threshold recommended.

**Vindex layout:** Identical to gated FFN. `gate_vectors.bin` contains W_in rows. `down_meta` computed from W_out columns. All query operations work unchanged.

```
Dense FFN (GPT-2 1.5B):  4 × 1600 = 6,400 features per layer × 48 layers = 307,200 total
Gated FFN (Gemma 4B):    10,240 features per layer × 34 layers = 348,160 total
```

**Not yet supported:**
- **Non-transformer models:** Mamba, RWKV, etc. Not in scope — different architecture entirely.

The `model_config` field in `index.json` stores architecture-specific parameters that INFER needs but browse operations ignore:

```json
"model_config": {
    "model_type": "gemma3",
    "head_dim": 256,
    "num_q_heads": 8,
    "num_kv_heads": 4,
    "rope_base": 10000.0,
    "rope_scaling_factor": 1.0,
    "sliding_window": 1024,
    "attention_type": "gqa",
    "activation": "geglu",
    "tie_word_embeddings": true
}
```

For MoE models, the config includes MoE-specific fields:

```json
"model_config": {
    "model_type": "mixtral",
    "head_dim": 128,
    "num_q_heads": 32,
    "num_kv_heads": 8,
    "rope_base": 1000000.0,
    "attention_type": "gqa",
    "activation": "silu",
    "moe": {
        "num_experts": 8,
        "top_k": 2,
        "shared_expert": false,
        "router_type": "top_k_softmax"
    }
}
```

Layer band boundaries (`layer_bands` in config) are model-specific and auto-detected during EXTRACT. Larger models have more layers and wider bands, but the three-band structure (syntax → knowledge → output) is consistent across architectures.

---

## 1.2 How Vindex Relates to Other Formats

Vindex serves a different purpose from existing model formats. Where safetensors and GGUF optimise for inference, and SAE features optimise for interpretability, vindex optimises for knowledge access — treating the model's weights as a queryable, editable database.

### Where each format excels

**safetensors** is the standard distribution format. It stores weight tensors efficiently with fast random access and memory mapping. Every inference framework reads it. When you need to run a model, safetensors is the right starting point. Vindex doesn't replace safetensors for inference — it's what you produce *from* safetensors when you want to query the knowledge.

**GGUF** is optimised for efficient CPU inference with quantisation. It compresses models to 2-4× smaller than f16 while maintaining quality. When you need fast inference on consumer hardware, GGUF is excellent. Vindex doesn't do quantised inference — it separates the knowledge browsing from inference entirely.

**ONNX** targets cross-framework deployment with computation graph optimisation. It's the right choice when you need to deploy a model across different runtimes. Vindex isn't a deployment format — it's an inspection and editing format.

**SAE features** (sparse autoencoders) are the closest relative to vindex. Both extract meaningful features from model weights. The key difference: SAEs learn features post-hoc by training an additional autoencoder on model activations. This produces interpretable features but requires significant compute to train and is read-only. Vindex extracts features directly from the existing weight matrices — no additional training. And vindex features are writable: INSERT adds a fact, COMPILE produces a new model.

### Comparative strengths

| Dimension | safetensors | GGUF | SAE features | Vindex |
|-----------|-------------|------|-------------|--------|
| **Primary purpose** | Model distribution | Efficient CPU inference | Interpretability research | Knowledge access + editing |
| **Query without inference** | No | No | Partially (need forward pass + SAE) | Yes (gate KNN, 33ms) |
| **Edit a fact** | Retrain (hours, GPU) | Retrain + requantise | Not supported | INSERT + COMPILE (seconds, CPU) |
| **Browse-only size (4B)** | 8 GB (full model) | 2-4 GB (full model) | 2-10 GB (on top of model) | 3 GB (gate + embed only) |
| **GPU required** | For inference | No | For training SAE | No (browse), optional (infer) |
| **Typed knowledge edges** | No | No | Manual labelling | Yes (probe + Wikidata + WordNet) |
| **Hostable on CDN** | No (needs compute) | No (needs compute) | No (needs compute) | Yes (static files, dot products) |
| **Round-trip to weights** | N/A (is the weights) | Lossy (quantisation) | No | Yes (EXTRACT → edit → COMPILE) |

### Performance comparison

To answer "What does the model know about France?":

| Format | Method | Time | Hardware |
|--------|--------|------|----------|
| safetensors | Full forward pass with probing prompts | ~800ms | GPU |
| GGUF Q4 | Quantised forward pass | ~200ms | CPU |
| SAE features | Forward pass + SAE decode + manual inspection | ~2s | GPU |
| Vindex browse | Gate KNN across 14 knowledge layers | ~33ms | CPU |
| Vindex infer | Attention + walk FFN (full prediction) | ~200ms | CPU |

Vindex browse is faster because it skips the forward pass entirely. It's a matrix-vector dot product against pre-extracted gate vectors — the same operation the model performs internally, but without the surrounding computation.

### Scaling by model size

| Model | safetensors | GGUF Q4 | Vindex browse | Vindex full |
|-------|-------------|---------|---------------|-------------|
| 4B (Gemma 3) | 8 GB, 1 GPU | 2 GB, CPU | 3 GB, CPU | 10 GB, CPU |
| 8B (Llama 3) | 16 GB, 1 GPU | 4 GB, CPU | 5 GB, CPU | 18 GB, CPU |
| 70B (Llama 3) | 140 GB, multi-GPU | 35 GB, CPU | ~25 GB, CPU | ~80 GB, CPU |
| 405B (Llama 3) | 800 GB, GPU cluster | 200 GB, multi-node | ~120 GB, distributed | ~500 GB, distributed |

At larger scales, the browse-only vindex stays CPU-accessible while inference formats require increasingly expensive hardware. A 70B model's knowledge is browsable on a single 32GB machine — no GPU needed. Running that same model for inference requires multi-GPU.

### Hosting economics

| Format | Hosting requirement | Annual cost (1 model) | Marginal cost per user |
|--------|--------------------|-----------------------|-----------------------|
| safetensors | GPU inference cluster | ~$50,000 | Linear (more GPUs) |
| GGUF | CPU compute nodes | ~$5,000 | Linear (more CPUs) |
| Vindex browse | Static file server / CDN | ~$240 | Near-zero (CDN-cacheable) |
| Vindex infer | CPU compute + static files | ~$2,500 | Linear for infer, zero for browse |

The fundamental difference: every other format requires compute per query. Vindex browse serves static files — the "computation" is a dot product the client performs locally, or a lightweight server performs once and caches. This makes the cost model more like hosting a website than running an inference service.

### What this means in practice

A researcher wanting to understand what GPT knows about France today would: load the model (~8GB GPU memory), craft prompts, run inference, analyse activations. With SAE features, they'd additionally train a sparse autoencoder and inspect which features fire.

With vindex: `DESCRIBE "France"` — 33ms, no GPU, no model loaded, typed relations with confidence scores. The knowledge is in the index. The model is the database.

A developer wanting to add a fact today would: curate a dataset, fine-tune (hours, GPU cluster), evaluate, deploy. With ROME/MEMIT, they'd run a gradient-based rank-1 update (minutes, GPU).

With vindex: `INSERT ("John Coyle", "lives-in", "Colchester"); COMPILE;` — seconds, CPU, produces a standard safetensors file loadable by any framework.

These are complementary tools solving different problems. safetensors is the distribution format. GGUF is the deployment format. SAEs are the research tool. Vindex is the knowledge access layer.

---

## 2. File Layout

### 2.1 Current Layout

```
gemma3-4b.vindex/
│
│  # ═══ Query Index (browse-only core) ═══
│  # WALK, DESCRIBE, SELECT, EXPLAIN WALK use only these files.
│
├── gate_vectors.bin          # W_gate rows per layer (KNN index)
├── embeddings.bin            # W_embed matrix (token lookup)
├── down_meta.jsonl           # Per-feature output metadata (top tokens + scores)
│
│  # ═══ Inference & Compile Weights ═══
│  # Contains weights NOT in the query index (attention, up, norms, lm_head)
│  # KNOWN ISSUE: currently also contains gate/down weights (duplicated).
│  # See Section 10.2 for the planned deduplicated format.
│
├── model_weights.bin         # All model weights (when built with --include-weights)
├── weight_manifest.json      # Key → offset/length mapping into model_weights.bin
│
│  # ═══ Metadata & Labels ═══
│
├── index.json                # VindexConfig: layers, sizes, extract level
├── tokenizer.json            # HuggingFace tokenizer
├── relation_clusters.json    # Cluster centres, labels, counts (from build)
├── feature_clusters.jsonl    # Per-feature cluster assignments (from build)
└── feature_labels.json       # Probe-confirmed labels (from larql-knowledge)
```

### 2.2 Planned Deduplicated Layout

The target format eliminates duplication by splitting weights into component files. Each weight is stored exactly once:

```
gemma3-4b.vindex/
│
│  # ═══ Query Index (browse-only core, ~3 GB f16) ═══
│  # These ARE the FFN gate + embedding weights in queryable format.
│  # WALK, DESCRIBE, SELECT, EXPLAIN WALK use only these files.
│
├── gate_vectors.bin          # W_gate rows per layer (IS the gate weight matrix)
├── embeddings.bin            # W_embed matrix (IS the embedding weight matrix)
├── down_meta.bin             # Per-feature top token IDs + scores (binary, ~2 MB)
│
│  # ═══ Inference Weights (~3 GB f16, for INFER) ═══
│  # Only attention weights — gate is already in the query index.
│
├── attn_weights.bin          # Q, K, V, O projection matrices per layer
│
│  # ═══ Compile Weights (~4 GB f16, for COMPILE) ═══
│  # Additional weights needed to reconstruct full safetensors.
│
├── up_weights.bin            # W_up per layer
├── down_weights.bin          # W_down per layer (full vectors, not just top-K metadata)
├── norms.bin                 # LayerNorm parameters per layer
├── lm_head.bin               # Output projection (or shared with embeddings if tied)
│
│  # ═══ Metadata & Labels ═══
│
├── index.json                # VindexConfig: layers, sizes, extract level, manifest
├── tokenizer.json            # HuggingFace tokenizer
├── relation_clusters.json    # Cluster centres, labels, counts
├── feature_clusters.jsonl    # Per-feature cluster assignments
└── feature_labels.json       # Probe-confirmed labels (from larql-knowledge)
```

**Size by use case (planned, f16):**

| Use Case | Files Loaded | Size (f16) |
|----------|-------------|------------|
| Browse only (WALK, DESCRIBE, SELECT) | gate + embed + down_meta + labels | ~3 GB |
| Browse + Inference (+ INFER) | Above + attn_weights | ~6 GB |
| Full (+ COMPILE) | All files | ~10 GB |

**Compared to original model:**

```
Original HuggingFace model (f16):    ~8 GB
Vindex browse-only (f16):            ~3 GB  (62% smaller)
Vindex browse + infer (f16):         ~6 GB  (25% smaller)
Vindex full (f16):                   ~10 GB (25% larger, but queryable + compilable)
```

---

## 3. Extract Levels

A vindex can be built at three levels, each adding more weight components:

| Level | LQL Syntax | Components | Size (f16 est.) | Enables |
|-------|-----------|------------|-----------------|---------|
| Browse | `EXTRACT MODEL ... INTO ...` | gate + embed + down_meta | ~3 GB | WALK, DESCRIBE, SELECT, EXPLAIN WALK |
| Inference | `... WITH INFERENCE` | + attention weights | ~6 GB | + INFER, EXPLAIN INFER |
| All | `... WITH ALL` | + up, down (full), norms, lm_head | ~10 GB | + COMPILE |

**Current implementation:** A single `--include-weights` flag that bundles all weights into `model_weights.bin`. The tiered split is planned (Section 10.2).

```rust
pub enum ExtractLevel {
    Browse,      // Default: gate + embed + down_meta
    Inference,   // + attention weights
    All,         // + all remaining weights for COMPILE
}
```

The `index.json` stores the extract level:

```json
{
  "extract_level": "inference",
  "has_model_weights": true
}
```

---

## 4. Binary Formats

### 4.1 gate_vectors.bin

Raw f32 floats, contiguous, no headers. Layer-by-layer concatenation.

**Layout:**
```
[Layer 0: num_features × hidden_size × f32]
[Layer 1: num_features × hidden_size × f32]
...
[Layer N: num_features × hidden_size × f32]
```

**Per-layer shape:** `(intermediate_size, hidden_size)` — one row per FFN feature.

**Byte order:** Native endian (platform-dependent, typically little-endian).

**Index:** `VindexLayerInfo` in `index.json` stores byte offset and length for each layer, enabling random access without reading the entire file.

**Usage:** Gate KNN — `residual × gate_vectors^T` finds which features fire. This is both the gate computation and the similarity search: same operation, different framing.

**Deduplication note:** This file IS the W_gate weight matrix. COMPILE reads it directly to reconstruct the safetensors gate tensor. No separate gate weight storage needed.

### 4.2 embeddings.bin

Raw f32 floats, no headers. Single contiguous matrix.

**Shape:** `(vocab_size, hidden_size)` in row-major order.

**Usage:** Token embedding lookup. Multiply by `embed_scale` (from config) to match gate vector magnitudes. For multi-token entities, average the scaled embeddings.

**Deduplication note:** This file IS the W_embed weight matrix. If the model ties embeddings and LM head, this also serves as the output projection.

### 4.3 down_meta.jsonl (current)

NDJSON (newline-delimited JSON). One record per feature per layer.

**Record format:**
```json
{"l":0,"f":512,"t":"the","i":278,"c":3.45,"k":[{"t":"the","i":278,"s":3.45},{"t":"and","i":345,"s":2.12}]}
```

| Field | Type | Description |
|-------|------|-------------|
| `l` | usize | Layer index |
| `f` | usize | Feature index within layer |
| `t` | string | Top output token (decoded) |
| `i` | u32 | Top output token ID |
| `c` | f32 | C-score (selectivity of top token) |
| `k` | array | Top-K output tokens with scores |

Each `k` entry: `{"t": token_string, "i": token_id, "s": logit_score}`

**Usage:** After gate KNN finds which features fire, down_meta provides what each feature outputs. This is the edge label: gate fires on entity → down outputs target.

**Size issue:** 160 MB as JSONL. See Section 10.1 for planned binary format (~2 MB).

### 4.4 model_weights.bin (current)

Binary container for model weights. Used by INFER and COMPILE.

**Format:** Sequential f32 tensors, no headers. Layout described by `weight_manifest.json`.

**Contents (per layer):**
- Attention: Q, K, V, O projection matrices
- FFN: gate, up, down weight matrices
- Norms: input_layernorm, post_attention_layernorm vectors

**Plus global:**
- Final norm vector
- LM head (output projection) matrix

**Known duplication:** Currently contains W_gate and W_down which are also in `gate_vectors.bin` and partially in `down_meta.jsonl`. The planned deduplicated format (Section 10.2) eliminates this.

### 4.5 weight_manifest.json

JSON array mapping tensor keys to byte offsets in `model_weights.bin`.

```json
[
  {
    "key": "layers.0.self_attn.q_proj.weight",
    "kind": "tensor",
    "shape": [2048, 2560],
    "offset": 0,
    "length": 20971520
  },
  {
    "key": "norm.weight",
    "kind": "vector",
    "shape": [2560],
    "offset": 1234567890,
    "length": 10240
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `key` | string | Tensor key (architecture-specific naming) |
| `kind` | string | `"tensor"` (2D) or `"vector"` (1D) |
| `shape` | [usize] | Dimensions |
| `offset` | u64 | Byte offset into model_weights.bin |
| `length` | u64 | Byte length (element_count × 4) |

### 4.6 index.json (VindexConfig)

```json
{
  "version": 2,
  "model": "google/gemma-3-4b-it",
  "family": "gemma3",

  "source": {
    "huggingface_repo": "google/gemma-3-4b-it",
    "huggingface_revision": "a1b2c3d4e5f6",
    "safetensors_sha256": "e3b0c44298fc1c149afb...",
    "extracted_at": "2026-03-31T14:22:00Z",
    "larql_version": "0.1.0"
  },

  "checksums": {
    "gate_vectors.bin": "a1b2c3d4...",
    "embeddings.bin": "e5f6a7b8...",
    "down_meta.jsonl": "c9d0e1f2...",
    "model_weights.bin": "13141516..."
  },

  "num_layers": 34,
  "hidden_size": 2560,
  "intermediate_size": 10240,
  "vocab_size": 262144,
  "embed_scale": 50.596,
  "extract_level": "inference",
  "has_model_weights": true,

  "layer_bands": {
    "syntax": [0, 13],
    "knowledge": [14, 27],
    "output": [28, 33]
  },

  "layers": [
    {"layer": 0, "num_features": 10240, "offset": 0, "length": 104857600},
    {"layer": 1, "num_features": 10240, "offset": 104857600, "length": 104857600}
  ],

  "down_top_k": 10,

  "model_config": {
    "model_type": "gemma3",
    "head_dim": 256,
    "num_q_heads": 8,
    "num_kv_heads": 4,
    "rope_base": 10000.0,
    "sliding_window": 1024
  }
}
```

**`source`** — Provenance tracking. Records exactly which model checkpoint was extracted, when, and by which version of LARQL. The `safetensors_sha256` is a hash of the original weight files, enabling verification that a vindex was built from a specific model release.

**`checksums`** — SHA256 hash of each binary file. Enables `larql verify` to confirm integrity after download or transfer. Computed during EXTRACT, checked optionally during USE.

```bash
larql verify gemma3-4b.vindex
# Checking gate_vectors.bin... OK (3.32 GB, sha256 matches)
# Checking embeddings.bin... OK (2.50 GB, sha256 matches)
# Checking down_meta.jsonl... OK (160 MB, sha256 matches)
# All files verified.
```

**`layer_bands`** — Model-specific boundaries for DESCRIBE layer grouping and layer-aware label matching. Default boundaries are computed during EXTRACT by analysing feature distributions. They can be overridden manually or refined by `SHOW LAYERS` analysis.

### 4.7 Version History and Compatibility

| Version | Changes | Backwards Compatible |
|---------|---------|---------------------|
| 1 | Original format: gate + embed + down_meta JSONL | — |
| 2 | Added extract_level, layer_bands, model_config, source, checksums | Yes — v1 vindexes load with default layer_bands |
| 3 | Binary down_meta (planned) | Yes — engine reads both JSONL and binary |
| 4 | Split weight files (planned) | Yes — engine checks for split files, falls back to model_weights.bin |
| 5 | f16 storage (planned) | Yes — dtype field in config, engine casts on load |

**Compatibility policy:** The engine reads any version ≤ current. Older vindexes load with sensible defaults for missing fields:
- Missing `layer_bands` → auto-computed as `syntax=[0, N/3], knowledge=[N/3, 2N/3], output=[2N/3, N-1]`
- Missing `source` → empty (provenance unknown)
- Missing `checksums` → skip verification
- Missing `extract_level` → inferred from `has_model_weights`

**Migration:** `larql upgrade <vindex>` adds missing fields to index.json without touching binary files. Non-destructive, append-only metadata update.

### 4.7 Label Files

**relation_clusters.json** — Discovered relation clusters from offset-direction clustering:
```json
{
  "k": 512,
  "labels": ["capital", "language", "morphological", ...],
  "counts": [142, 89, 1203, ...],
  "top_tokens": [["Paris", "Berlin", "Tokyo"], ["French", "English", "German"], ...]
}
```

**feature_clusters.jsonl** — Per-feature cluster assignments:
```json
{"l":26,"f":9515,"c":42}
```
Maps (layer, feature) → cluster_id.

**feature_labels.json** — Probe-confirmed labels (from larql-knowledge pipeline):
```json
{
  "26:9515": "capital",
  "24:4532": "language",
  "25:3877": "continent"
}
```
Key format: `"layer:feature"`. These override cluster labels at query time.

---

## 5. Core Operations

### 5.1 Load

```rust
let config = load_vindex_config(&path)?;
let mut cb = SilentLoadCallbacks;
let index = VectorIndex::load_vindex(&path, &mut cb)?;
```

Loading reads:
1. `index.json` → `VindexConfig` (layer offsets, model metadata)
2. `gate_vectors.bin` → per-layer `Array2<f32>` matrices (via offset lookup)
3. `down_meta.jsonl` → per-feature `FeatureMeta` (top token, c_score, top_k)

Embeddings, tokenizer, and label files are loaded separately on demand:
```rust
let (embed, embed_scale) = load_vindex_embeddings(&path)?;
let tokenizer = load_vindex_tokenizer(&path)?;
let labels = load_feature_labels(&path)?;
```

Model weights for INFER are loaded lazily — only when an INFER statement is executed.

### 5.2 Gate KNN

```rust
let hits: Vec<(usize, f32)> = index.gate_knn(layer, &residual, top_k);
```

Computes `gate_matrix @ residual` via BLAS matmul, returns top-K feature indices sorted by absolute dot product. This is both the gate computation and the nearest-neighbor search.

**Performance:** 0.98ms per layer on CPU (M-series Mac). 34 layers = 33ms for a full walk.

### 5.3 Walk

```rust
let trace: WalkTrace = index.walk(&query, &layers, top_k);
```

Runs gate KNN at each layer, annotates hits with down_meta (what each feature outputs). Returns a `WalkTrace` with per-layer `WalkHit` entries:

```rust
pub struct WalkHit {
    pub layer: usize,
    pub feature: usize,
    pub gate_score: f32,
    pub meta: FeatureMeta,
}
```

### 5.4 Describe

```rust
let edges: Vec<DescribeEdge> = describe_entity(
    &entity, &index, &embed, embed_scale, &tokenizer, &labels, &clusters, opts
);
```

Multi-layer gate KNN across the specified layer band, with:
- Probe label lookup (highest priority)
- Cluster label lookup (fallback)
- Edge merging across layers (same target token → combined entry)
- Noise filtering (non-Latin tokens, low gate scores)
- Source tagging (`(probe)`, `(cluster)`, or blank for TF-IDF)

### 5.5 Mutate

```rust
// Insert: set gate vector + metadata for a feature
index.set_gate_vector(layer, feature, &gate_vec);
index.set_feature_meta(layer, feature, meta);

// Delete: clear metadata
index.delete_feature_meta(layer, feature);

// Find unused slot
let slot = index.find_free_feature(layer);

// Save changes to disk
index.save_down_meta(&path)?;
index.save_gate_vectors(&path)?;
VectorIndex::save_config(&config, &path)?;
```

### 5.6 Compile

Reconstructs HuggingFace-compatible model files from the vindex.

```rust
compile_vindex(&vindex_path, &output_path, format)?;
```

**Algorithm:**
1. Read `gate_vectors.bin` → reshape rows into W_gate matrices per layer
2. Read `model_weights.bin` → extract W_down, W_up, attention weights, norms via manifest
3. For any features modified by INSERT/DELETE/UPDATE: use the vindex's updated vectors
4. Write safetensors (or GGUF) with standard HuggingFace naming conventions
5. Copy `config.json` and `tokenizer.json` from vindex metadata

**Output directory:**
```
gemma3-4b-edited/
  model.safetensors         # Reconstructed weights
  config.json               # Model architecture config
  tokenizer.json            # Tokenizer
  tokenizer_config.json     # Tokenizer config
```

Loadable by any framework:
```python
model = AutoModelForCausalLM.from_pretrained("gemma3-4b-edited/")
```

**Round-trip test:** EXTRACT → COMPILE (no edits) should produce weights identical to the original within floating-point tolerance.

**Requires:** Extract level `All` — needs W_up, norms, and LM head in addition to the query index and attention weights.

### 5.7 Patches

Patches are lightweight, shareable diffs that modify a vindex without changing the base files. They capture INSERT, DELETE, and UPDATE operations as a portable file that can be applied to any vindex built from the same base model.

#### 5.7.1 Patch File Format (.vlp)

```json
{
  "version": 1,
  "base_model": "google/gemma-3-4b-it",
  "base_checksum": "a1b2c3d4...",
  "created_at": "2026-03-31T15:00:00Z",
  "larql_version": "0.1.0",
  "description": "Medical knowledge: drug interactions and side effects",
  "author": "medical-team",
  "tags": ["medical", "pharmacology"],
  "operations": [
    {
      "op": "insert",
      "layer": 26,
      "feature": 8821,
      "relation": "side_effect",
      "entity": "aspirin",
      "target": "bleeding",
      "confidence": 0.85,
      "gate_vector_b64": "<base64 encoded f32 × hidden_size>",
      "down_meta": {"t": "bleeding", "i": 12847, "c": 4.2}
    },
    {
      "op": "update",
      "layer": 27,
      "feature": 9515,
      "gate_vector_b64": "<base64 encoded f32 × hidden_size>",
      "down_meta": {"t": "Paris", "i": 8921, "c": 5.1}
    },
    {
      "op": "delete",
      "layer": 24,
      "feature": 1337,
      "reason": "hallucinated fact"
    }
  ]
}
```

**Size:** A single fact is approximately 10 KB (one gate vector at 2,560 × 4 bytes ≈ 10 KB + metadata). A patch with 1,000 facts is approximately 10 MB. Compared to the full model at 8 GB, this is 1/800th the size.

#### 5.7.2 LQL Patch Operations

```sql
-- ═══ Creating Patches ═══

-- Start a patch session (edits captured, base vindex unchanged)
BEGIN PATCH "medical-knowledge.vlp";

INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "side_effect", "bleeding");

INSERT INTO EDGES (entity, relation, target)
    VALUES ("aspirin", "treats", "headache");

INSERT INTO EDGES (entity, relation, target)
    VALUES ("ibuprofen", "side_effect", "nausea");

-- Save the patch (base vindex is NOT modified)
SAVE PATCH;
-- Saved: medical-knowledge.vlp (3 operations, 30 KB)

-- ═══ Applying Patches ═══

-- Apply a single patch
USE "gemma3-4b.vindex";
APPLY PATCH "medical-knowledge.vlp";
-- Applied: 3 operations (3 inserts)

-- Stack multiple patches (applied in order)
APPLY PATCH "medical-knowledge.vlp";
APPLY PATCH "fix-hallucinations.vlp";
APPLY PATCH "company-facts.vlp";

-- See active patches
SHOW PATCHES;
-- 1. medical-knowledge.vlp     (3 inserts, 30 KB)
-- 2. fix-hallucinations.vlp    (20 deletes, 2 KB)
-- 3. company-facts.vlp         (200 inserts, 2 MB)
-- Total: 223 operations

-- Remove a patch
REMOVE PATCH "fix-hallucinations.vlp";

-- ═══ Baking Down ═══

-- Flatten patches into a new clean vindex
COMPILE CURRENT INTO VINDEX "gemma3-4b-medical.vindex";
-- New vindex with all patches applied. No patch stack. Clean base.

-- Or compile straight to safetensors for deployment
COMPILE CURRENT INTO MODEL "gemma3-4b-medical/" FORMAT safetensors;
-- Standard HuggingFace model with patched knowledge. Loadable by any framework.

-- ═══ Extracting Patches from Diffs ═══

-- Compare two vindexes, extract the diff as a patch
DIFF "gemma3-4b.vindex" "gemma3-4b-medical.vindex"
    INTO PATCH "medical-changes.vlp";
-- Extracted: 223 features that differ → portable patch file

-- This means anyone who has edited a model can share their changes
-- without redistributing the full model.
```

#### 5.7.3 Patch Application

Patches modify the in-memory vindex without touching base files:

```rust
pub struct PatchedVindex {
    base: VectorIndex,                          // Immutable base
    patches: Vec<VindexPatch>,                  // Applied in order
    overrides: HashMap<(usize, usize), PatchOp>, // (layer, feature) → operation
}

impl PatchedVindex {
    /// Gate KNN checks overrides first, then falls through to base
    fn gate_knn(&self, layer: usize, residual: &[f32], top_k: usize) -> Vec<(usize, f32)>;
    
    /// Feature lookup checks overrides, then base
    fn feature_meta(&self, layer: usize, feature: usize) -> Option<&FeatureMeta>;
    
    /// Flatten all patches into the base, producing a new clean VectorIndex
    fn bake_down(&self) -> VectorIndex;
}
```

The base vindex files on disk are never modified. Patches are an overlay. This means:
- Multiple users can apply different patches to the same base vindex
- Patches can be reverted cleanly (just remove from the stack)
- The base vindex remains cacheable and immutable

#### 5.7.4 Patch Composition and Conflicts

Patches apply in order. Later patches override earlier ones for the same feature:

```sql
-- Patch A inserts feature F8821 at L26 with target "Colchester"
-- Patch B updates feature F8821 at L26 with target "London"
-- Result: F8821 at L26 → "London" (Patch B wins)
```

Explicit conflict strategies for COMPILE INTO VINDEX:

```sql
COMPILE CURRENT INTO VINDEX "output.vindex"
    ON CONFLICT LAST_WINS;          -- Default: later patch overrides
    
COMPILE CURRENT INTO VINDEX "output.vindex"
    ON CONFLICT HIGHEST_CONFIDENCE; -- Keep the operation with higher confidence
    
COMPILE CURRENT INTO VINDEX "output.vindex"
    ON CONFLICT FAIL;               -- Error if any conflicts exist
```

#### 5.7.5 Use Cases

**Bug fixes.** Someone discovers the model hallucinates that "Sydney is the capital of Australia." They create a patch that deletes the hallucinated feature and inserts "Canberra." Share the .vlp file — 20 KB. Anyone applies it.

**Domain adaptation.** A medical team curates 5,000 clinical facts as a patch. A legal team curates 3,000 legal facts. A finance team curates 2,000 market facts. Each patch is 50 MB. Stack them on the same base model. No fine-tuning, no GPU, no retraining.

**A/B testing knowledge.** Apply Patch A (conservative medical knowledge) or Patch B (aggressive medical knowledge) to the same base model. Compare outputs. Revert instantly.

**Collaborative editing.** Multiple teams contribute patches to a shared base model. Patches are version-controlled in Git (they're just JSON files). CI validates that patches don't conflict. Merge produces a clean vindex.

**Model versioning.** Each release is a base vindex + a set of patches. Rollback = remove the latest patch. Audit trail = read the patch descriptions.

#### 5.7.6 Comparison with LoRA

| Dimension | LoRA Adapter | Vindex Patch |
|-----------|-------------|-------------|
| **Size** | ~50-200 MB | ~10 KB per fact |
| **Creation** | Training with gradient descent (hours, GPU) | INSERT statement (seconds, CPU) |
| **Granularity** | Low-rank approximation across many weights | Exact: specific features, specific facts |
| **Human-readable** | No (opaque weight matrices) | Yes (JSON with entity, relation, target) |
| **Composable** | Limited (merging LoRAs is lossy) | Yes (patches stack, conflicts resolved explicitly) |
| **Reversible** | Partially (remove adapter) | Fully (remove from stack, base unchanged) |
| **Round-trip** | No (approximate) | Yes (COMPILE → exact safetensors) |
| **Training required** | Yes (GPU, dataset, hyperparameters) | No |

LoRA is the right tool when you need to adapt model behaviour broadly (tone, style, domain expertise across many interactions). Vindex patches are the right tool when you need to add, remove, or correct specific facts. They're complementary — a production deployment might use a LoRA for style adaptation and a vindex patch stack for knowledge corrections.

### 5.8 Feature Lookup

```rust
let meta: Option<&FeatureMeta> = index.feature_meta(layer, feature);
let n: usize = index.num_features(layer);
let layers: Vec<usize> = index.loaded_layers();
let label: Option<&str> = labels.get(&(layer, feature));
```

---

## 6. Rust API

### 6.1 Core Types

```rust
// The index — owns gate vectors + down metadata in memory
pub struct VectorIndex { ... }

// Per-feature output metadata
pub struct FeatureMeta {
    pub top_token: String,
    pub top_token_id: u32,
    pub c_score: f32,
    pub top_k: Vec<TopKEntry>,
}

// Walk result
pub struct WalkTrace {
    pub layers: Vec<(usize, Vec<WalkHit>)>,
}

// Describe result
pub struct DescribeEdge {
    pub relation: Option<String>,
    pub source: LabelSource,
    pub target: String,
    pub gate_score: f32,
    pub layer_min: usize,
    pub layer_max: usize,
    pub count: usize,
    pub also_tokens: Vec<String>,
}

pub enum LabelSource {
    Probe,      // Model inference confirmed
    Cluster,    // Cluster-based matching
    Pattern,    // Entity pattern detection
    None,       // TF-IDF fallback (no tag shown)
}

// Config from index.json
pub struct VindexConfig {
    pub version: u32,
    pub model: String,
    pub family: String,
    pub source: Option<VindexSource>,
    pub checksums: Option<HashMap<String, String>>,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub embed_scale: f32,
    pub extract_level: ExtractLevel,
    pub has_model_weights: bool,
    pub layer_bands: LayerBands,
    pub layers: Vec<VindexLayerInfo>,
    pub down_top_k: usize,
    pub model_config: Option<VindexModelConfig>,
}

/// Provenance: which model checkpoint this vindex was built from.
pub struct VindexSource {
    pub huggingface_repo: Option<String>,
    pub huggingface_revision: Option<String>,
    pub safetensors_sha256: Option<String>,
    pub extracted_at: String,          // ISO 8601 timestamp
    pub larql_version: String,
}

/// Model-specific layer band boundaries.
/// Computed during EXTRACT, stored in index.json, used by DESCRIBE and label matching.
pub struct LayerBands {
    pub syntax: (usize, usize),      // (start, end) inclusive
    pub knowledge: (usize, usize),
    pub output: (usize, usize),
}

pub enum ExtractLevel {
    Browse,
    Inference,
    All,
}

// Patch types
pub struct VindexPatch {
    pub version: u32,
    pub base_model: String,
    pub base_checksum: Option<String>,
    pub created_at: String,
    pub description: Option<String>,
    pub author: Option<String>,
    pub tags: Vec<String>,
    pub operations: Vec<PatchOp>,
}

pub enum PatchOp {
    Insert {
        layer: usize,
        feature: usize,
        relation: Option<String>,
        entity: String,
        target: String,
        confidence: Option<f32>,
        gate_vector: Vec<f32>,
        down_meta: FeatureMeta,
    },
    Update {
        layer: usize,
        feature: usize,
        gate_vector: Option<Vec<f32>>,
        down_meta: Option<FeatureMeta>,
    },
    Delete {
        layer: usize,
        feature: usize,
        reason: Option<String>,
    },
}

// Patched vindex — overlay on immutable base
pub struct PatchedVindex {
    pub base: VectorIndex,
    pub patches: Vec<VindexPatch>,
    pub overrides: HashMap<(usize, usize), PatchOp>,
}

// Error type
pub enum VindexError {
    NotADirectory(PathBuf),
    NoSafetensors(PathBuf),
    MissingTensor(String),
    Parse(String),
    UnsupportedDtype(String),
    InsufficientExtractLevel { needed: ExtractLevel, have: ExtractLevel },
    Io(std::io::Error),
}
```

### 6.2 Load Functions

```rust
pub fn load_vindex_config(dir: &Path) -> Result<VindexConfig, VindexError>;
pub fn load_vindex_embeddings(dir: &Path) -> Result<(Array2<f32>, f32), VindexError>;
pub fn load_vindex_tokenizer(dir: &Path) -> Result<Tokenizer, VindexError>;
pub fn load_feature_labels(path: &Path) -> Result<HashMap<(usize, usize), String>, VindexError>;
```

### 6.3 Callbacks

```rust
pub trait IndexLoadCallbacks {
    fn on_file_start(&mut self, component: &str, path: &str) {}
    fn on_progress(&mut self, records: usize) {}
    fn on_file_done(&mut self, component: &str, records: usize, elapsed_ms: f64) {}
}

pub struct SilentLoadCallbacks;
```

---

## 7. Crate Structure

```
larql-vindex/
├── Cargo.toml
└── src/
    ├── lib.rs          Module exports + crate docs
    ├── config.rs       VindexConfig, VindexLayerInfo, ExtractLevel
    ├── error.rs        VindexError (including InsufficientExtractLevel)
    ├── index.rs        VectorIndex, FeatureMeta, KNN, WalkTrace
    ├── load.rs         Load gate vectors, down_meta, embeddings, config
    ├── describe.rs     DescribeEdge, describe_entity, label resolution
    ├── mutate.rs       Set/delete features, save to disk
    └── compile.rs      Vindex → safetensors reconstruction
```

**Dependencies:** `larql-models` (TopKEntry, weight loading), `ndarray` (BLAS), `serde`/`serde_json`, `tokenizers`, `thiserror`

**Build pipeline** (EXTRACT) lives in `larql-inference/vindex/` because it needs `ModelWeights` to extract vectors from safetensors.

---

## 8. Size Reference (Gemma 3 4B)

### 8.1 Current Layout

| File | Size | Description |
|------|------|-------------|
| gate_vectors.bin | 3.32 GB | 34 layers × 10,240 features × 2,560 dim × 4 bytes |
| embeddings.bin | 2.50 GB | 262,144 vocab × 2,560 dim × 4 bytes (f32) |
| down_meta.jsonl | ~160 MB | 348,160 features × ~460 bytes avg per record |
| model_weights.bin | ~12 GB | Full model weights (includes duplicated gate/down) |
| index.json | ~8 KB | Config + layer offsets |
| tokenizer.json | ~32 MB | HuggingFace tokenizer |
| relation_clusters.json | ~2 MB | 512 clusters with labels + top tokens |
| feature_clusters.jsonl | ~5 MB | 348,160 feature → cluster assignments |
| feature_labels.json | ~10 KB | 157 probe-confirmed labels |
| **Browse-only total** | **~6 GB** | gate + embed + down_meta + labels |
| **With weights total** | **~18 GB** | Everything (with duplication) |

### 8.2 Planned Deduplicated Layout (f16)

| File | Size (f16) | Description |
|------|------------|-------------|
| gate_vectors.bin | 1.66 GB | Same data, half precision |
| embeddings.bin | 1.25 GB | Same data, half precision |
| down_meta.bin | ~2 MB | Binary token IDs, not JSONL |
| attn_weights.bin | ~3 GB | Q, K, V, O only (not duplicated) |
| up_weights.bin | ~1.7 GB | W_up only |
| down_weights.bin | ~1.7 GB | Full W_down vectors (for COMPILE) |
| norms.bin | ~1 MB | LayerNorm parameters |
| lm_head.bin | ~1.3 GB | Output projection (if not tied to embeddings) |
| **Browse-only total** | **~3 GB** | 62% smaller than original model |
| **Browse + infer total** | **~6 GB** | 25% smaller than original model |
| **Full total** | **~10 GB** | 25% larger but queryable + compilable |

---

## 9. Build Pipeline

### 9.1 Extract from Model

```bash
# Browse-only (default)
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex

# With inference weights
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --include-weights

# With all weights (future: --extract-level all)
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex --include-weights
```

**Build steps:**
1. Load model from safetensors
2. Extract gate vectors → `gate_vectors.bin` (0.5s)
3. Extract embeddings → `embeddings.bin` (0.4s)
4. Compute down metadata → `down_meta.jsonl` (~11 min)
5. Compute relation clusters → `relation_clusters.json` (~19s at k=512)
6. Copy tokenizer → `tokenizer.json`
7. (Optional) Copy model weights → `model_weights.bin` (~2s)

**Total build time:** ~12 minutes on M-series Mac.

### 9.2 Add Labels

```bash
# Merge probe labels from larql-knowledge
larql label gemma3-4b.vindex --probes feature_labels.json

# Recluster with updated triples
larql label gemma3-4b.vindex --triples wikidata_triples.json --wordnet wordnet_relations.json
```

Labels are additive — new probes add new feature labels without removing existing ones.

### 9.3 Resume Support

The build pipeline supports resuming interrupted builds:

```bash
larql extract-index ... --resume
```

Checks which files already exist and skips completed steps. Enables incremental rebuilds when only labels or clusters need updating.

---

## 10. Planned Format Changes

### 10.1 Binary down_meta (Priority: HIGH)

Replace JSONL with compact binary format.

**Current:** 160 MB as human-readable JSON  
**Target:** ~2 MB as packed binary

**Format:**
```
Per feature (fixed size):
  top_token_id:  u32 (4 bytes)
  c_score:       f32 (4 bytes)
  num_top_k:     u8  (1 byte)
  top_k entries:
    token_id:    u32 (4 bytes)
    score:       f32 (4 bytes)
```

Token strings resolved at read time via tokenizer. No string storage needed.

### 10.2 Split Weight Files (Priority: HIGH)

Replace single `model_weights.bin` with component files.

**Current:** One 12 GB file containing everything (with duplication)  
**Target:** Separate files per component, no duplication:
- `attn_weights.bin` — Q, K, V, O only (~3 GB f16)
- `up_weights.bin` — W_up per layer (~1.7 GB f16)
- `down_weights.bin` — Full W_down per layer (~1.7 GB f16)
- `norms.bin` — LayerNorm parameters (~1 MB)
- `lm_head.bin` — Output projection (~1.3 GB f16)

Each file loads independently. INFER loads only attn_weights.bin. COMPILE loads all.

### 10.3 f16 Storage (Priority: MEDIUM)

Store gate vectors and embeddings at half precision.

**Current:** f32 (4 bytes per float)  
**Target:** f16 (2 bytes per float), cast to f32 at load time

Gate KNN accuracy at f16 is effectively identical to f32 — the top-K results don't change.

### 10.4 Memory-Mapped Loading (Priority: MEDIUM)

Use `mmap` for gate_vectors.bin instead of reading into heap.

**Benefit:** Instant load time. OS pages in data on demand. Multiple vindexes can share physical memory for overlapping pages.

### 10.5 Incremental down_meta (Priority: LOW)

Append-only format for mutation without full rewrite.

**Current:** INSERT requires rewriting entire down_meta.jsonl  
**Target:** Append new/modified records to end of file, mark originals as superseded via a tombstone index.

### 10.6 Streaming Build (Priority: LOW)

Build vindex without loading entire model into memory. Process one layer at a time, write immediately to disk.

**Benefit:** Build vindexes for 70B+ models on machines with limited RAM.

---

## 11. Distributed Hosting

The vindex format is designed for distributed access. Each file is independently loadable, serves a specific function, and has a known size. This maps directly to distributed architectures where different components live on different servers.

### 11.1 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client                                   │
│  Phone / Laptop / Edge device                                    │
│  Runs: larql binary (no ML framework, no GPU)                   │
│                                                                  │
│  USE REMOTE "https://models.example.com/gemma3-4b.vindex";      │
│  DESCRIBE "France";                                              │
│  WALK "Einstein" TOP 10;                                         │
└─────────────┬───────────────────────────────────────────────────┘
              │ HTTPS / gRPC
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Knowledge Server                              │
│  Hosts: gate_vectors.bin + embeddings.bin + down_meta.bin        │
│  Serves: DESCRIBE, WALK, SELECT, EXPLAIN WALK                   │
│  Size: ~3 GB (f16 browse-only)                                  │
│  Hardware: Any CPU, no GPU needed. Rust binary.                 │
│  Latency: <50ms per query (gate KNN + label lookup)             │
└─────────────┬───────────────────────────────────────────────────┘
              │ (optional, only for INFER)
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Server                               │
│  Hosts: attn_weights.bin (+ gate_vectors from knowledge server) │
│  Serves: INFER, EXPLAIN INFER                                   │
│  Size: ~3 GB additional (f16 attention weights)                 │
│  Hardware: CPU (or GPU for speed)                               │
│  Latency: ~200ms per inference (attention + walk FFN)           │
└─────────────┬───────────────────────────────────────────────────┘
              │ (optional, rare)
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Compile Server                                 │
│  Hosts: up_weights.bin + down_weights.bin + norms.bin           │
│  Serves: COMPILE (batch jobs, not real-time)                    │
│  Size: ~4 GB additional                                         │
│  Hardware: Any CPU                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 Access Patterns

| Operation | Files Needed | Server | Latency | Bandwidth |
|-----------|-------------|--------|---------|-----------|
| DESCRIBE | gate + embed + down_meta + labels | Knowledge | <50ms | ~1 KB response |
| WALK | gate + embed | Knowledge | <50ms | ~1 KB response |
| SELECT | gate + down_meta + labels | Knowledge | <10ms | ~1 KB response |
| SHOW RELATIONS | labels only | Knowledge | <1ms | ~10 KB response |
| STATS | index.json only | Any | <1ms | ~8 KB response |
| INFER | gate + embed + attn_weights | Inference | ~200ms | ~1 KB response |
| COMPILE | All files | Compile | minutes | ~10 GB output |

### 11.3 Remote Protocol

LQL statements become network calls:

```sql
-- Connect to a remote vindex
USE REMOTE "https://models.example.com/gemma3-4b.vindex";

-- Knowledge queries go to the knowledge server
DESCRIBE "France";
-- → GET /api/v1/describe?entity=France
-- ← JSON: [{relation: "capital", target: "Paris", gate_score: 1436.9, ...}]

WALK "Einstein" TOP 10;
-- → GET /api/v1/walk?prompt=Einstein&top=10
-- ← JSON: [{layer: 25, feature: 1995, gate_score: 11.2, target: "art", ...}]

SELECT entity, target FROM EDGES WHERE relation = "capital" LIMIT 5;
-- → POST /api/v1/select
-- ← JSON: [{entity: "France", target: "Paris"}, ...]

-- Inference goes to the inference server
INFER "The capital of France is" TOP 5;
-- → POST /api/v1/infer {prompt: "The capital of France is", top: 5}
-- ← JSON: [{token: "Paris", probability: 0.9791}, ...]
```

The REPL detects whether the backend is local or remote and routes accordingly. Same LQL statements, same output format. The user doesn't know or care where the vindex lives.

### 11.4 Layer-Level Access

For constrained clients (mobile, embedded), individual layers can be fetched on demand instead of downloading the entire gate_vectors.bin:

```
GET /api/v1/layers/27/gate_vectors
← Binary: 10,240 × 2,560 × 2 bytes = ~50 MB (f16)
```

The `index.json` layer offsets enable byte-range requests against a static file server:

```
GET /files/gate_vectors.bin
Range: bytes=5242880000-5347737600
← Layer 27 data (100 MB at f32)
```

This means a client can query a single knowledge layer (~50 MB download) instead of the full index (~3 GB). Useful for:
- Mobile apps querying specific facts
- Edge devices with limited storage
- Progressive loading — start with one layer, fetch more as needed

### 11.5 Decoupled Models

The distributed architecture enables mixing components from different models:

```
┌──────────────────────────────┐
│  Edge Device                  │
│  Small model attention (4B)  │──── local attention
│                              │
│  Remote knowledge (70B)      │──── fetched from server
└──────────────────────────────┘
```

```sql
-- Use a small model's attention with a large model's knowledge
USE ATTENTION MODEL "google/gemma-3-4b-it";
USE KNOWLEDGE REMOTE "https://models.example.com/llama-70b.vindex";

INFER "The capital of France is" TOP 5;
-- Small model runs attention locally
-- Knowledge query hits the 70B vindex remotely
-- Paris comes out (70B's knowledge, 4B's speed)
```

This works because:
- The cross-model entity×relation coordinates align at 0.946 cosine (proven via Procrustes)
- Attention routing is template-based, not model-specific (99% of heads are fixed)
- The FFN graph walk is independent of the attention mechanism
- The gate KNN accepts any residual vector — it doesn't check which model produced it

### 11.6 Caching

Remote vindexes benefit from aggressive caching:

- **index.json** — cache indefinitely (immutable after build, versioned)
- **feature_labels.json** — cache with TTL (updates as probes run)
- **gate_vectors.bin** — cache indefinitely (immutable)
- **embeddings.bin** — cache indefinitely (immutable)
- **down_meta** — cache indefinitely (immutable unless INSERT)

The binary files are immutable after EXTRACT — they never change unless you INSERT/DELETE and rebuild. Standard HTTP caching (ETag, Cache-Control: immutable) works perfectly. The checksums in index.json serve as ETags.

For INSERT/DELETE, the vindex creates a new version. Old cached files remain valid for the old version. Clients check the version number and invalidate on change.

### 11.7 Publishing to HuggingFace

Vindexes can be published as HuggingFace datasets:

```bash
# Build locally
larql extract-index google/gemma-3-4b-it -o gemma3-4b.vindex

# Run probes (optional, adds labels)
python3 scripts/probe_mlx.py --output gemma3-4b.vindex/feature_labels.json

# Push to HuggingFace
huggingface-cli upload chrishayuk/gemma3-4b-vindex gemma3-4b.vindex/

# Anyone can now use it
larql> USE REMOTE "https://huggingface.co/datasets/chrishayuk/gemma3-4b-vindex";
larql> DESCRIBE "France";
```

The vindex directory maps directly to a HuggingFace dataset repo. Each file is a separate blob. HuggingFace's CDN handles distribution, caching, and access control.

**Pre-labelled vindexes** with probe-confirmed labels are the premium artifacts. Build once, publish, everyone benefits from the labels without running their own probes.

### 11.8 Scaling Economics

The cost model for distributed vindexes is fundamentally different from GPU inference:

```
Traditional inference hosting:
  GPU cluster:     $2-10/hour per A100
  Serving 1 model: ~$50K/year
  Scaling:         More GPUs, linear cost

Vindex knowledge hosting:
  VPS with 8GB RAM: $20/month
  Serving 1 model:  $240/year
  Scaling:          CDN for static files, near-zero marginal cost

  The "inference" is a dot product. No GPU. No framework.
  The files are static. CDN-cacheable.
  10,000 users querying the same vindex = same cost as 1 user.
```

This is why the decoupled model architecture matters. Knowledge hosting scales like a CDN. Inference (when needed) scales like compute. Most queries are knowledge queries. The expensive operation (attention) is only needed for generation, not for understanding what the model knows.

---

## 12. Large Models on Small Hardware

The vindex format fundamentally changes what it means to "run" a large model. Traditional inference requires loading all weights and executing a forward pass. Vindex separates knowledge access from generation, enabling large model capabilities on consumer hardware.

### 12.1 The Hardware Problem Today

| Model | Size (f16) | Minimum Hardware for Inference |
|-------|-----------|-------------------------------|
| Llama 3 8B | 16 GB | 1× GPU (24GB VRAM) or 32GB RAM (CPU, slow) |
| Llama 3 70B | 140 GB | 2-4× A100 GPUs ($8-20/hour cloud) |
| Llama 3 405B | 800 GB | 8× A100 or 4× H100 ($40-100/hour cloud) |

Most people cannot run a 70B model. Almost nobody can run a 405B model. The knowledge inside these models is locked behind a hardware wall.

### 12.2 What Vindex Changes

The insight: most of what people use LLMs for is knowledge retrieval — "What is X? Who did Y? How does Z work?" These questions don't need a forward pass. They need a database lookup. The vindex makes the model's knowledge accessible without running the model.

| Model | Full Inference | Vindex Browse | Hardware for Browse |
|-------|---------------|---------------|---------------------|
| Llama 3 8B | 16 GB, GPU | ~5 GB | Any laptop |
| Llama 3 70B | 140 GB, multi-GPU | ~25 GB | Workstation (32GB RAM) |
| Llama 3 405B | 800 GB, GPU cluster | ~120 GB | Server (128GB RAM) or streamed |

A 70B model's entire knowledge graph — every fact, every relation, every entity — queryable on a single machine with 32GB of RAM. No GPU. No ML framework. Just a Rust binary and static files.

### 12.3 Five Strategies for Small Hardware

**Strategy 1: Browse-only.** Load the gate vectors and embeddings. DESCRIBE, WALK, SELECT work fully. No generation, but full knowledge access.

```sql
-- On a laptop with 8GB RAM, browse a 4B model's knowledge
larql> USE "gemma3-4b.vindex";  -- loads 3 GB
larql> DESCRIBE "France";       -- 33ms, full knowledge graph
larql> DESCRIBE "Einstein";     -- 33ms, occupation, birthplace, field
```

**Strategy 2: Layer-on-demand.** Don't load the full vindex. Fetch individual layers as needed. Each layer is ~50-100MB (f16). A phone can query one layer at a time.

```sql
-- On a phone with 4GB RAM
larql> USE "gemma3-4b.vindex" LAYERS ON DEMAND;
larql> DESCRIBE "France" AT LAYER 27;  -- fetches ~100 MB, scans 1 layer
-- capital → Paris (probe)
```

**Strategy 3: Decoupled inference.** Run a small model's attention locally. Query a large model's knowledge remotely. Get large-model answers at small-model cost.

```sql
-- On a MacBook: 4B attention runs locally, 70B knowledge is remote
larql> USE ATTENTION MODEL "google/gemma-3-4b-it";
larql> USE KNOWLEDGE REMOTE "https://models.example.com/llama-70b.vindex";
larql> INFER "The mechanism of action of metformin is" TOP 5;
-- 4B attention routes the query
-- 70B vindex provides the medical knowledge
-- Result: 70B-quality answer, 4B-level hardware
```

**Strategy 4: Knowledge patching.** Start with a small model you can run locally. Apply patches from a large model to inject specific domain knowledge.

```sql
-- Start with a 4B model (runs on any laptop)
larql> USE "gemma3-4b.vindex";
larql> DESCRIBE "metformin";
-- occupation → drug (cluster) — sparse knowledge

-- Apply medical knowledge patch extracted from a 70B model
larql> APPLY PATCH "llama70b-medical.vlp";  -- 50 MB
larql> DESCRIBE "metformin";
-- mechanism → biguanide (probe)
-- treats → diabetes (probe)
-- side_effect → lactic acidosis (probe)
-- 70B-level medical knowledge in a 4B-level model
```

**Strategy 5: Quantised browse.** Store gate vectors at 4-bit or 8-bit precision. The KNN accuracy is nearly identical — the top-K results don't change because the ranking is preserved even at low precision.

```
Gate vectors at f32:  3.32 GB (current)
Gate vectors at f16:  1.66 GB
Gate vectors at int8: 0.83 GB
Gate vectors at int4: 0.42 GB — a 4B model's knowledge in 400 MB
```

At int4 quantisation, a 70B model's knowledge index fits in ~6GB. Browsable on a phone.

### 12.4 What Requires Full Hardware

Not everything can run on small hardware. Generation — producing fluent multi-token responses — requires attention, and attention requires the model weights proportional to model size:

| Operation | Small Hardware | Full Hardware |
|-----------|--------------|---------------|
| DESCRIBE (knowledge lookup) | ✅ | ✅ |
| WALK (feature scan) | ✅ | ✅ |
| SELECT (knowledge query) | ✅ | ✅ |
| SHOW RELATIONS | ✅ | ✅ |
| INFER (text generation) | ⚠️ Decoupled only | ✅ |
| COMPILE (model editing) | ❌ Needs all weights | ✅ |

The split is clean: knowledge access works on anything, generation needs compute. Most use cases are knowledge access. The vindex makes 80% of what people use LLMs for available on 20% of the hardware.

### 12.5 Implications

**Democratisation.** A student with a Chromebook can query what GPT-4-class models know about their research topic. They can't generate essays, but they can explore the knowledge graph, discover relations, find facts.

**Offline access.** Download a 3GB vindex to your phone. Query the model's knowledge on an airplane, in a field, in a classroom with no internet. No API calls, no cloud dependency, no usage fees.

**Edge deployment.** IoT devices, embedded systems, robots — anything with 1GB of RAM can carry a quantised knowledge index. The device knows facts without needing to run inference.

**Cost reduction.** A company paying $50K/year for GPU inference to answer knowledge questions switches to a $240/year VPS serving vindex queries. The 99% of queries that are knowledge lookups become nearly free. The 1% that need generation route to the GPU cluster.

**Privacy.** The vindex runs locally. No data leaves the device. No API calls to log. No prompts stored on someone else's server. The model's knowledge is a local file, queryable offline, with no network dependency.