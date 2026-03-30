# LQL -- Lazarus Query Language Specification

**Version:** 0.1
**Author:** Chris Hay
**Date:** 2026-03-30
**Status:** Draft
**Implementation target:** `larql-cli` (Rust)

---

## 1. Design Principles

LQL is a query language for neural network weights treated as a graph database. It is not SQL. It is not SPARQL. It borrows from both but serves a different purpose: decompiling, inspecting, editing, and recompiling neural networks.

**Principles:**

1. **Weights are rows.** Every W_gate row is a record. Every W_down column is a record. Every embedding vector is a record. The model IS the database.
2. **Two backends, one language.** LQL operates on either a `.vindex` (pre-extracted, fast) or directly on model weights via safetensors (live, no extraction needed). The vindex is preferred for production -- sub-millisecond lookups from a pre-built index. Direct weight access is for exploration -- point at any model, start querying immediately. Same statements, same results, different performance.
3. **Statements, not scripts.** Each LQL statement is self-contained. No variables that persist across statements (except `USE` context). Pipe results with `|>`.
4. **Three verbs for the demo.** The video needs exactly: `EXTRACT`, `WALK`, `INSERT`, `COMPILE`. Everything else is power-user.
5. **Rust-native.** The parser lives in `larql-lql`. No Python dependency. No runtime. One binary.

---

## 2. Statement Categories

### 2.1 Model Lifecycle

| Statement | Purpose |
|---|---|
| `EXTRACT` | Decompile model weights -> vindex |
| `COMPILE` | Recompile vindex -> model weights |
| `DIFF` | Compare two vindexes |
| `USE` | Set active vindex / model context |

### 2.2 Knowledge Browser (pure vindex)

| Statement | Purpose |
|---|---|
| `WALK` | Feature scan -- what gate features fire for a token's embedding |
| `SELECT` | Query edges by entity, relation, layer |
| `DESCRIBE` | Show all knowledge for an entity, grouped by layer band |
| `EXPLAIN` | Show the feature trace (which features fired per layer) |

### 2.3 Inference (requires model weights)

| Statement | Purpose |
|---|---|
| `INFER` | Full forward pass with attention -- actual next-token prediction |

### 2.4 Knowledge Mutation

| Statement | Purpose |
|---|---|
| `INSERT` | Add edge(s) to the vindex |
| `DELETE` | Remove edge(s) from the vindex |
| `UPDATE` | Modify existing edge(s) |
| `MERGE` | Merge edges from another vindex |

### 2.5 Schema Introspection

| Statement | Purpose |
|---|---|
| `SHOW RELATIONS` | List discovered relation types |
| `SHOW LAYERS` | Layer-by-layer summary |
| `SHOW FEATURES` | Feature details at a layer |
| `SHOW MODELS` | List available vindexes |
| `STATS` | Counts, coverage, size |

---

## 3. Grammar

### 3.1 Notation

```
UPPERCASE  = keyword (case-insensitive in parser)
<name>     = required parameter
[name]     = optional parameter
{a | b}    = choice
...        = repeatable
```

### 3.2 Model Lifecycle Statements

```
EXTRACT MODEL <model_id> INTO <vindex_path>
    [COMPONENTS <component_list>]
    [LAYERS <range>]

-- Decompile a HuggingFace model (or local path) into a vindex.
-- Components: ffn_gate, ffn_down, ffn_up, embeddings, attn_ov, attn_qk
-- Default: ffn_gate, ffn_down, ffn_up, embeddings (sufficient for recompile)
-- Layers: 0-33 (default: all)

EXTRACT MODEL "google/gemma-3-4b-it"
    INTO "gemma3-4b.vindex";

EXTRACT MODEL "google/gemma-3-4b-it"
    INTO "gemma3-4b.vindex"
    COMPONENTS ffn_gate, ffn_down, ffn_up, embeddings
    LAYERS 0-33;
```

```
COMPILE <vindex_path> INTO MODEL <output_path>
    [FORMAT {safetensors | gguf}]

-- Recompile a vindex back to model weights.
-- Round-trip: EXTRACT then COMPILE should produce identical weights.

COMPILE "gemma3-4b.vindex"
    INTO MODEL "gemma3-4b-edited/"
    FORMAT safetensors;
```

```
DIFF <vindex_a> <vindex_b>
    [LAYER <n>]
    [RELATION <type>]
    [LIMIT <n>]

-- Show edges that differ between two vindexes.

DIFF "gemma3-4b.vindex" "gemma3-4b-edited.vindex";

DIFF "gemma3-4b.vindex" "gemma3-4b-edited.vindex"
    RELATION "lives-in"
    LIMIT 20;
```

```
USE <vindex_path>;
USE MODEL <model_id>;

-- Set the active backend for subsequent statements.
-- USE with a .vindex path: fast, pre-extracted, all statements available.
-- USE MODEL with a HuggingFace ID or local path: live weight access,
--   reads safetensors directly. Slower (dense matmul per query) but
--   zero setup.

USE "gemma3-4b.vindex";

USE MODEL "google/gemma-3-4b-it";
```

### 3.3 Knowledge Browser Statements

```
WALK <prompt>
    [TOP <n>]
    [LAYERS {<range> | ALL}]

-- Feature scan: what gate features fire for the last token's embedding.
-- This is a knowledge browser operation, NOT inference.
-- No attention is used. The query is the raw token embedding.

WALK "France" TOP 10;

WALK "The capital of France is"
    TOP 10
    LAYERS 24-33;
```

```
DESCRIBE <entity>
    [AT LAYER <n>]
    [RELATIONS ONLY]

-- Show all knowledge for an entity. Groups by layer band:
--   Knowledge (L14-27): semantic/factual features
--   Output (L28-33): formatting/output features
--   Morphological (L0-13): syntactic patterns
-- Filters encoding noise (non-Latin tokens) automatically.

DESCRIBE "France";

DESCRIBE "Mozart"
    AT LAYER 26;
```

```
SELECT [<fields>]
    FROM EDGES
    [NEAREST TO <entity> AT LAYER <n>]
    [WHERE <conditions>]
    [ORDER BY <field> {ASC | DESC}]
    [LIMIT <n>]

-- Query edges in the vindex.

SELECT entity, relation, target, confidence
    FROM EDGES
    WHERE entity = "France"
    ORDER BY confidence DESC
    LIMIT 10;

SELECT *
    FROM EDGES
    WHERE layer = 26
    AND feature = 9515;

SELECT entity, target, distance
    FROM EDGES
    NEAREST TO "Mozart"
    AT LAYER 26
    LIMIT 20;
```

```
EXPLAIN WALK <prompt>
    [LAYERS <range>]
    [VERBOSE]

-- Show the per-layer feature trace.

EXPLAIN WALK "The capital of France is";
```

### 3.4 Inference Statement

```
INFER <prompt>
    [TOP <n>]
    [COMPARE]

-- Full forward pass with attention. Requires model weights.
-- Uses walk FFN (gate KNN from vindex) as the FFN backend,
-- but runs real attention for token routing.
-- COMPARE: also run dense inference and show both.

INFER "The capital of France is" TOP 5;

INFER "The capital of France is" TOP 5 COMPARE;
```

### 3.5 Knowledge Mutation Statements

```
INSERT INTO EDGES
    (entity, relation, target)
    VALUES (<entity>, <relation>, <target>)
    [AT LAYER <n>]
    [CONFIDENCE <float>]

INSERT INTO EDGES
    (entity, relation, target)
    VALUES ("John Coyle", "lives-in", "Colchester");
```

```
DELETE FROM EDGES
    WHERE <conditions>

DELETE FROM EDGES
    WHERE entity = "John Coyle"
    AND relation = "lives-in";
```

```
UPDATE EDGES
    SET <field> = <value>
    WHERE <conditions>

UPDATE EDGES
    SET target = "London"
    WHERE entity = "John Coyle"
    AND relation = "lives-in";
```

```
MERGE <source_vindex>
    [INTO <target_vindex>]
    [ON CONFLICT {KEEP_SOURCE | KEEP_TARGET | HIGHEST_CONFIDENCE}]

MERGE "medical-knowledge.vindex"
    INTO "gemma3-4b.vindex"
    ON CONFLICT HIGHEST_CONFIDENCE;
```

### 3.6 Schema Introspection Statements

```
SHOW RELATIONS
    [AT LAYER <n>]
    [WITH EXAMPLES]

SHOW LAYERS
    [RANGE <start>-<end>]

SHOW FEATURES <layer>
    [WHERE <conditions>]
    [LIMIT <n>]

SHOW MODELS;

STATS [<vindex_path>];
```

---

## 4. Backend Architecture

LQL abstracts over two backends through a common trait. Every query statement works against either backend.

### 4.1 The Two Backends

```
                        LQL Parser
                           |
              +------------+------------+
              v                         v
   +------------------+     +------------------+
   |  VindexBackend   |     |  WeightBackend   |
   |                  |     |                  |
   |  Pre-extracted   |     |  Live safetensors|
   |  KNN index       |     |  Dense matmul    |
   |  0.98ms/layer    |     |  ~6ms/layer      |
   |  Read + write    |     |  Read only       |
   |  No model needed |     |  Model in memory |
   +------------------+     +------------------+
```

### 4.2 Backend Capabilities

| Statement | Vindex | Direct Weights |
|---|---|---|
| WALK (feature scan) | KNN (0.98ms/layer) | Dense matmul (~6ms/layer) |
| INFER (with attention) | Needs model weights in vindex | Full forward pass |
| SELECT | Index lookup | Live gate x embedding scan |
| DESCRIBE | Pre-computed edges | On-the-fly per entity |
| EXPLAIN | Walk trace from index | Walk trace from matmul |
| SHOW RELATIONS | From schema cache | Cluster on-the-fly (slow first time) |
| SHOW LAYERS | From metadata | Computed from weights |
| SHOW FEATURES | Index lookup | Dense scan per layer |
| STATS | Instant | Computed |
| INSERT | Supported | Error: "requires vindex" |
| DELETE | Supported | Error: "requires vindex" |
| UPDATE | Supported | Error: "requires vindex" |
| COMPILE | Supported | Error: "requires vindex" |
| DIFF | Supported | One side can be weights |
| MERGE | Supported | Error: "requires vindex" |

---

## 5. Pipe Operator

LQL supports `|>` to chain statements. The output of the left statement becomes context for the right.

```sql
-- Walk, then explain the walk
WALK "The capital of France is" TOP 5
    |> EXPLAIN WALK "The capital of France is";

-- Describe an entity, then diff against another model
DESCRIBE "France"
    |> DIFF WITH "llama3-8b.vindex";
```

---

## 6. The Demo Script

One terminal. One language. The full loop.

```sql
-- ACT 1: DECOMPILE
EXTRACT MODEL "google/gemma-3-4b-it"
    INTO "gemma3-4b.vindex";
USE "gemma3-4b.vindex";
STATS;

-- ACT 2: INSPECT
SHOW RELATIONS WITH EXAMPLES;
DESCRIBE "France";
SELECT entity, target, confidence
    FROM EDGES
    WHERE relation = "capital-of"
    ORDER BY confidence DESC
    LIMIT 10;

-- ACT 3: WALK + INFER
WALK "France" TOP 10;
EXPLAIN WALK "The capital of France is";
INFER "The capital of France is" TOP 5 COMPARE;

-- ACT 4: EDIT
WALK "John Coyle" TOP 5;
INSERT INTO EDGES
    (entity, relation, target)
    VALUES ("John Coyle", "lives-in", "Colchester");
DESCRIBE "John Coyle";
INFER "Where does John Coyle live?" TOP 5;

-- ACT 5: RECOMPILE
DIFF "gemma3-4b.vindex" CURRENT;
COMPILE CURRENT
    INTO MODEL "gemma3-4b-edited/"
    FORMAT safetensors;
```

---

## 7. Implementation Notes

### 7.1 Parser Architecture

```
Input string
    -> Lexer (tokenise keywords, strings, numbers, operators)
    -> Parser (recursive descent, one statement at a time)
    -> AST (Statement enum with variants per statement type)
    -> Executor (dispatches to larql-core / larql-inference / larql-models)
```

The parser lives in `larql-lql`.

### 7.2 AST

```rust
pub enum Statement {
    // Lifecycle
    Extract { model, output, components, layers },
    Compile { vindex, output, format },
    Diff { a, b, layer, relation, limit },
    Use { target },

    // Knowledge browser (pure vindex)
    Walk { prompt, top, layers, mode, compare },
    Select { fields, conditions, nearest, order, limit },
    Describe { entity, layer, relations_only },
    Explain { prompt, layers, verbose },

    // Inference (requires model weights)
    Infer { prompt, top, compare },

    // Mutation
    Insert { entity, relation, target, layer, confidence },
    Delete { conditions },
    Update { set, conditions },
    Merge { source, target, conflict },

    // Introspection
    ShowRelations { layer, with_examples },
    ShowLayers { range },
    ShowFeatures { layer, conditions, limit },
    ShowModels,
    Stats { vindex },

    // Pipe
    Pipe { left, right },
}
```

### 7.3 Crate Mapping

| Statement | Crate | Function |
|---|---|---|
| EXTRACT | `larql-models` | read safetensors -> write vindex |
| COMPILE | `larql-models` | read vindex -> write safetensors |
| WALK | `larql-inference` | gate KNN on VectorIndex |
| INFER | `larql-inference` | predict_with_ffn (attention + walk FFN) |
| SELECT | `larql-core` | edge query on graph |
| INSERT/DELETE/UPDATE | `larql-core` | graph mutation |
| DESCRIBE | `larql-inference` | multi-layer gate KNN + noise filtering |
| EXPLAIN | `larql-inference` | walk with trace capture |
| MERGE | `larql-core` | graph union |
| DIFF | `larql-core` | graph comparison |
| SHOW/STATS | `larql-core` + `larql-models` | metadata queries |
| USE | `larql-lql` | session state |

### 7.4 What Exists vs What's New

| Component | Status |
|---|---|
| Vector extraction (EXTRACT) | Done -- `vector-extract` + `extract-index` commands |
| Vindex build (EXTRACT) | Done -- `extract-index` command |
| Feature scan (WALK) | Done -- gate KNN on VectorIndex |
| Inference (INFER) | Done -- `predict_with_ffn` (needs model weights) |
| Edge query (SELECT) | Done -- down_meta scan |
| DESCRIBE | Done -- gate KNN + noise filtering + layer band grouping |
| EXPLAIN | Done -- per-layer feature trace |
| LQL Parser | Done -- recursive descent, 80+ keywords |
| REPL | Done -- rustyline, history, arrow keys |
| INSERT | Planned -- requires gate/down vector synthesis |
| DELETE | Planned -- requires vector zeroing |
| UPDATE | Planned -- requires vector replacement |
| COMPILE | Planned -- requires vindex -> safetensors writer |
| DIFF | Planned -- requires vindex comparison |
| MERGE | Planned -- requires graph union with conflict |
| W_up in vindex | Planned -- needed for COMPILE |

### 7.5 INSERT Semantics -- How Edge Becomes Vector

When you `INSERT ("John Coyle", "lives-in", "Colchester")`:

1. **Find the relation direction.** Look up the "lives-in" cluster centre from schema discovery.
2. **Find the entity embedding.** Look up "John" and "Coyle" token embeddings. Combine to get the entity vector.
3. **Find the target embedding.** Look up "Colchester" token embedding.
4. **Synthesise the gate vector.** Gate direction ~ entity embedding, scaled to match existing gate magnitudes.
5. **Synthesise the down vector.** Down direction ~ target embedding, scaled to match existing down magnitudes.
6. **Synthesise the up vector.** Up direction ~ gate direction (for simple facts).
7. **Find a free feature slot.** Use an unused feature (low activation across all entities).
8. **Write the vectors.** Update gate_vectors.bin, down metadata, and up vectors in the vindex.

### 7.6 Priority Order for Implementation

1. LQL Parser + REPL -- **done**
2. USE / STATS / SHOW -- **done**
3. SELECT / DESCRIBE -- **done**
4. WALK / EXPLAIN -- **done**
5. INFER -- **done** (requires model weights in vindex)
6. EXTRACT -- wiring to existing CLI commands
7. W_up extraction -- add to vindex (needed for COMPILE)
8. INSERT -- the hard one. Vector synthesis from edge semantics.
9. COMPILE -- vindex -> safetensors.
10. DIFF / DELETE / UPDATE / MERGE -- round out mutation support.

---

## 8. The REPL

```
$ larql repl

   ╦   ╔═╗ ╦═╗ ╔═╗ ╦
   ║   ╠═╣ ╠╦╝ ║═╬╗║
   ╩═╝ ╩ ╩ ╩╚═ ╚═╝╚╩═╝
   Lazarus Query Language v0.1

larql> USE "output/gemma3-4b.vindex";
Using: output/gemma3-4b.vindex (34 layers, 348.2K features, model: google/gemma-3-4b-it)

larql> DESCRIBE "France";
France
  Knowledge (L14-27):
    french               score=35.2  L15-32  (12x)
    europe               score=14.4  L10-25  (6x)
    german               score=15.0  L6-30   (4x)
    ...
  Output (L28-33):
    ...

larql> INFER "The capital of France is" TOP 3;
Predictions (walk FFN):
   1. Paris                (97.91%)
   2. the                  (0.42%)
   3. a                    (0.31%)
```

---

## 9. Future Extensions (Not for V1)

- **STEER** -- relation steering via +/- delta vectors at a layer.
- **CHAIN** -- multi-hop via L24 residual injection.
- **LIFT** -- analogy operation (same edge, different node).
- **TIMELINE / SUBGRAPH** -- graph algorithm queries (A*, PageRank, community detection).
- **CROSS MODEL** -- queries across multiple vindexes (Gemma features not in Llama).
- **INGEST** -- document -> context graph extraction.
- **SurrealDB backend** -- third backend for vector-level queries via HNSW indexes.
