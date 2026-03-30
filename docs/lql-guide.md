# LQL Quick Start Guide

**LQL** (Lazarus Query Language) is the query language for neural network weights treated as a graph database. One binary, no Python, no GPU.

LQL has two modes:

- **Knowledge browser** (pure vindex, no model needed): DESCRIBE, WALK, SELECT, SHOW, EXPLAIN, STATS. Instant. Runs anywhere.
- **Inference engine** (requires model weights): INFER. Full forward pass with attention. Produces actual predictions.

## Launch the REPL

```bash
cargo run -p larql-cli -- repl
```

Features: arrow keys, command history (persisted to `~/.larql_history`), Ctrl-R search, Ctrl-C cancel, Ctrl-D exit.

Or execute a single statement:

```bash
cargo run -p larql-cli -- lql 'SHOW MODELS;'
```

## Getting Started

### 1. Connect to a vindex

A `.vindex` is a decompiled model -- gate vectors, embeddings, and metadata packed into a directory. Point LQL at one:

```sql
USE "output/gemma3-4b.vindex";
```

If you don't know where your vindexes are:

```sql
SHOW MODELS;
```

This scans the current directory for `.vindex` folders.

### 2. Inspect

```sql
-- Summary statistics
STATS;

-- What layers are loaded?
SHOW LAYERS;

-- What tokens appear most in the features?
SHOW RELATIONS;

-- Feature details at a specific layer
SHOW FEATURES 26;
SHOW FEATURES 26 LIMIT 5;
```

### 3. Browse knowledge (pure vindex, no model)

```sql
-- What does the model know about France?
DESCRIBE "France";
-- Groups results by layer band:
--   Knowledge (L14-27): French, Europe, Paris...
--   Output (L28-33): formatting tokens
--   Morphological (L0-13): syntactic patterns
-- Filters encoding noise automatically.

-- What features fire for a specific token?
WALK "France" TOP 10;
-- Shows per-layer gate activations with down-projection outputs.
-- This is a feature scan, not inference.

-- Feature trace across all layers
EXPLAIN WALK "The capital of France is";

-- Search edges by layer, feature, or token
SELECT * FROM EDGES WHERE layer = 26 LIMIT 10;
SELECT * FROM EDGES WHERE layer = 26 AND feature = 9515;
```

### 4. Run inference (requires model weights)

```sql
-- Full prediction with attention
INFER "The capital of France is" TOP 5;
-- Requires model weights in the vindex (--include-weights) or a loaded model.

-- Compare walk FFN vs dense
INFER "The capital of France is" TOP 5 COMPARE;
```

If your vindex doesn't have model weights, INFER will tell you how to rebuild it:

```
INFER requires model weights. This vindex was built without --include-weights.
Options:
1. Rebuild: larql extract-index --model "google/gemma-3-4b-it" --output "..." --include-weights
2. Use the CLI: larql walk --index "..." --predict --model "google/gemma-3-4b-it"
```

### 5. Direct model access (coming soon)

```sql
-- Point at a HuggingFace model directly (no extraction needed)
USE MODEL "google/gemma-3-4b-it";

-- Same queries work, just slower (dense matmul instead of KNN)
DESCRIBE "France";
```

### 6. Edit and recompile (coming soon)

```sql
-- Add a fact the model doesn't know
INSERT INTO EDGES (entity, relation, target)
    VALUES ("John Coyle", "lives-in", "Colchester");

-- Check it's there
DESCRIBE "John Coyle";

-- What changed?
DIFF "gemma3-4b.vindex" CURRENT;

-- Recompile to weights
COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;
```

## Full Statement Reference

### Lifecycle

| Statement | Purpose | Status |
|---|---|---|
| `USE "path.vindex"` | Set active vindex | Working |
| `USE MODEL "id"` | Set active model (live weights) | Planned |
| `EXTRACT MODEL "id" INTO "path"` | Decompile model to vindex | Use CLI: `larql extract-index` |
| `COMPILE vindex INTO MODEL "path"` | Recompile vindex to weights | Planned |
| `DIFF "a" "b"` | Compare two vindexes | Planned |

### Knowledge Browser (pure vindex)

| Statement | Purpose | Status |
|---|---|---|
| `WALK "token" TOP n` | Feature scan -- what gate features fire | Working |
| `DESCRIBE "entity"` | Knowledge about an entity, grouped by layer band | Working |
| `SELECT ... FROM EDGES WHERE ...` | Query edges by layer, feature, token | Working |
| `EXPLAIN WALK "prompt"` | Per-layer feature trace | Working |

### Inference (requires model weights)

| Statement | Purpose | Status |
|---|---|---|
| `INFER "prompt" TOP n` | Full prediction with attention | Working (needs weights) |
| `INFER "prompt" TOP n COMPARE` | Walk FFN vs dense comparison | Working (needs weights) |

### Mutation

| Statement | Purpose | Status |
|---|---|---|
| `INSERT INTO EDGES (...) VALUES (...)` | Add edge | Planned |
| `DELETE FROM EDGES WHERE ...` | Remove edges | Planned |
| `UPDATE EDGES SET ... WHERE ...` | Modify edges | Planned |
| `MERGE "source" INTO "target"` | Merge vindexes | Planned |

### Introspection

| Statement | Purpose | Status |
|---|---|---|
| `SHOW MODELS` | List vindexes in current directory | Working |
| `SHOW LAYERS` | Layer-by-layer summary | Working |
| `SHOW RELATIONS` | Top tokens by frequency | Working |
| `SHOW FEATURES n` | Feature details at layer n | Working |
| `STATS` | Summary statistics | Working |

## Syntax Notes

- **Strings** use double or single quotes: `"France"` or `'France'`
- **Ranges** use dash: `LAYERS 0-33`
- **Comments** use `--`: `-- this is a comment`
- **Semicolons** end statements: `STATS;`
- **Pipe** chains statements: `WALK "x" TOP 5 |> EXPLAIN WALK "x";`
- **Case insensitive** keywords: `walk`, `WALK`, `Walk` all work

## WHERE Clause Operators

```sql
WHERE entity = "France"           -- equality
WHERE confidence > 0.5            -- greater than
WHERE confidence >= 0.5           -- greater or equal
WHERE layer != 26                 -- not equal
WHERE entity LIKE "Fran%"         -- pattern match
WHERE entity IN ("France", "Germany")  -- set membership
```

Multiple conditions with `AND`:

```sql
WHERE relation = "capital-of" AND confidence > 0.5
```

## ORDER BY and LIMIT

```sql
SELECT * FROM EDGES
    WHERE layer = 26
    ORDER BY confidence DESC
    LIMIT 10;
```

## REPL Commands

| Command | Action |
|---|---|
| `help`, `\h`, `\?` | Show help |
| `exit`, `quit`, `\q` | Exit REPL |
| Up/Down arrows | Navigate command history |
| Ctrl-R | Reverse search history |
| Ctrl-C | Cancel current input |
| Ctrl-D | Exit |

## Examples

### Explore what a model knows

```sql
USE "output/gemma3-4b.vindex";
STATS;
DESCRIBE "France";
DESCRIBE "Mozart";
DESCRIBE "Einstein";
SHOW FEATURES 26 LIMIT 10;
```

### Feature scan for a token

```sql
WALK "France" TOP 10;
WALK "Mozart" TOP 10;
WALK "capital" TOP 10;
```

### Trace how features activate across layers

```sql
EXPLAIN WALK "The capital of France is";
EXPLAIN WALK "Mozart was born in";
```

### Search edges

```sql
SELECT * FROM EDGES WHERE layer = 26 LIMIT 20;
SELECT * FROM EDGES WHERE layer = 26 AND feature = 9515;
SELECT * FROM EDGES ORDER BY confidence DESC LIMIT 10;
```

### Full demo script

```sql
-- Connect
USE "output/gemma3-4b.vindex";
STATS;

-- Browse
SHOW RELATIONS;
DESCRIBE "France";
SELECT entity, target, confidence
    FROM EDGES
    WHERE relation = "capital-of"
    ORDER BY confidence DESC
    LIMIT 10;

-- Trace
WALK "France" TOP 10;
EXPLAIN WALK "The capital of France is";

-- Inference (if model weights available)
INFER "The capital of France is" TOP 5;
INFER "The capital of France is" TOP 5 COMPARE;

-- Edit (coming soon)
INSERT INTO EDGES (entity, relation, target)
    VALUES ("John Coyle", "lives-in", "Colchester");
DESCRIBE "John Coyle";
DIFF "gemma3-4b.vindex" CURRENT;
COMPILE CURRENT INTO MODEL "gemma3-4b-edited/" FORMAT safetensors;
```
