# larql expert-server on fly.io

Deploy `larql-server` as a CPU-only MoE expert server on fly.io. The server loads the
`layers/` expert weights from a vindex slice and handles expert dispatch requests from
a local `larql run` client via `--moe-shards`.

## Prerequisites

- [`fly` CLI](https://fly.io/docs/hands-on/install-flyctl/) installed and authenticated
- `docker` (used by `fly deploy --remote-only` — build happens on fly infrastructure)
- The 26B vindex already extracted locally at `output/gemma4-26b-a4b-q4k.vindex`
- A HuggingFace account to host the sliced vindex

## Step 1 — Publish the vindex slice to HuggingFace

Create a minimal slice containing only the expert weights and tokenizer (~12.3 GB):

```bash
larql slice output/gemma4-26b-a4b-q4k.vindex \
  -o /tmp/gemma4-26b-expert-server.vindex \
  --preset expert-server

larql publish /tmp/gemma4-26b-expert-server.vindex \
  --hf-repo chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server
```

## Step 2 — Deploy one app (all experts)

```bash
fly apps create larql-expert-server
fly volumes create expert_data --size 25 --app larql-expert-server
fly deploy --app larql-expert-server --remote-only
```

On first start the machine downloads the vindex from HuggingFace (~10 min for 12 GB)
and caches it on the persistent volume. Subsequent restarts skip the download.

Set `HF_TOKEN` as a fly secret if the HuggingFace repo is private:

```bash
fly secrets set HF_TOKEN=hf_... --app larql-expert-server
```

## Step 3 — Shard by expert range (two apps)

Edit `fly.toml` and deploy two separate apps, each serving half the experts:

**App A — experts 0–63:**

```bash
# In fly.toml, set EXPERTS = "0-63" under [env], then:
fly apps create larql-expert-a
fly volumes create expert_data --size 25 --app larql-expert-a
fly deploy --app larql-expert-a --remote-only --config deploy/fly/fly.toml
fly secrets set HF_TOKEN=hf_... --app larql-expert-a  # if private repo
```

**App B — experts 64–127:**

```bash
# In fly.toml, set EXPERTS = "64-127" under [env], then:
fly apps create larql-expert-b
fly volumes create expert_data --size 25 --app larql-expert-b
fly deploy --app larql-expert-b --remote-only --config deploy/fly/fly.toml
fly secrets set HF_TOKEN=hf_... --app larql-expert-b  # if private repo
```

## Test it

```bash
larql run output/gemma4-26b-a4b-q4k.vindex --max-tokens 1 \
  --moe-shards "0-127=https://larql-expert-server.fly.dev" \
  "The capital of France is"
```

For the two-app sharded setup:

```bash
larql run output/gemma4-26b-a4b-q4k.vindex --max-tokens 20 \
  --moe-shards "0-63=https://larql-expert-a.fly.dev,64-127=https://larql-expert-b.fly.dev" \
  "The capital of France is"
```

## Cold start note

The first request after a fresh deploy triggers the vindex download from HuggingFace
(~10 min for 12 GB over the fly.io network). Subsequent starts reuse the `/data` volume.
`auto_stop_machines = false` in `fly.toml` keeps the machine running to avoid re-downloads
on the demo. Set it to `true` to reduce cost when idle.
