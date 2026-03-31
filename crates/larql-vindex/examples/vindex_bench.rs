//! Vindex Benchmark — measures KNN, walk, load, save, and binary down_meta performance.
//!
//! Creates realistic-sized synthetic indexes and times core operations.
//! No real model needed — pure in-memory benchmarks.
//!
//! Run: cargo run -p larql-vindex --example vindex_bench --release

use larql_models::TopKEntry;
use larql_vindex::{FeatureMeta, VectorIndex, VindexConfig};
use ndarray::{Array1, Array2};
use std::time::Instant;

fn main() {
    println!("=== Vindex Benchmark ===\n");

    // ── Configuration ──
    let hidden = 256;       // reduced from 2560 for bench speed
    let features = 1024;    // reduced from 10240
    let num_layers = 8;     // reduced from 34
    let top_k_meta = 5;
    let knn_top_k = 10;

    println!("Config: {}L × {} features × {} hidden ({}K gate vectors)\n",
        num_layers, features, hidden,
        (num_layers * features * hidden * 4) / 1024);

    // ── Build synthetic index ──
    let start = Instant::now();
    let index = build_synthetic_index(num_layers, features, hidden, top_k_meta);
    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("Build:           {:.1}ms ({} features, {} with meta)",
        build_ms, index.total_gate_vectors(), index.total_down_meta());

    // ── Gate KNN (single layer) ──
    let query = random_query(hidden);
    let warmup_iters = 10;
    let bench_iters = 100;

    // Warmup
    for _ in 0..warmup_iters {
        index.gate_knn(0, &query, knn_top_k);
    }

    let start = Instant::now();
    for _ in 0..bench_iters {
        for layer in 0..num_layers {
            index.gate_knn(layer, &query, knn_top_k);
        }
    }
    let knn_total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let knn_per_layer = knn_total_ms / (bench_iters * num_layers) as f64;
    let knn_full_walk = knn_per_layer * num_layers as f64;
    println!("Gate KNN:        {:.3}ms/layer, {:.1}ms/walk ({} layers × {} iters)",
        knn_per_layer, knn_full_walk, num_layers, bench_iters);

    // ── Walk (all layers) ──
    let layers: Vec<usize> = (0..num_layers).collect();
    let start = Instant::now();
    for _ in 0..bench_iters {
        let _ = index.walk(&query, &layers, knn_top_k);
    }
    let walk_total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let walk_per = walk_total_ms / bench_iters as f64;
    println!("Walk:            {:.3}ms/walk ({} layers, top-{})",
        walk_per, num_layers, knn_top_k);

    // ── Feature lookup ──
    let start = Instant::now();
    for _ in 0..100_000 {
        let _ = index.feature_meta(4, 512);
    }
    let lookup_ns = start.elapsed().as_nanos() / 100_000;
    println!("Feature lookup:  {}ns/lookup", lookup_ns);

    // ── Save to disk ──
    let dir = std::env::temp_dir().join("larql_vindex_bench");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let start = Instant::now();
    let layer_infos = index.save_gate_vectors(&dir).unwrap();
    let gate_ms = start.elapsed().as_secs_f64() * 1000.0;
    let gate_size = std::fs::metadata(dir.join("gate_vectors.bin")).unwrap().len();
    println!("Save gates:      {:.1}ms ({:.1} MB)", gate_ms, gate_size as f64 / 1_048_576.0);

    let start = Instant::now();
    let dm_count = index.save_down_meta(&dir).unwrap();
    let dm_ms = start.elapsed().as_secs_f64() * 1000.0;
    let bin_size = std::fs::metadata(dir.join("down_meta.bin")).unwrap().len();
    let jsonl_size = std::fs::metadata(dir.join("down_meta.jsonl")).unwrap().len();
    println!("Save down_meta:  {:.1}ms ({} records)", dm_ms, dm_count);
    println!("  Binary:        {:.1} KB", bin_size as f64 / 1024.0);
    println!("  JSONL:         {:.1} KB", jsonl_size as f64 / 1024.0);
    println!("  Ratio:         {:.1}x smaller", jsonl_size as f64 / bin_size as f64);

    // Save config for load test
    let config = VindexConfig {
        version: 2,
        model: "bench-model".into(),
        family: "bench".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: features,
        vocab_size: 100,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        layer_bands: None,
        layers: layer_infos,
        down_top_k: top_k_meta,
        has_model_weights: false,
        model_config: None,
    };
    VectorIndex::save_config(&config, &dir).unwrap();

    // ── Load from disk ──
    let mut cb = larql_vindex::SilentLoadCallbacks;

    let start = Instant::now();
    let loaded = VectorIndex::load_vindex(&dir, &mut cb).unwrap();
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("Load vindex:     {:.1}ms ({} features)", load_ms, loaded.total_gate_vectors());

    // Verify loaded index works
    let hits = loaded.gate_knn(0, &query, 1);
    let original_hits = index.gate_knn(0, &query, 1);
    assert_eq!(hits[0].0, original_hits[0].0, "loaded index should match original");

    // ── Checksum computation ──
    let start = Instant::now();
    let _checksums = larql_vindex::checksums::compute_checksums(&dir).unwrap();
    let checksum_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("Checksums:       {:.1}ms (SHA256 of all files)", checksum_ms);

    // ── Mutation benchmark ──
    let mut mutable_index = loaded;
    let meta = FeatureMeta {
        top_token: "test".into(),
        top_token_id: 42,
        c_score: 0.99,
        top_k: vec![TopKEntry { token: "test".into(), token_id: 42, logit: 0.99 }],
    };
    let gate_vec = random_query(hidden);

    let start = Instant::now();
    for i in 0..1000 {
        let layer = i % num_layers;
        let feat = i % features;
        mutable_index.set_feature_meta(layer, feat, meta.clone());
        mutable_index.set_gate_vector(layer, feat, &gate_vec);
    }
    let mutate_ns = start.elapsed().as_nanos() / 1000;
    println!("Mutate:          {}ns/op (set meta + gate vector)", mutate_ns);

    // ── MoE scaling ──
    println!("\n── MoE Scaling ──\n");
    for n_experts in [1, 2, 4, 8] {
        let total_features = features * n_experts;
        let mut gate = Array2::<f32>::zeros((total_features, hidden));
        // Fill with random-ish values
        for i in 0..total_features {
            gate[[i, i % hidden]] = 1.0;
        }
        let moe_idx = VectorIndex::new(
            vec![Some(gate)],
            vec![None],
            1,
            hidden,
        );
        let q = random_query(hidden);

        let start = Instant::now();
        for _ in 0..bench_iters {
            moe_idx.gate_knn(0, &q, knn_top_k);
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;
        println!("  {}x experts ({} features): {:.3}ms/KNN",
            n_experts, total_features, ms);
    }

    let _ = std::fs::remove_dir_all(&dir);
    println!("\n=== Done ===");
}

fn random_query(hidden: usize) -> Array1<f32> {
    // Deterministic pseudo-random for reproducibility
    let mut v = vec![0.0f32; hidden];
    for i in 0..hidden {
        v[i] = ((i * 7 + 13) % 100) as f32 / 100.0 - 0.5;
    }
    Array1::from_vec(v)
}

fn build_synthetic_index(
    num_layers: usize,
    features: usize,
    hidden: usize,
    top_k: usize,
) -> VectorIndex {
    let mut gate_vectors = Vec::with_capacity(num_layers);
    let mut down_meta = Vec::with_capacity(num_layers);

    for _layer in 0..num_layers {
        // Create gate matrix with sparse structure (each feature has one strong direction)
        let mut gate = Array2::<f32>::zeros((features, hidden));
        for f in 0..features {
            gate[[f, f % hidden]] = 1.0;
            if f + 1 < hidden {
                gate[[f, (f + 1) % hidden]] = 0.3; // some cross-activation
            }
        }
        gate_vectors.push(Some(gate));

        // Create metadata for every feature
        let metas: Vec<Option<FeatureMeta>> = (0..features)
            .map(|f| {
                let top_k_entries: Vec<TopKEntry> = (0..top_k)
                    .map(|k| TopKEntry {
                        token: format!("tok_{}_{}", f, k),
                        token_id: (f * top_k + k) as u32,
                        logit: 1.0 - k as f32 * 0.1,
                    })
                    .collect();
                Some(FeatureMeta {
                    top_token: format!("tok_{}", f),
                    top_token_id: f as u32,
                    c_score: 0.9 - (f as f32 * 0.001),
                    top_k: top_k_entries,
                })
            })
            .collect();
        down_meta.push(Some(metas));
    }

    VectorIndex::new(gate_vectors, down_meta, num_layers, hidden)
}
