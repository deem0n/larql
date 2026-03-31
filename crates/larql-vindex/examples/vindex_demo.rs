//! Vindex Demo — full v2 feature showcase.
//!
//! Demonstrates: build, KNN, walk, mutate, layer bands, MoE layout,
//! binary down_meta, source provenance, checksum verification.
//!
//! Run: cargo run -p larql-vindex --example vindex_demo

use larql_models::TopKEntry;
use larql_vindex::{FeatureMeta, VectorIndex, VindexConfig};
use ndarray::{Array1, Array2};

fn main() {
    println!("=== Vindex Demo (v2 — full feature showcase) ===\n");

    // ── 1. Build ──
    section("Build in-memory index");
    let index = build_demo_index();
    println!("  {} layers, {} features, {} with metadata",
        index.num_layers, index.total_gate_vectors(), index.total_down_meta());

    // ── 2. Layer bands ──
    section("Layer bands (per-family, exact boundaries)");
    let families = [
        ("gpt2", 12), ("gemma2", 26), ("llama", 32),
        ("gemma3", 34), ("qwen2", 40), ("llama", 80),
        ("mixtral", 32), ("unknown", 3),
    ];
    for &(family, layers) in &families {
        match larql_vindex::LayerBands::for_family(family, layers) {
            Some(b) => println!("  {:<8} {:>2}L  syntax={:>2}-{:<2}  knowledge={:>2}-{:<2}  output={:>2}-{:<2}",
                family, layers, b.syntax.0, b.syntax.1, b.knowledge.0, b.knowledge.1, b.output.0, b.output.1),
            None => println!("  {:<8} {:>2}L  (too few layers)", family, layers),
        }
    }

    // ── 3. Gate KNN ──
    section("Gate KNN");
    let q = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    println!("  Query [1,0,0,0]:");
    for (feat, score) in index.gate_knn(0, &q, 3) {
        let tok = index.feature_meta(0, feat).map(|m| m.top_token.as_str()).unwrap_or("-");
        println!("    F{}: {} ({:.1})", feat, tok, score);
    }

    // ── 4. Walk ──
    section("Walk (multi-layer)");
    let trace = index.walk(&q, &[0, 1], 2);
    for (layer, hits) in &trace.layers {
        if hits.is_empty() { println!("  L{}: (none)", layer); continue; }
        for h in hits { println!("  L{}: F{} → {} ({:.1})", layer, h.feature, h.meta.top_token, h.gate_score); }
    }

    // ── 5. MoE layout ──
    section("MoE layout (2 experts × 3 features)");
    let moe_index = build_moe_index();
    println!("  Total features at L0: {} (2 experts × 3)", moe_index.num_features(0));
    let q_dim0 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let q_dim3 = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    println!("  Query [1,0,0,0] (Expert 0 territory):");
    for (f, s) in moe_index.gate_knn(0, &q_dim0, 2) {
        let e = if f < 3 { 0 } else { 1 };
        let tok = moe_index.feature_meta(0, f).map(|m| m.top_token.as_str()).unwrap_or("-");
        println!("    E{}:F{} → {} ({:.1})", e, f % 3, tok, s);
    }
    println!("  Query [0,0,0,1] (Expert 1 territory):");
    for (f, s) in moe_index.gate_knn(0, &q_dim3, 2) {
        let e = if f < 3 { 0 } else { 1 };
        let tok = moe_index.feature_meta(0, f).map(|m| m.top_token.as_str()).unwrap_or("-");
        println!("    E{}:F{} → {} ({:.1})", e, f % 3, tok, s);
    }

    // ── 6. Mutate ──
    section("Mutate (insert + delete)");
    let mut index = build_demo_index();
    let slot = index.find_free_feature(0).unwrap();
    index.set_gate_vector(0, slot, &Array1::from_vec(vec![0.0, 0.0, 0.0, 10.0]));
    index.set_feature_meta(0, slot, meta("Canberra", 104, 0.85));
    println!("  Inserted F{} → Canberra", slot);
    index.delete_feature_meta(0, 2);
    println!("  Deleted F2 (was Tokyo)");

    // ── 7. Save with binary down_meta + checksums ──
    section("Save (binary down_meta + checksums + verification)");
    let dir = std::env::temp_dir().join("larql_vindex_demo_v2_full");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let layer_infos = index.save_gate_vectors(&dir).unwrap();
    let dm_count = index.save_down_meta(&dir).unwrap();

    let bin_size = std::fs::metadata(dir.join("down_meta.bin")).unwrap().len();
    let jsonl_size = std::fs::metadata(dir.join("down_meta.jsonl")).unwrap().len();
    println!("  down_meta.bin:   {} bytes", bin_size);
    println!("  down_meta.jsonl: {} bytes", jsonl_size);
    println!("  Compression:     {:.0}% smaller", (1.0 - bin_size as f64 / jsonl_size as f64) * 100.0);
    println!("  Features saved:  {}", dm_count);

    let config = VindexConfig {
        version: 2,
        model: "demo-model/v2".into(),
        family: "demo".into(),
        source: Some(larql_vindex::VindexSource {
            huggingface_repo: Some("demo/demo-model".into()),
            huggingface_revision: Some("main".into()),
            safetensors_sha256: None,
            extracted_at: "2026-04-01T12:00:00Z".into(),
            larql_version: env!("CARGO_PKG_VERSION").into(),
        }),
        checksums: larql_vindex::checksums::compute_checksums(&dir).ok(),
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 5,
        vocab_size: 200,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        layer_bands: Some(larql_vindex::LayerBands {
            syntax: (0, 0),
            knowledge: (0, 1),
            output: (1, 1),
        }),
        layers: layer_infos,
        down_top_k: 1,
        has_model_weights: false,
        model_config: None,
    };
    VectorIndex::save_config(&config, &dir).unwrap();

    // Verify checksums
    if let Some(ref stored) = config.checksums {
        let results = larql_vindex::checksums::verify_checksums(&dir, stored).unwrap();
        println!("  Checksums:");
        for (file, ok) in &results {
            println!("    {}: {}", file, if *ok { "OK" } else { "MISMATCH" });
        }
    }

    // ── 8. Reload and verify ──
    section("Reload and verify round-trip");
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let loaded = VectorIndex::load_vindex(&dir, &mut cb).unwrap();
    let loaded_config = larql_vindex::load_vindex_config(&dir).unwrap();

    println!("  Version:       {}", loaded_config.version);
    println!("  Extract level: {}", loaded_config.extract_level);
    println!("  Features:      {}", loaded.total_gate_vectors());
    println!("  With meta:     {}", loaded.total_down_meta());

    if let Some(ref src) = loaded_config.source {
        println!("  Source:        {}", src.huggingface_repo.as_deref().unwrap_or("?"));
    }

    let hits = loaded.gate_knn(0, &Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]), 1);
    let m = loaded.feature_meta(0, hits[0].0).unwrap();
    println!("  KNN [1,0,0,0]:  F{} → {} (round-trip OK)", hits[0].0, m.top_token);

    let _ = std::fs::remove_dir_all(&dir);
    println!("\n=== Done ===");
}

fn section(name: &str) { println!("\n── {} ──\n", name); }

fn meta(token: &str, id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.into(),
        top_token_id: id,
        c_score: score,
        top_k: vec![TopKEntry { token: token.into(), token_id: id, logit: score }],
    }
}

fn build_demo_index() -> VectorIndex {
    let h = 4;
    let mut g0 = Array2::<f32>::zeros((5, h));
    g0[[0, 0]] = 10.0;
    g0[[1, 1]] = 10.0;
    g0[[2, 2]] = 10.0;
    g0[[3, 0]] = 5.0; g0[[3, 1]] = 5.0;
    let g1 = Array2::<f32>::zeros((5, h));
    let m0 = vec![
        Some(meta("Paris", 100, 0.95)),
        Some(meta("Berlin", 101, 0.92)),
        Some(meta("Tokyo", 102, 0.88)),
        Some(meta("European", 103, 0.70)),
        None,
    ];
    let m1 = vec![None; 5];
    VectorIndex::new(vec![Some(g0), Some(g1)], vec![Some(m0), Some(m1)], 2, h)
}

fn build_moe_index() -> VectorIndex {
    let h = 4;
    let mut g = Array2::<f32>::zeros((6, h));
    g[[0, 0]] = 10.0; g[[1, 1]] = 10.0; g[[2, 2]] = 10.0; // Expert 0
    g[[3, 3]] = 10.0; g[[4, 0]] = 5.0; g[[4, 3]] = 5.0; g[[5, 1]] = 3.0; // Expert 1
    let m = vec![
        Some(meta("Paris", 100, 0.95)),
        Some(meta("Berlin", 101, 0.92)),
        Some(meta("Tokyo", 102, 0.88)),
        Some(meta("London", 103, 0.90)),
        Some(meta("Rome", 104, 0.85)),
        Some(meta("Madrid", 105, 0.80)),
    ];
    VectorIndex::new(vec![Some(g)], vec![Some(m)], 1, h)
}
