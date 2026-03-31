//! Tests for the larql-vindex crate.

use larql_vindex::{
    FeatureMeta, VectorIndex, VindexConfig, VindexLayerInfo,
};
use ndarray::{Array1, Array2};

fn make_top_k(token: &str, id: u32, logit: f32) -> larql_models::TopKEntry {
    larql_models::TopKEntry {
        token: token.to_string(),
        token_id: id,
        logit,
    }
}

fn make_meta(token: &str, id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: id,
        c_score: score,
        top_k: vec![make_top_k(token, id, score)],
    }
}

/// Build a small in-memory VectorIndex for testing.
fn test_index() -> VectorIndex {
    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;

    // Layer 0: 3 features × 4 hidden
    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = 1.0; // feature 0 responds to dim 0
    gate0[[1, 1]] = 1.0; // feature 1 responds to dim 1
    gate0[[2, 2]] = 1.0; // feature 2 responds to dim 2

    // Layer 1: 3 features × 4 hidden
    let mut gate1 = Array2::<f32>::zeros((num_features, hidden));
    gate1[[0, 3]] = 1.0;
    gate1[[1, 0]] = 0.5;
    gate1[[1, 1]] = 0.5;
    gate1[[2, 2]] = -1.0;

    let gate_vectors = vec![Some(gate0), Some(gate1)];

    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),
        Some(make_meta("French", 101, 0.88)),
        Some(make_meta("Europe", 102, 0.75)),
    ];
    let meta1 = vec![
        Some(make_meta("Berlin", 200, 0.90)),
        None, // feature 1 has no metadata
        Some(make_meta("Spain", 202, 0.70)),
    ];

    let down_meta = vec![Some(meta0), Some(meta1)];

    VectorIndex::new(gate_vectors, down_meta, num_layers, hidden)
}

// ══════════════════════════════════════════════════════════════
// CONSTRUCTION
// ══════════════════════════════════════════════════════════════

#[test]
fn new_index_has_correct_dimensions() {
    let idx = test_index();
    assert_eq!(idx.num_layers, 2);
    assert_eq!(idx.hidden_size, 4);
}

#[test]
fn loaded_layers() {
    let idx = test_index();
    assert_eq!(idx.loaded_layers(), vec![0, 1]);
}

#[test]
fn num_features_per_layer() {
    let idx = test_index();
    assert_eq!(idx.num_features(0), 3);
    assert_eq!(idx.num_features(1), 3);
    assert_eq!(idx.num_features(99), 0); // out of range
}

#[test]
fn total_counts() {
    let idx = test_index();
    assert_eq!(idx.total_gate_vectors(), 6); // 3 + 3
    assert_eq!(idx.total_down_meta(), 5); // 3 + 2 (one None)
}

// ══════════════════════════════════════════════════════════════
// FEATURE LOOKUP
// ══════════════════════════════════════════════════════════════

#[test]
fn feature_meta_lookup() {
    let idx = test_index();
    let meta = idx.feature_meta(0, 0).unwrap();
    assert_eq!(meta.top_token, "Paris");
    assert_eq!(meta.top_token_id, 100);
    assert!((meta.c_score - 0.95).abs() < 0.01);
}

#[test]
fn feature_meta_none_for_missing() {
    let idx = test_index();
    assert!(idx.feature_meta(1, 1).is_none()); // explicitly None
    assert!(idx.feature_meta(99, 0).is_none()); // out of range layer
    assert!(idx.feature_meta(0, 99).is_none()); // out of range feature
}

#[test]
fn down_meta_at_returns_slice() {
    let idx = test_index();
    let metas = idx.down_meta_at(0).unwrap();
    assert_eq!(metas.len(), 3);
    assert!(metas[0].is_some());
    assert!(metas[1].is_some());
    assert!(metas[2].is_some());

    let metas1 = idx.down_meta_at(1).unwrap();
    assert!(metas1[1].is_none()); // the gap
}

// ══════════════════════════════════════════════════════════════
// GATE KNN
// ══════════════════════════════════════════════════════════════

#[test]
fn gate_knn_finds_best_match() {
    let idx = test_index();

    // Query along dim 0 → should match feature 0 at layer 0
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = idx.gate_knn(0, &query, 1);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, 0); // feature 0
    assert!((hits[0].1 - 1.0).abs() < 0.01); // dot product = 1.0
}

#[test]
fn gate_knn_top_k_ordering() {
    let idx = test_index();

    // Query with components in dim 0 and dim 1
    let query = Array1::from_vec(vec![0.8, 0.6, 0.0, 0.0]);
    let hits = idx.gate_knn(0, &query, 3);

    assert_eq!(hits.len(), 3);
    // Feature 0 (dim 0): dot = 0.8
    // Feature 1 (dim 1): dot = 0.6
    // Feature 2 (dim 2): dot = 0.0
    assert_eq!(hits[0].0, 0); // highest
    assert_eq!(hits[1].0, 1);
}

#[test]
fn gate_knn_empty_for_missing_layer() {
    let idx = test_index();
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = idx.gate_knn(99, &query, 5);
    assert!(hits.is_empty());
}

// ══════════════════════════════════════════════════════════════
// WALK
// ══════════════════════════════════════════════════════════════

#[test]
fn walk_across_layers() {
    let idx = test_index();
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = idx.walk(&query, &[0, 1], 2);

    assert_eq!(trace.layers.len(), 2);

    // Layer 0: feature 0 fires (dim 0 = 1.0)
    let (layer, hits) = &trace.layers[0];
    assert_eq!(*layer, 0);
    assert!(!hits.is_empty());
    assert_eq!(hits[0].feature, 0);
    assert_eq!(hits[0].meta.top_token, "Paris");

    // Layer 1: feature 1 fires (dim 0 contributes 0.5)
    let (layer1, hits1) = &trace.layers[1];
    assert_eq!(*layer1, 1);
    assert!(!hits1.is_empty());
}

#[test]
fn walk_skips_features_without_meta() {
    let idx = test_index();
    // Query that activates feature 1 at layer 1 (which has no metadata)
    let query = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0]);
    let trace = idx.walk(&query, &[1], 3);

    // Feature 1 at layer 1 has None metadata — should be filtered out
    let (_, hits) = &trace.layers[0];
    for hit in hits {
        assert_ne!(hit.feature, 1); // feature 1 should not appear
    }
}

// ══════════════════════════════════════════════════════════════
// MUTATION
// ══════════════════════════════════════════════════════════════

#[test]
fn set_feature_meta() {
    let mut idx = test_index();
    assert!(idx.feature_meta(1, 1).is_none());

    let meta = make_meta("London", 300, 0.85);
    idx.set_feature_meta(1, 1, meta);

    let loaded = idx.feature_meta(1, 1).unwrap();
    assert_eq!(loaded.top_token, "London");
    assert_eq!(loaded.top_token_id, 300);
}

#[test]
fn delete_feature_meta() {
    let mut idx = test_index();
    assert!(idx.feature_meta(0, 0).is_some());

    idx.delete_feature_meta(0, 0);
    assert!(idx.feature_meta(0, 0).is_none());
}

#[test]
fn find_free_feature() {
    let mut idx = test_index();

    // Layer 0: all 3 features have metadata → no free slot
    assert!(idx.find_free_feature(0).is_none());

    // Layer 1: feature 1 is None → free slot
    assert_eq!(idx.find_free_feature(1), Some(1));

    // Delete one in layer 0
    idx.delete_feature_meta(0, 2);
    assert_eq!(idx.find_free_feature(0), Some(2));
}

#[test]
fn set_gate_vector() {
    let mut idx = test_index();
    let new_vec = Array1::from_vec(vec![0.0, 0.0, 0.0, 9.9]);
    idx.set_gate_vector(0, 1, &new_vec);

    // Query along dim 3 should now match feature 1 at layer 0
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    let hits = idx.gate_knn(0, &query, 1);
    assert_eq!(hits[0].0, 1); // feature 1
    assert!((hits[0].1 - 9.9).abs() < 0.01);
}

#[test]
fn mutation_does_not_affect_other_features() {
    let mut idx = test_index();

    // Mutate feature 0
    idx.set_feature_meta(0, 0, make_meta("Modified", 999, 0.5));

    // Feature 1 should be unchanged
    let meta1 = idx.feature_meta(0, 1).unwrap();
    assert_eq!(meta1.top_token, "French");
}

// ══════════════════════════════════════════════════════════════
// SAVE / LOAD ROUND-TRIP
// ══════════════════════════════════════════════════════════════

#[test]
fn save_and_load_down_meta_round_trip() {
    let idx = test_index();
    let dir = std::env::temp_dir().join("larql_test_down_meta_rt");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Save gate vectors + down_meta + config (needed for load_vindex)
    let layer_infos = idx.save_gate_vectors(&dir).unwrap();
    let count = idx.save_down_meta(&dir).unwrap();
    assert_eq!(count, 5); // 3 + 2 (one None skipped)

    let config = VindexConfig {
        version: 2,
        model: "test".into(),
        family: "test".into(),
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 100,
        embed_scale: 1.0,
        layers: layer_infos,
        down_top_k: 1,
        has_model_weights: false,
        source: None,
        checksums: None,
        extract_level: larql_vindex::ExtractLevel::Browse,
        layer_bands: None,
        model_config: None,
    };
    VectorIndex::save_config(&config, &dir).unwrap();

    // Load it back via the proper load path
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let idx2 = VectorIndex::load_vindex(&dir, &mut cb).unwrap();

    // Verify content
    let meta = idx2.feature_meta(0, 0).unwrap();
    assert_eq!(meta.top_token, "Paris");
    assert_eq!(meta.top_token_id, 100);

    let meta1 = idx2.feature_meta(1, 0).unwrap();
    assert_eq!(meta1.top_token, "Berlin");

    // Feature 1 at layer 1 should still be None
    assert!(idx2.feature_meta(1, 1).is_none());

    // Gate vectors should also round-trip
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = idx2.gate_knn(0, &query, 1);
    assert_eq!(hits[0].0, 0); // feature 0

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_and_load_gate_vectors_round_trip() {
    let idx = test_index();
    let dir = std::env::temp_dir().join("larql_test_gate_rt");
    std::fs::create_dir_all(&dir).unwrap();

    let layer_infos = idx.save_gate_vectors(&dir).unwrap();
    assert_eq!(layer_infos.len(), 2);
    assert_eq!(layer_infos[0].layer, 0);
    assert_eq!(layer_infos[0].num_features, 3);
    assert_eq!(layer_infos[1].layer, 1);

    // Verify file exists with expected size
    let gate_path = dir.join("gate_vectors.bin");
    assert!(gate_path.exists());
    let file_size = std::fs::metadata(&gate_path).unwrap().len();
    // 2 layers × 3 features × 4 hidden × 4 bytes = 96 bytes
    assert_eq!(file_size, 96);

    // Clean up
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_config_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_config_rt");
    std::fs::create_dir_all(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "test-model".into(),
        family: "test".into(),
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 100,
        embed_scale: 1.0,
        layers: vec![
            VindexLayerInfo { layer: 0, num_features: 3, offset: 0, length: 48, num_experts: None, num_features_per_expert: None },
            VindexLayerInfo { layer: 1, num_features: 3, offset: 48, length: 48, num_experts: None, num_features_per_expert: None },
        ],
        down_top_k: 10,
        has_model_weights: false,
        source: None,
        checksums: None,
        extract_level: larql_vindex::ExtractLevel::Browse,
        layer_bands: None,
        model_config: None,
    };

    VectorIndex::save_config(&config, &dir).unwrap();

    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();
    assert_eq!(loaded.model, "test-model");
    assert_eq!(loaded.num_layers, 2);
    assert_eq!(loaded.hidden_size, 4);
    assert_eq!(loaded.layers.len(), 2);
    assert_eq!(loaded.layers[0].num_features, 3);

    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// BINARY DOWN_META
// ══════════════════════════════════════════════════════════════

#[test]
fn binary_down_meta_write_read_round_trip() {
    let idx = test_index();
    let dir = std::env::temp_dir().join("larql_test_binary_dm");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Write binary format
    let count = larql_vindex::down_meta::write_binary(
        &dir,
        &[
            Some(vec![
                Some(make_meta("Paris", 100, 0.95)),
                Some(make_meta("French", 101, 0.88)),
                None,
            ]),
            Some(vec![
                Some(make_meta("Berlin", 200, 0.90)),
                None,
                Some(make_meta("Spain", 202, 0.70)),
            ]),
        ],
        1, // top_k = 1
    ).unwrap();
    assert_eq!(count, 4); // 2 + 2 (Nones don't count)

    // Verify file exists and is much smaller than JSONL would be
    let bin_path = dir.join("down_meta.bin");
    assert!(bin_path.exists());
    let bin_size = std::fs::metadata(&bin_path).unwrap().len();
    // Header (16) + 2 layers × (4 bytes layer header + 3 features × (4+4+1×8) bytes)
    assert!(bin_size > 0);
    assert!(bin_size < 200); // should be very small for 6 features

    // Read back — needs a tokenizer for string resolution
    // Create a minimal tokenizer that maps IDs to strings
    // Since we can't easily create a real tokenizer in tests,
    // verify the raw binary structure is correct
    let data = std::fs::read(&bin_path).unwrap();
    // Check magic
    assert_eq!(u32::from_le_bytes([data[0], data[1], data[2], data[3]]), 0x444D4554);
    // Check version
    assert_eq!(u32::from_le_bytes([data[4], data[5], data[6], data[7]]), 1);
    // Check num_layers
    assert_eq!(u32::from_le_bytes([data[8], data[9], data[10], data[11]]), 2);
    // Check top_k
    assert_eq!(u32::from_le_bytes([data[12], data[13], data[14], data[15]]), 1);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_down_meta_writes_both_formats() {
    let idx = test_index();
    let dir = std::env::temp_dir().join("larql_test_both_dm");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let count = idx.save_down_meta(&dir).unwrap();
    assert_eq!(count, 5); // 3 + 2

    // Both files should exist
    assert!(dir.join("down_meta.bin").exists());
    assert!(dir.join("down_meta.jsonl").exists());

    // Binary should be smaller than JSONL
    let bin_size = std::fs::metadata(dir.join("down_meta.bin")).unwrap().len();
    let jsonl_size = std::fs::metadata(dir.join("down_meta.jsonl")).unwrap().len();
    assert!(bin_size < jsonl_size, "binary ({bin_size}) should be smaller than JSONL ({jsonl_size})");

    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// ERROR HANDLING
// ══════════════════════════════════════════════════════════════

#[test]
fn load_nonexistent_vindex_errors() {
    let mut cb = larql_vindex::SilentLoadCallbacks;
    let result = VectorIndex::load_vindex(
        std::path::Path::new("/nonexistent/fake.vindex"),
        &mut cb,
    );
    assert!(result.is_err());
}

#[test]
fn load_nonexistent_config_errors() {
    let result = larql_vindex::load_vindex_config(
        std::path::Path::new("/nonexistent/fake.vindex"),
    );
    assert!(result.is_err());
}

// ══════════════════════════════════════════════════════════════
// LAYER BANDS
// ══════════════════════════════════════════════════════════════

#[test]
fn layer_bands_gemma3_4b() {
    let bands = larql_vindex::LayerBands::for_family("gemma3", 34).unwrap();
    assert_eq!(bands.syntax, (0, 13));
    assert_eq!(bands.knowledge, (14, 27));
    assert_eq!(bands.output, (28, 33));
}

#[test]
fn layer_bands_gemma2_9b() {
    let bands = larql_vindex::LayerBands::for_family("gemma2", 42).unwrap();
    assert_eq!(bands.syntax, (0, 16));
    assert_eq!(bands.knowledge, (17, 34));
    assert_eq!(bands.output, (35, 41));
}

#[test]
fn layer_bands_llama3_70b() {
    let bands = larql_vindex::LayerBands::for_family("llama", 80).unwrap();
    assert_eq!(bands.syntax, (0, 31));
    assert_eq!(bands.knowledge, (32, 63));
    assert_eq!(bands.output, (64, 79));
}

#[test]
fn layer_bands_llama3_8b() {
    let bands = larql_vindex::LayerBands::for_family("llama", 32).unwrap();
    assert_eq!(bands.syntax, (0, 12));
    assert_eq!(bands.knowledge, (13, 25));
    assert_eq!(bands.output, (26, 31));
}

#[test]
fn layer_bands_mixtral() {
    let bands = larql_vindex::LayerBands::for_family("mixtral", 32).unwrap();
    assert_eq!(bands.syntax, (0, 12));
    assert_eq!(bands.knowledge, (13, 25));
    assert_eq!(bands.output, (26, 31));
}

#[test]
fn layer_bands_gpt2_small() {
    let bands = larql_vindex::LayerBands::for_family("gpt2", 12).unwrap();
    assert_eq!(bands.syntax, (0, 4));
    assert_eq!(bands.knowledge, (5, 9));
    assert_eq!(bands.output, (10, 11));
}

#[test]
fn layer_bands_unknown_family_fallback() {
    // Unknown family with enough layers → falls back to heuristic
    let bands = larql_vindex::LayerBands::for_family("unknown_model", 40).unwrap();
    assert_eq!(bands.syntax.0, 0);
    assert!(bands.knowledge.0 > bands.syntax.1);
    assert!(bands.output.0 > bands.knowledge.1);
    assert_eq!(bands.output.1, 39);
}

#[test]
fn layer_bands_tiny_model_returns_none() {
    // Too few layers to band meaningfully
    assert!(larql_vindex::LayerBands::for_family("test", 2).is_none());
    assert!(larql_vindex::LayerBands::for_family("test", 4).is_none());
}

#[test]
fn layer_bands_band_for_layer() {
    let bands = larql_vindex::LayerBands::for_family("gemma3", 34).unwrap();
    assert_eq!(bands.band_for_layer(0), "syntax");
    assert_eq!(bands.band_for_layer(13), "syntax");
    assert_eq!(bands.band_for_layer(14), "knowledge");
    assert_eq!(bands.band_for_layer(27), "knowledge");
    assert_eq!(bands.band_for_layer(28), "output");
    assert_eq!(bands.band_for_layer(33), "output");
}

#[test]
fn v1_config_loads_with_defaults() {
    // Simulate a v1 index.json that lacks new fields
    let dir = std::env::temp_dir().join("larql_test_v1_compat");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let v1_json = r#"{
        "version": 1,
        "model": "old-model",
        "family": "test",
        "num_layers": 32,
        "hidden_size": 4,
        "intermediate_size": 3,
        "vocab_size": 100,
        "embed_scale": 1.0,
        "layers": [],
        "down_top_k": 10
    }"#;
    std::fs::write(dir.join("index.json"), v1_json).unwrap();

    let config = larql_vindex::load_vindex_config(&dir).unwrap();
    assert_eq!(config.version, 1);
    assert_eq!(config.model, "old-model");
    // New fields should have sensible defaults
    assert_eq!(config.extract_level, larql_vindex::ExtractLevel::Browse);
    assert!(config.layer_bands.is_none());
    assert!(config.source.is_none());
    assert!(config.checksums.is_none());
    assert!(!config.has_model_weights);
    assert!(config.model_config.is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn v2_config_full_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_v2_full_rt");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Write a dummy gate_vectors.bin so checksums have something to hash
    std::fs::write(dir.join("gate_vectors.bin"), b"test data").unwrap();

    let checksums = larql_vindex::checksums::compute_checksums(&dir).ok();

    let config = VindexConfig {
        version: 2,
        model: "google/gemma-3-4b-it".into(),
        family: "gemma3".into(),
        source: Some(larql_vindex::VindexSource {
            huggingface_repo: Some("google/gemma-3-4b-it".into()),
            huggingface_revision: Some("abc123".into()),
            safetensors_sha256: None,
            extracted_at: "2026-04-01T12:00:00Z".into(),
            larql_version: "0.1.0".into(),
        }),
        checksums,
        num_layers: 34,
        hidden_size: 2560,
        intermediate_size: 10240,
        vocab_size: 262144,
        embed_scale: 50.596,
        extract_level: larql_vindex::ExtractLevel::Inference,
        layer_bands: Some(larql_vindex::LayerBands {
            syntax: (0, 13),
            knowledge: (14, 27),
            output: (28, 33),
        }),
        layers: vec![],
        down_top_k: 10,
        has_model_weights: true,
        model_config: Some(larql_vindex::VindexModelConfig {
            model_type: "gemma3".into(),
            head_dim: 256,
            num_q_heads: 8,
            num_kv_heads: 4,
            rope_base: 10000.0,
            sliding_window: Some(1024),
            moe: None,
        }),
    };

    VectorIndex::save_config(&config, &dir).unwrap();
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();

    // Verify all v2 fields round-trip
    assert_eq!(loaded.version, 2);
    assert_eq!(loaded.model, "google/gemma-3-4b-it");
    assert_eq!(loaded.extract_level, larql_vindex::ExtractLevel::Inference);
    assert!(loaded.has_model_weights);

    let source = loaded.source.unwrap();
    assert_eq!(source.huggingface_repo.as_deref(), Some("google/gemma-3-4b-it"));
    assert_eq!(source.huggingface_revision.as_deref(), Some("abc123"));
    assert_eq!(source.larql_version, "0.1.0");

    let bands = loaded.layer_bands.unwrap();
    assert_eq!(bands.syntax, (0, 13));
    assert_eq!(bands.knowledge, (14, 27));
    assert_eq!(bands.output, (28, 33));

    let mc = loaded.model_config.unwrap();
    assert_eq!(mc.model_type, "gemma3");
    assert_eq!(mc.head_dim, 256);
    assert_eq!(mc.sliding_window, Some(1024));
    assert!(mc.moe.is_none());

    assert!(loaded.checksums.is_some());
    let cs = loaded.checksums.unwrap();
    assert!(cs.contains_key("gate_vectors.bin"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn v2_config_with_moe() {
    let dir = std::env::temp_dir().join("larql_test_v2_moe");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "mistralai/Mixtral-8x7B".into(),
        family: "mixtral".into(),
        source: None,
        checksums: None,
        num_layers: 32,
        hidden_size: 4096,
        intermediate_size: 14336,
        vocab_size: 32000,
        embed_scale: 64.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        layer_bands: Some(larql_vindex::LayerBands::for_family("mixtral", 32).unwrap()),
        layers: vec![],
        down_top_k: 10,
        has_model_weights: false,
        model_config: Some(larql_vindex::VindexModelConfig {
            model_type: "mixtral".into(),
            head_dim: 128,
            num_q_heads: 32,
            num_kv_heads: 8,
            rope_base: 1000000.0,
            sliding_window: None,
            moe: Some(larql_vindex::MoeConfig {
                num_experts: 8,
                top_k: 2,
                shared_expert: false,
                router_type: "top_k_softmax".into(),
            }),
        }),
    };

    VectorIndex::save_config(&config, &dir).unwrap();
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();

    let mc = loaded.model_config.unwrap();
    let moe = mc.moe.unwrap();
    assert_eq!(moe.num_experts, 8);
    assert_eq!(moe.top_k, 2);
    assert!(!moe.shared_expert);
    assert_eq!(moe.router_type, "top_k_softmax");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn moe_index_gate_knn_across_experts() {
    // Simulate a MoE layer: 2 experts × 3 features = 6 total features
    // Expert 0 features respond to dims 0,1,2
    // Expert 1 features respond to dim 3
    let hidden = 4;
    let features_per_expert = 3;
    let num_experts = 2;

    // Concatenate expert gate matrices (as build_vindex would)
    let mut gate0 = Array2::<f32>::zeros((num_experts * features_per_expert, hidden));
    // Expert 0
    gate0[[0, 0]] = 10.0; // E0F0 responds to dim 0
    gate0[[1, 1]] = 10.0; // E0F1 responds to dim 1
    gate0[[2, 2]] = 10.0; // E0F2 responds to dim 2
    // Expert 1
    gate0[[3, 3]] = 10.0; // E1F0 responds to dim 3
    gate0[[4, 0]] = 5.0; gate0[[4, 3]] = 5.0; // E1F1 mixed
    gate0[[5, 1]] = 3.0;  // E1F2 weak dim 1

    let gate_vectors = vec![Some(gate0)];

    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),    // E0F0
        Some(make_meta("Berlin", 101, 0.92)),   // E0F1
        Some(make_meta("Tokyo", 102, 0.88)),    // E0F2
        Some(make_meta("London", 103, 0.90)),   // E1F0
        Some(make_meta("Rome", 104, 0.85)),     // E1F1
        Some(make_meta("Madrid", 105, 0.80)),   // E1F2
    ];
    let down_meta = vec![Some(meta0)];

    let idx = VectorIndex::new(gate_vectors, down_meta, 1, hidden);
    assert_eq!(idx.num_features(0), 6); // 2 experts × 3 features

    // Query dim 0 → should match E0F0 (Paris) strongest
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = idx.gate_knn(0, &query, 2);
    assert_eq!(hits[0].0, 0); // E0F0 = Paris
    assert_eq!(hits[1].0, 4); // E1F1 = Rome (has dim 0 component)

    // Query dim 3 → should match E1F0 (London) strongest
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    let hits = idx.gate_knn(0, &query, 2);
    assert_eq!(hits[0].0, 3); // E1F0 = London
    assert_eq!(hits[1].0, 4); // E1F1 = Rome (has dim 3 component)

    // Walk should find features across experts
    let query = Array1::from_vec(vec![0.5, 0.0, 0.0, 0.5]);
    let trace = idx.walk(&query, &[0], 3);
    let (_, hits) = &trace.layers[0];
    // Both E0F0 (Paris, dim0) and E1F0 (London, dim3) should appear
    let tokens: Vec<&str> = hits.iter().map(|h| h.meta.top_token.as_str()).collect();
    assert!(tokens.contains(&"Paris") || tokens.contains(&"London"));
}

#[test]
fn moe_layer_info_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_moe_layer_info");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "test-moe".into(),
        family: "mixtral".into(),
        source: None,
        checksums: None,
        num_layers: 1,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 100,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        layer_bands: larql_vindex::LayerBands::for_family("mixtral", 32),
        layers: vec![
            VindexLayerInfo {
                layer: 0,
                num_features: 24, // 8 experts × 3 features
                offset: 0,
                length: 384,
                num_experts: Some(8),
                num_features_per_expert: Some(3),
            },
        ],
        down_top_k: 10,
        has_model_weights: false,
        model_config: Some(larql_vindex::VindexModelConfig {
            model_type: "mixtral".into(),
            head_dim: 128,
            num_q_heads: 32,
            num_kv_heads: 8,
            rope_base: 1000000.0,
            sliding_window: None,
            moe: Some(larql_vindex::MoeConfig {
                num_experts: 8,
                top_k: 2,
                shared_expert: false,
                router_type: "top_k_softmax".into(),
            }),
        }),
    };

    VectorIndex::save_config(&config, &dir).unwrap();
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();

    // Verify MoE layer info round-trips
    assert_eq!(loaded.layers[0].num_experts, Some(8));
    assert_eq!(loaded.layers[0].num_features_per_expert, Some(3));
    assert_eq!(loaded.layers[0].num_features, 24);

    // Verify MoE config round-trips
    let moe = loaded.model_config.unwrap().moe.unwrap();
    assert_eq!(moe.num_experts, 8);
    assert_eq!(moe.top_k, 2);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn layer_bands_config_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_bands_rt");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "test-bands".into(),
        family: "test".into(),
        num_layers: 34,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 100,
        embed_scale: 1.0,
        layers: vec![],
        down_top_k: 10,
        has_model_weights: false,
        source: None,
        checksums: None,
        extract_level: larql_vindex::ExtractLevel::Browse,
        layer_bands: Some(larql_vindex::LayerBands {
            syntax: (0, 13),
            knowledge: (14, 27),
            output: (28, 33),
        }),
        model_config: None,
    };

    VectorIndex::save_config(&config, &dir).unwrap();
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();

    let bands = loaded.layer_bands.unwrap();
    assert_eq!(bands.syntax, (0, 13));
    assert_eq!(bands.knowledge, (14, 27));
    assert_eq!(bands.output, (28, 33));

    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// CHECKSUM VERIFICATION
// ══════════════════════════════════════════════════════════════

#[test]
fn checksum_compute_and_verify() {
    let dir = std::env::temp_dir().join("larql_test_checksums");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Write some test data
    std::fs::write(dir.join("gate_vectors.bin"), b"test gate data").unwrap();
    std::fs::write(dir.join("embeddings.bin"), b"test embed data").unwrap();
    std::fs::write(dir.join("down_meta.jsonl"), b"test down data").unwrap();

    // Compute checksums
    let checksums = larql_vindex::checksums::compute_checksums(&dir).unwrap();
    assert_eq!(checksums.len(), 3); // 3 files present
    assert!(checksums.contains_key("gate_vectors.bin"));
    assert!(checksums.contains_key("embeddings.bin"));
    assert!(checksums.contains_key("down_meta.jsonl"));

    // Verify — should all pass
    let results = larql_vindex::checksums::verify_checksums(&dir, &checksums).unwrap();
    assert!(results.iter().all(|(_, ok)| *ok));

    // Corrupt a file
    std::fs::write(dir.join("gate_vectors.bin"), b"corrupted!").unwrap();
    let results = larql_vindex::checksums::verify_checksums(&dir, &checksums).unwrap();
    let gate_result = results.iter().find(|(f, _)| f == "gate_vectors.bin").unwrap();
    assert!(!gate_result.1); // should fail

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn checksum_individual_file() {
    let dir = std::env::temp_dir().join("larql_test_sha256");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    std::fs::write(dir.join("test.bin"), b"hello world").unwrap();
    let hash = larql_vindex::checksums::sha256_file(&dir.join("test.bin")).unwrap();
    // SHA256 of "hello world" is known
    assert_eq!(hash, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");

    let _ = std::fs::remove_dir_all(&dir);
}

// ══════════════════════════════════════════════════════════════
// EXTRACT LEVEL
// ══════════════════════════════════════════════════════════════

#[test]
fn extract_level_serialization() {
    assert_eq!(format!("{}", larql_vindex::ExtractLevel::Browse), "browse");
    assert_eq!(format!("{}", larql_vindex::ExtractLevel::Inference), "inference");
    assert_eq!(format!("{}", larql_vindex::ExtractLevel::All), "all");

    // serde round-trip
    let json = serde_json::to_string(&larql_vindex::ExtractLevel::Inference).unwrap();
    assert_eq!(json, "\"inference\"");
    let parsed: larql_vindex::ExtractLevel = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, larql_vindex::ExtractLevel::Inference);
}

#[test]
fn extract_level_default_is_browse() {
    let level: larql_vindex::ExtractLevel = Default::default();
    assert_eq!(level, larql_vindex::ExtractLevel::Browse);
}

// ══════════════════════════════════════════════════════════════
// DESCRIBE TYPES
// ══════════════════════════════════════════════════════════════

#[test]
fn label_source_display() {
    assert_eq!(format!("{}", larql_vindex::LabelSource::Probe), "probe");
    assert_eq!(format!("{}", larql_vindex::LabelSource::Cluster), "cluster");
    assert_eq!(format!("{}", larql_vindex::LabelSource::Pattern), "pattern");
    assert_eq!(format!("{}", larql_vindex::LabelSource::None), "");
}

#[test]
fn describe_edge_construction() {
    let edge = larql_vindex::DescribeEdge {
        relation: Some("capital".into()),
        source: larql_vindex::LabelSource::Probe,
        target: "Paris".into(),
        gate_score: 1436.9,
        layer_min: 27,
        layer_max: 27,
        count: 1,
        also_tokens: vec![],
    };
    assert_eq!(edge.relation.as_deref(), Some("capital"));
    assert_eq!(edge.source, larql_vindex::LabelSource::Probe);
    assert_eq!(edge.target, "Paris");
}

// ══════════════════════════════════════════════════════════════
// SOURCE PROVENANCE
// ══════════════════════════════════════════════════════════════

#[test]
fn source_provenance_round_trip() {
    let dir = std::env::temp_dir().join("larql_test_provenance");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let config = VindexConfig {
        version: 2,
        model: "test/provenance".into(),
        family: "test".into(),
        source: Some(larql_vindex::VindexSource {
            huggingface_repo: Some("google/gemma-3-4b-it".into()),
            huggingface_revision: Some("abc123def456".into()),
            safetensors_sha256: Some("deadbeef".into()),
            extracted_at: "2026-04-01T12:00:00Z".into(),
            larql_version: "0.1.0".into(),
        }),
        checksums: None,
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 3,
        vocab_size: 100,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::All,
        layer_bands: None,
        layers: vec![],
        down_top_k: 10,
        has_model_weights: true,
        model_config: None,
    };

    VectorIndex::save_config(&config, &dir).unwrap();
    let loaded = larql_vindex::load_vindex_config(&dir).unwrap();

    let src = loaded.source.unwrap();
    assert_eq!(src.huggingface_repo.as_deref(), Some("google/gemma-3-4b-it"));
    assert_eq!(src.huggingface_revision.as_deref(), Some("abc123def456"));
    assert_eq!(src.safetensors_sha256.as_deref(), Some("deadbeef"));
    assert_eq!(src.extracted_at, "2026-04-01T12:00:00Z");
    assert_eq!(src.larql_version, "0.1.0");
    assert_eq!(loaded.extract_level, larql_vindex::ExtractLevel::All);

    let _ = std::fs::remove_dir_all(&dir);
}
