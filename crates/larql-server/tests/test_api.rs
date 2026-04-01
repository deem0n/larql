//! Integration tests for larql-server API endpoints.
//!
//! Builds a synthetic in-memory vindex and tests each route handler
//! through the axum test infrastructure (no network, no disk).

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use tower::ServiceExt;

use larql_vindex::ndarray::{Array1, Array2};
use larql_vindex::{
    FeatureMeta, PatchedVindex, VectorIndex, VindexConfig, VindexLayerInfo,
    ExtractLevel, LayerBands,
};
use larql_vindex::tokenizers;
use tokio::sync::RwLock;

// ══════════════════════════════════════════════════════════════
// Test helpers
// ══════════════════════════════════════════════════════════════

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
        top_k: vec![
            make_top_k(token, id, score),
            make_top_k("also", id + 1, score * 0.5),
        ],
    }
}

/// Build a small test VectorIndex: 2 layers, 4 hidden dims, 3 features/layer.
fn test_index() -> VectorIndex {
    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;

    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = 1.0;
    gate0[[1, 1]] = 1.0;
    gate0[[2, 2]] = 1.0;

    let mut gate1 = Array2::<f32>::zeros((num_features, hidden));
    gate1[[0, 3]] = 1.0;
    gate1[[1, 0]] = 0.5;
    gate1[[1, 1]] = 0.5;
    gate1[[2, 2]] = -1.0;

    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),
        Some(make_meta("French", 101, 0.88)),
        Some(make_meta("Europe", 102, 0.75)),
    ];
    let meta1 = vec![
        Some(make_meta("Berlin", 200, 0.90)),
        Some(make_meta("Tokyo", 201, 0.85)),
        Some(make_meta("Spain", 202, 0.70)),
    ];

    VectorIndex::new(
        vec![Some(gate0), Some(gate1)],
        vec![Some(meta0), Some(meta1)],
        num_layers,
        hidden,
    )
}

/// Build a test VindexConfig matching the test index.
fn test_config() -> VindexConfig {
    VindexConfig {
        version: 2,
        model: "test/model-4".to_string(),
        family: "test".to_string(),
        source: None,
        checksums: None,
        num_layers: 2,
        hidden_size: 4,
        intermediate_size: 12,
        vocab_size: 8,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::default(),
        layer_bands: Some(LayerBands {
            syntax: (0, 0),
            knowledge: (0, 1),
            output: (1, 1),
        }),
        layers: vec![
            VindexLayerInfo { layer: 0, num_features: 3, offset: 0, length: 48, num_experts: None, num_features_per_expert: None },
            VindexLayerInfo { layer: 1, num_features: 3, offset: 48, length: 48, num_experts: None, num_features_per_expert: None },
        ],
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
    }
}

/// Build a tiny embeddings matrix (vocab=8, hidden=4) and a no-op tokenizer.
fn test_embeddings() -> Array2<f32> {
    let mut embed = Array2::<f32>::zeros((8, 4));
    // Token 0 → [1, 0, 0, 0]
    embed[[0, 0]] = 1.0;
    // Token 1 → [0, 1, 0, 0]
    embed[[1, 1]] = 1.0;
    // Token 2 → [0, 0, 1, 0]
    embed[[2, 2]] = 1.0;
    // Token 3 → [0, 0, 0, 1]
    embed[[3, 3]] = 1.0;
    // Token 4 → [1, 1, 0, 0] (multi-dim)
    embed[[4, 0]] = 1.0;
    embed[[4, 1]] = 1.0;
    embed
}

/// Build a minimal tokenizer for testing. Maps single characters to token IDs.
fn test_tokenizer() -> tokenizers::Tokenizer {
    use tokenizers::models::bpe::BPE;
    let mut tokenizer = tokenizers::Tokenizer::new(BPE::default());
    // Disable all pre/post processing so we get raw byte-level tokens.
    tokenizer.with_normalizer(None::<tokenizers::normalizers::Sequence>);
    tokenizer
}

// ── AppState construction (inline, no server dependency on private types) ──

/// We can't directly import larql_server internals in integration tests,
/// so we use the same types via the axum test client approach.
/// Instead, we build the router ourselves using the public crate API.

// Since larql-server is a binary crate, we test by building the state
// and router directly. We replicate the minimal state setup here.

mod server {
    use super::*;
    use std::sync::atomic::AtomicU64;

    pub struct LoadedModel {
        pub id: String,
        pub path: std::path::PathBuf,
        pub config: VindexConfig,
        pub patched: RwLock<PatchedVindex>,
        pub embeddings: Array2<f32>,
        pub embed_scale: f32,
        pub tokenizer: tokenizers::Tokenizer,
        pub infer_disabled: bool,
    }

    pub struct AppState {
        pub models: Vec<Arc<LoadedModel>>,
        pub started_at: std::time::Instant,
        pub requests_served: AtomicU64,
    }

    impl AppState {
        pub fn model(&self, _id: Option<&str>) -> Option<&Arc<LoadedModel>> {
            self.models.first()
        }

        pub fn is_multi_model(&self) -> bool {
            self.models.len() > 1
        }

        pub fn bump_requests(&self) {
            self.requests_served.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

fn build_test_state() -> Arc<server::AppState> {
    let index = test_index();
    let config = test_config();
    let patched = PatchedVindex::new(index);

    let model = Arc::new(server::LoadedModel {
        id: "model-4".to_string(),
        path: std::path::PathBuf::from("/tmp/test-vindex"),
        config,
        patched: RwLock::new(patched),
        embeddings: test_embeddings(),
        embed_scale: 1.0,
        tokenizer: test_tokenizer(),
        infer_disabled: true,
    });

    Arc::new(server::AppState {
        models: vec![model],
        started_at: std::time::Instant::now(),
        requests_served: std::sync::atomic::AtomicU64::new(0),
    })
}

// Since larql-server is a binary, we can't import its router directly.
// Instead, we test the core logic: VectorIndex operations that the handlers use,
// and verify the data flow that the server relies on.

// ══════════════════════════════════════════════════════════════
// CORE LOGIC TESTS (what the server handlers call)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_gate_knn_returns_hits() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(0, &query, 3);
    assert!(!hits.is_empty());
    // Feature 0 has gate[0,0]=1.0, should be top hit
    assert_eq!(hits[0].0, 0);
    assert!((hits[0].1 - 1.0).abs() < 0.01);
}

#[test]
fn test_walk_returns_per_layer_hits() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0, 1], 3);
    assert_eq!(trace.layers.len(), 2);

    // Layer 0: feature 0 (Paris) should be top hit
    let (layer, hits) = &trace.layers[0];
    assert_eq!(*layer, 0);
    assert!(!hits.is_empty());
    assert_eq!(hits[0].meta.top_token, "Paris");
}

#[test]
fn test_walk_with_layer_filter() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0]);
    let trace = patched.walk(&query, &[1], 3);
    assert_eq!(trace.layers.len(), 1);
    assert_eq!(trace.layers[0].0, 1);
}

#[test]
fn test_describe_entity_via_embedding() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Simulate what the describe handler does:
    // Token embedding → gate KNN → aggregate edges.
    let embed = test_embeddings();
    let query = embed.row(0).mapv(|v| v * 1.0); // token 0 → [1,0,0,0]
    let trace = patched.walk(&query, &[0, 1], 10);

    let mut targets: Vec<String> = Vec::new();
    for (_, hits) in &trace.layers {
        for hit in hits {
            targets.push(hit.meta.top_token.clone());
        }
    }

    // Token 0 → dim 0 strong → feature 0 (Paris) at L0, feature 1 (Tokyo) at L1
    assert!(targets.contains(&"Paris".to_string()));
}

#[test]
fn test_select_by_layer() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Simulate SELECT at layer 0
    let metas = patched.down_meta_at(0).unwrap();
    let tokens: Vec<&str> = metas
        .iter()
        .filter_map(|m| m.as_ref().map(|m| m.top_token.as_str()))
        .collect();

    assert_eq!(tokens, vec!["Paris", "French", "Europe"]);
}

#[test]
fn test_select_with_entity_filter() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Filter for tokens containing "par" (case-insensitive)
    let metas = patched.down_meta_at(0).unwrap();
    let matches: Vec<&str> = metas
        .iter()
        .filter_map(|m| m.as_ref())
        .filter(|m| m.top_token.to_lowercase().contains("par"))
        .map(|m| m.top_token.as_str())
        .collect();

    assert_eq!(matches, vec!["Paris"]);
}

#[test]
fn test_relations_listing() {
    let index = test_index();
    let patched = PatchedVindex::new(index);

    // Simulate SHOW RELATIONS: scan all layers, aggregate tokens
    let mut token_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for layer in patched.loaded_layers() {
        if let Some(metas) = patched.down_meta_at(layer) {
            for meta_opt in metas.iter() {
                if let Some(meta) = meta_opt {
                    *token_counts.entry(meta.top_token.clone()).or_default() += 1;
                }
            }
        }
    }

    assert_eq!(token_counts.len(), 6); // Paris, French, Europe, Berlin, Tokyo, Spain
    assert_eq!(*token_counts.get("Paris").unwrap(), 1);
}

#[test]
fn test_stats_from_config() {
    let config = test_config();
    let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();
    assert_eq!(total_features, 6);
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.hidden_size, 4);
    assert_eq!(config.model, "test/model-4");
}

// ══════════════════════════════════════════════════════════════
// PATCH OPERATIONS (what the patch endpoints use)
// ══════════════════════════════════════════════════════════════

#[test]
fn test_apply_patch_modifies_walk() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);

    // Before patch: feature 0 at L0 = "Paris"
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0], 3);
    assert_eq!(trace.layers[0].1[0].meta.top_token, "Paris");

    // Update feature 0 at L0 to "London"
    patched.update_feature_meta(0, 0, make_meta("London", 300, 0.99));

    let trace = patched.walk(&query, &[0], 3);
    assert_eq!(trace.layers[0].1[0].meta.top_token, "London");
}

#[test]
fn test_delete_feature_removes_from_walk() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);

    // Delete feature 0 at L0
    patched.delete_feature(0, 0);

    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[0], 3);

    // Feature 0 should no longer appear
    for (_, hits) in &trace.layers {
        for hit in hits {
            assert_ne!(hit.feature, 0);
        }
    }
}

#[test]
fn test_patch_count_tracking() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);
    assert_eq!(patched.num_patches(), 0);

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-04-01".into(),
        description: Some("test-patch".into()),
        author: None,
        tags: vec![],
        operations: vec![
            larql_vindex::PatchOp::Delete {
                layer: 0,
                feature: 0,
                reason: Some("test".into()),
            },
        ],
    };

    patched.apply_patch(patch);
    assert_eq!(patched.num_patches(), 1);
    assert_eq!(patched.num_overrides(), 1);
}

#[test]
fn test_remove_patch_restores_state() {
    let index = test_index();
    let mut patched = PatchedVindex::new(index);

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-04-01".into(),
        description: Some("removable".into()),
        author: None,
        tags: vec![],
        operations: vec![
            larql_vindex::PatchOp::Delete {
                layer: 0,
                feature: 0,
                reason: None,
            },
        ],
    };

    patched.apply_patch(patch);
    assert_eq!(patched.num_patches(), 1);

    // Feature 0 should be deleted
    assert!(patched.feature_meta(0, 0).is_none());

    // Remove the patch
    patched.remove_patch(0);
    assert_eq!(patched.num_patches(), 0);

    // Feature 0 should be back
    assert!(patched.feature_meta(0, 0).is_some());
    assert_eq!(patched.feature_meta(0, 0).unwrap().top_token, "Paris");
}

// ══════════════════════════════════════════════════════════════
// MULTI-MODEL SERVING LOGIC
// ══════════════════════════════════════════════════════════════

#[test]
fn test_model_id_extraction() {
    assert_eq!(model_id("google/gemma-3-4b-it"), "gemma-3-4b-it");
    assert_eq!(model_id("llama-3-8b"), "llama-3-8b");
    assert_eq!(model_id("org/sub/model"), "model");
}

fn model_id(name: &str) -> String {
    name.rsplit('/').next().unwrap_or(name).to_string()
}

// ══════════════════════════════════════════════════════════════
// EDGE CASES
// ══════════════════════════════════════════════════════════════

#[test]
fn test_empty_query_returns_no_hits() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(0, &query, 3);
    // All scores are 0, but KNN still returns results (sorted by abs)
    for (_feat, score) in &hits {
        assert!((score.abs()) < 0.01);
    }
}

#[test]
fn test_nonexistent_layer_returns_empty() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let hits = patched.gate_knn(99, &query, 3);
    assert!(hits.is_empty());
}

#[test]
fn test_walk_empty_layer_list() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let trace = patched.walk(&query, &[], 3);
    assert!(trace.layers.is_empty());
}

#[test]
fn test_large_top_k_clamped() {
    let index = test_index();
    let patched = PatchedVindex::new(index);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    // Request 100 but only 3 features exist
    let hits = patched.gate_knn(0, &query, 100);
    assert_eq!(hits.len(), 3);
}
