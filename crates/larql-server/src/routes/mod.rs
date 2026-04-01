//! Router setup — maps URL paths to handlers.

pub mod describe;
pub mod health;
pub mod models;
pub mod patches;
pub mod relations;
pub mod select;
pub mod stats;
pub mod walk;

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post, delete};

use crate::state::AppState;

/// Build the router for single-model serving.
pub fn single_model_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/describe", get(describe::handle_describe))
        .route("/v1/walk", get(walk::handle_walk))
        .route("/v1/select", post(select::handle_select))
        .route("/v1/relations", get(relations::handle_relations))
        .route("/v1/stats", get(stats::handle_stats))
        .route("/v1/patches/apply", post(patches::handle_apply_patch))
        .route("/v1/patches", get(patches::handle_list_patches))
        .route("/v1/patches/{name}", delete(patches::handle_remove_patch))
        .route("/v1/health", get(health::handle_health))
        .route("/v1/models", get(models::handle_models))
        .with_state(state)
}

/// Build the router for multi-model serving.
pub fn multi_model_router(state: Arc<AppState>) -> Router {
    let mut router = Router::new()
        .route("/v1/health", get(health::handle_health))
        .route("/v1/models", get(models::handle_models));

    // Per-model routes: /v1/{model_id}/describe, etc.
    router = router
        .route("/v1/{model_id}/describe", get(describe::handle_describe_multi))
        .route("/v1/{model_id}/walk", get(walk::handle_walk_multi))
        .route("/v1/{model_id}/select", post(select::handle_select_multi))
        .route("/v1/{model_id}/relations", get(relations::handle_relations_multi))
        .route("/v1/{model_id}/stats", get(stats::handle_stats_multi))
        .route("/v1/{model_id}/patches/apply", post(patches::handle_apply_patch_multi))
        .route("/v1/{model_id}/patches", get(patches::handle_list_patches_multi))
        .route("/v1/{model_id}/patches/{name}", delete(patches::handle_remove_patch_multi));

    router.with_state(state)
}
