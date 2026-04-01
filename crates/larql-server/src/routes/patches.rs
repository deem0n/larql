//! POST/GET/DELETE /v1/patches — patch management endpoints.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use serde::Deserialize;

use crate::error::ServerError;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct ApplyPatchRequest {
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub patch: Option<larql_vindex::VindexPatch>,
}

async fn apply_patch_to_model(
    state: &AppState,
    model_id: Option<&str>,
    req: ApplyPatchRequest,
) -> Result<Json<serde_json::Value>, ServerError> {
    let model = state
        .model(model_id)
        .ok_or_else(|| ServerError::NotFound("model not found".into()))?;

    let patch = if let Some(patch) = req.patch {
        patch
    } else if let Some(url) = &req.url {
        // Load patch from URL or HuggingFace path.
        let path = if larql_vindex::is_hf_path(url) {
            let resolved = larql_vindex::resolve_hf_vindex(url)
                .map_err(|e| ServerError::Internal(format!("failed to resolve HF path: {e}")))?;
            // Look for .vlp files in the resolved directory
            let vlp_path = resolved.join("patch.vlp");
            if vlp_path.exists() {
                vlp_path
            } else {
                return Err(ServerError::BadRequest(format!(
                    "no patch.vlp found at {url}"
                )));
            }
        } else {
            std::path::PathBuf::from(url)
        };
        larql_vindex::VindexPatch::load(&path)
            .map_err(|e| ServerError::Internal(format!("failed to load patch: {e}")))?
    } else {
        return Err(ServerError::BadRequest(
            "must provide 'url' or 'patch' in request body".into(),
        ));
    };

    let name = req
        .url
        .clone()
        .or_else(|| patch.description.clone())
        .unwrap_or_else(|| "inline-patch".into());
    let op_count = patch.operations.len();

    let mut patched = model.patched.write().await;
    patched.apply_patch(patch);
    let active = patched.num_patches();

    Ok(Json(serde_json::json!({
        "applied": name,
        "operations": op_count,
        "active_patches": active,
    })))
}

pub async fn handle_apply_patch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ApplyPatchRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    apply_patch_to_model(&state, None, req).await
}

pub async fn handle_apply_patch_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    Json(req): Json<ApplyPatchRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    apply_patch_to_model(&state, Some(&model_id), req).await
}

async fn list_patches_for_model(
    state: &AppState,
    model_id: Option<&str>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let model = state
        .model(model_id)
        .ok_or_else(|| ServerError::NotFound("model not found".into()))?;

    let patched = model.patched.read().await;
    let patches: Vec<serde_json::Value> = patched
        .patches
        .iter()
        .map(|p| {
            serde_json::json!({
                "name": p.description.as_deref().unwrap_or("unnamed"),
                "operations": p.operations.len(),
                "base_model": p.base_model,
            })
        })
        .collect();

    Ok(Json(serde_json::json!({ "patches": patches })))
}

pub async fn handle_list_patches(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    list_patches_for_model(&state, None).await
}

pub async fn handle_list_patches_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    list_patches_for_model(&state, Some(&model_id)).await
}

async fn remove_patch_from_model(
    state: &AppState,
    model_id: Option<&str>,
    name: &str,
) -> Result<Json<serde_json::Value>, ServerError> {
    let model = state
        .model(model_id)
        .ok_or_else(|| ServerError::NotFound("model not found".into()))?;

    let mut patched = model.patched.write().await;

    let idx = patched
        .patches
        .iter()
        .position(|p| {
            p.description.as_deref().unwrap_or("unnamed") == name
        })
        .ok_or_else(|| ServerError::NotFound(format!("patch '{}' not found", name)))?;

    patched.remove_patch(idx);

    Ok(Json(serde_json::json!({
        "removed": name,
        "active_patches": patched.num_patches(),
    })))
}

pub async fn handle_remove_patch(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    remove_patch_from_model(&state, None, &name).await
}

pub async fn handle_remove_patch_multi(
    State(state): State<Arc<AppState>>,
    Path((model_id, name)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    remove_patch_from_model(&state, Some(&model_id), &name).await
}
