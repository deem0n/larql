//! GET /v1/relations — list all known relation types (top tokens).

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, Query, State};
use serde::Deserialize;

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

#[derive(Deserialize, Default)]
pub struct RelationsParams {
    #[serde(default)]
    pub source: Option<String>,
}

fn list_relations(
    model: &LoadedModel,
) -> Result<serde_json::Value, ServerError> {
    let start = std::time::Instant::now();

    let patched = model.patched.blocking_read();
    let all_layers = patched.loaded_layers();

    // Scan knowledge band layers (14-27 for Gemma, or use config).
    let config = &model.config;
    let last = config.num_layers.saturating_sub(1);
    let bands = config
        .layer_bands
        .clone()
        .or_else(|| larql_vindex::LayerBands::for_family(&config.family, config.num_layers))
        .unwrap_or(larql_vindex::LayerBands {
            syntax: (0, last),
            knowledge: (0, last),
            output: (0, last),
        });

    let scan_layers: Vec<usize> = all_layers
        .iter()
        .copied()
        .filter(|l| *l >= bands.knowledge.0 && *l <= bands.knowledge.1)
        .collect();

    struct TokenInfo {
        count: usize,
        max_score: f32,
        original: String,
        example: String,
    }

    let mut tokens: HashMap<String, TokenInfo> = HashMap::new();

    for &layer in &scan_layers {
        if let Some(metas) = patched.down_meta_at(layer) {
            for meta_opt in metas.iter() {
                if let Some(meta) = meta_opt {
                    let tok = meta.top_token.trim();
                    if tok.is_empty() || tok.len() < 2 {
                        continue;
                    }
                    if meta.c_score < 0.2 {
                        continue;
                    }
                    let key = tok.to_lowercase();
                    let example_tok = meta
                        .top_k
                        .first()
                        .map(|t| t.token.trim().to_string())
                        .unwrap_or_default();
                    let entry = tokens.entry(key).or_insert(TokenInfo {
                        count: 0,
                        max_score: 0.0,
                        original: tok.to_string(),
                        example: example_tok,
                    });
                    entry.count += 1;
                    if meta.c_score > entry.max_score {
                        entry.max_score = meta.c_score;
                    }
                }
            }
        }
    }

    let mut sorted: Vec<&TokenInfo> = tokens.values().collect();
    sorted.sort_by(|a, b| b.count.cmp(&a.count));
    sorted.truncate(50);

    let relations: Vec<serde_json::Value> = sorted
        .iter()
        .map(|info| {
            serde_json::json!({
                "name": info.original,
                "count": info.count,
                "example": info.example,
            })
        })
        .collect();

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(serde_json::json!({
        "relations": relations,
        "total": tokens.len(),
        "latency_ms": (latency_ms * 10.0).round() / 10.0,
    }))
}

pub async fn handle_relations(
    State(state): State<Arc<AppState>>,
    Query(_params): Query<RelationsParams>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);
    let result = tokio::task::spawn_blocking(move || list_relations(&model))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

pub async fn handle_relations_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    Query(_params): Query<RelationsParams>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(Some(&model_id))
        .ok_or_else(|| ServerError::NotFound(format!("model '{}' not found", model_id)))?;
    let model = Arc::clone(model);
    let result = tokio::task::spawn_blocking(move || list_relations(&model))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}
