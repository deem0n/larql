//! larql-server — HTTP server for vindex knowledge queries.

mod error;
mod routes;
mod state;

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use tokio::sync::RwLock;
use tracing::{info, warn};

use larql_vindex::{
    PatchedVindex, SilentLoadCallbacks, VectorIndex,
    load_vindex_config, load_vindex_embeddings, load_vindex_tokenizer,
};

use state::{AppState, LoadedModel, model_id_from_name};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[derive(Parser)]
#[command(
    name = "larql-server",
    version,
    about = "HTTP server for vindex knowledge queries and inference"
)]
struct Cli {
    /// Path to a .vindex directory (or hf:// path).
    #[arg(value_name = "VINDEX_PATH")]
    vindex_path: Option<String>,

    /// Serve all .vindex directories in this folder.
    #[arg(long)]
    dir: Option<PathBuf>,

    /// Listen port.
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Disable INFER endpoint (browse-only, reduces memory).
    #[arg(long)]
    no_infer: bool,

    /// Enable CORS for browser access.
    #[arg(long)]
    cors: bool,

    /// Logging level.
    #[arg(long, default_value = "info")]
    log_level: String,
}

fn load_single_vindex(path_str: &str, no_infer: bool) -> Result<LoadedModel, BoxError> {
    let path = if larql_vindex::is_hf_path(path_str) {
        info!("Resolving HuggingFace path: {}", path_str);
        larql_vindex::resolve_hf_vindex(path_str)?
    } else {
        PathBuf::from(path_str)
    };

    info!("Loading: {}", path.display());

    let config = load_vindex_config(&path)?;
    let model_name = config.model.clone();
    let id = model_id_from_name(&model_name);

    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&path, &mut cb)?;
    let total_features: usize = config.layers.iter().map(|l| l.num_features).sum();

    info!(
        "  Model: {} ({} layers, {} features)",
        model_name, config.num_layers, total_features
    );

    let (embeddings, embed_scale) = load_vindex_embeddings(&path)?;
    info!("  Embeddings: {}x{}", embeddings.shape()[0], embeddings.shape()[1]);

    let tokenizer = load_vindex_tokenizer(&path)?;

    let patched = PatchedVindex::new(index);

    Ok(LoadedModel {
        id,
        path,
        config,
        patched: RwLock::new(patched),
        embeddings,
        embed_scale,
        tokenizer,
        infer_disabled: no_infer,
    })
}

fn discover_vindexes(dir: &PathBuf) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() && p.join("index.json").exists() {
                paths.push(p);
            }
        }
    }
    paths.sort();
    paths
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level)),
        )
        .init();

    info!("larql-server v{}", env!("CARGO_PKG_VERSION"));

    let mut models: Vec<Arc<LoadedModel>> = Vec::new();

    if let Some(ref dir) = cli.dir {
        let paths = discover_vindexes(dir);
        if paths.is_empty() {
            return Err(format!("no .vindex directories found in {}", dir.display()).into());
        }
        info!("Found {} vindexes in {}", paths.len(), dir.display());
        for p in &paths {
            match load_single_vindex(&p.to_string_lossy(), cli.no_infer) {
                Ok(m) => models.push(Arc::new(m)),
                Err(e) => warn!("  Skipping {}: {}", p.display(), e),
            }
        }
    } else if let Some(ref vindex_path) = cli.vindex_path {
        let m = load_single_vindex(vindex_path, cli.no_infer)?;
        models.push(Arc::new(m));
    } else {
        return Err("must provide a vindex path or --dir".into());
    }

    if models.is_empty() {
        return Err("no vindexes loaded".into());
    }

    let state = Arc::new(AppState {
        models: models.clone(),
        started_at: std::time::Instant::now(),
        requests_served: std::sync::atomic::AtomicU64::new(0),
    });

    let is_multi = state.is_multi_model();
    let mut app = if is_multi {
        info!("Multi-model mode ({} models)", state.models.len());
        for m in &state.models {
            info!("  /v1/{}/...", m.id);
        }
        routes::multi_model_router(state)
    } else {
        let m = &models[0];
        info!("Single-model mode: {}", m.config.model);
        routes::single_model_router(state)
    };

    if cli.cors {
        use tower_http::cors::CorsLayer;
        app = app.layer(CorsLayer::permissive());
        info!("CORS: enabled");
    }

    app = app.layer(tower_http::trace::TraceLayer::new_for_http());

    let addr = format!("{}:{}", cli.host, cli.port);
    info!("Listening: http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
