// chaosnexus-crucible/src/main.rs
//! ChaosNexus Crucible: local LLM HTTP service with per-project session SSOT,
//! dual-scope skills/rules context packing, and GGUF model pull/load.

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

mod config;
mod context;
mod engine;
mod mcp_client;
mod model_store;
mod sessions;
mod stub_backend;

use config::CrucibleConfig;
use engine::candle_backend::CandleBackend;
use engine::colibri_backend::ColibriBackend;
use engine::{GenerationParams, InferenceBackend};
use rust_mcp_sdk::mcp_client::ClientRuntime;
use stub_backend::StubBackend;

#[derive(Serialize, Deserialize)]
struct GenerationRequest {
    prompt: String,
    #[serde(default)]
    project: Option<String>,
    anvil_port: Option<u16>,
    anvil_token: Option<String>,
    #[serde(flatten)]
    params: GenerationParams,
}

#[derive(Serialize)]
struct GenerationResponse {
    result: String,
}

#[derive(Deserialize)]
struct ProjectQuery {
    project: String,
}

#[derive(Deserialize)]
struct SessionIdQuery {
    project: String,
}

#[derive(Deserialize)]
struct ContextSearchQuery {
    project: String,
    #[serde(default)]
    q: String,
}

#[derive(Deserialize)]
struct ContextReadQuery {
    project: String,
    kind: String,
    name: String,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Clone)]
struct AppState {
    backend: Arc<RwLock<Box<dyn InferenceBackend>>>,
    mcp_client: Arc<RwLock<Option<Arc<ClientRuntime>>>>,
    active_port: Arc<RwLock<Option<u16>>>,
    config: Arc<CrucibleConfig>,
}

#[tokio::main]
async fn main() {
    let config = CrucibleConfig::load();
    println!(
        "Loading Crucible backend={} model_id={} models_dir={}",
        config.backend,
        config.resolved_model_id(),
        config.models_dir
    );

    let model_id = config.resolved_model_id().to_string();
    let mut backend: Box<dyn InferenceBackend> = match config.backend.as_str() {
        "colibri" => Box::new(ColibriBackend::new()),
        "stub" => Box::new(StubBackend {
            reason: "configured as stub".into(),
        }),
        _ => Box::new(CandleBackend::new().with_config(config.clone())),
    };

    if let Err(e) = backend.initialize(&model_id).await {
        eprintln!(
            "Failed to initialize backend ({e}); falling back to stub so sessions/context stay up."
        );
        backend = Box::new(StubBackend {
            reason: e.to_string(),
        });
        let _ = backend.initialize(&model_id).await;
    }

    let shared_state = AppState {
        backend: Arc::new(RwLock::new(backend)),
        mcp_client: Arc::new(RwLock::new(None)),
        active_port: Arc::new(RwLock::new(None)),
        config: Arc::new(config.clone()),
    };

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/generate", post(generate_handler))
        .route("/sessions", get(list_sessions_handler).post(create_session_handler))
        .route(
            "/sessions/{id}",
            get(get_session_handler)
                .put(update_session_handler)
                .delete(delete_session_handler),
        )
        .route("/context/list", get(context_list_handler))
        .route("/context/search", get(context_search_handler))
        .route("/context/read", get(context_read_handler))
        .route("/context/pack", get(context_pack_handler))
        .route("/models/status", get(models_status_handler))
        .route("/models/list", get(models_list_handler))
        .route("/models/pull", post(models_pull_handler))
        .with_state(shared_state);

    let bind_addr = format!("127.0.0.1:{}", config.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await.unwrap();
    println!(
        "Crucible LLM Server running on {}",
        listener.local_addr().unwrap()
    );

    axum::serve(listener, app).await.unwrap();
}

async fn health_handler() -> &'static str {
    "Crucible is operational"
}

async fn models_status_handler(State(state): State<AppState>) -> Json<model_store::ModelStatus> {
    Json(model_store::status_for(&state.config))
}

async fn models_list_handler(State(state): State<AppState>) -> Json<Vec<String>> {
    Json(model_store::list_cached(PathBuf::from(&state.config.models_dir).as_path()))
}

async fn models_pull_handler(
    State(state): State<AppState>,
    Json(body): Json<model_store::PullRequest>,
) -> Result<Json<model_store::ModelStatus>, (StatusCode, String)> {
    model_store::pull(&state.config, body)
        .await
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))
}

async fn generate_handler(
    State(state): State<AppState>,
    Json(payload): Json<GenerationRequest>,
) -> Result<Json<GenerationResponse>, (StatusCode, String)> {
    if let (Some(port), Some(token)) = (payload.anvil_port, payload.anvil_token.clone()) {
        let mut active_port = state.active_port.write().await;
        if *active_port != Some(port) {
            println!("Connecting to new Anvil instance on port {port}");
            match mcp_client::init_mcp_client(port, &token).await {
                Ok(client) => {
                    let mut mcp_client_lock = state.mcp_client.write().await;
                    *mcp_client_lock = Some(client.clone());
                    *active_port = Some(port);
                    println!("Successfully connected to Anvil MCP server.");

                    use rust_mcp_sdk::McpClient;
                    match client.request_tool_list(None).await {
                        Ok(tools) => {
                            println!("Discovered {} MCP tools from Anvil:", tools.tools.len());
                            for tool in tools.tools {
                                println!(
                                    " - {}: {}",
                                    tool.name,
                                    tool.description.unwrap_or_default()
                                );
                            }
                        }
                        Err(e) => eprintln!("Failed to fetch tools from Anvil: {e}"),
                    }
                }
                Err(e) => eprintln!("Failed to initialize MCP client to Anvil: {e}"),
            }
        }
    }

    let prompt = if let Some(ref project) = payload.project {
        let pack = context::pack_context(project);
        format!("{}\n---\nUser prompt:\n{}", pack.preamble, payload.prompt)
    } else {
        payload.prompt
    };

    let backend = state.backend.read().await;
    match backend.generate(&prompt, &payload.params).await {
        Ok(result) => Ok(Json(GenerationResponse { result })),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

async fn list_sessions_handler(
    Query(q): Query<ProjectQuery>,
) -> Result<Json<Vec<sessions::SessionSummary>>, (StatusCode, String)> {
    sessions::list_sessions(&q.project)
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))
}

async fn create_session_handler(
    Json(body): Json<sessions::CreateSessionRequest>,
) -> Result<Json<sessions::ChatSession>, (StatusCode, String)> {
    sessions::create_session(body)
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))
}

async fn get_session_handler(
    Path(id): Path<String>,
    Query(q): Query<SessionIdQuery>,
) -> Result<Json<sessions::ChatSession>, (StatusCode, String)> {
    sessions::get_session(&q.project, &id)
        .map(Json)
        .map_err(|e| (StatusCode::NOT_FOUND, e))
}

async fn update_session_handler(
    Path(id): Path<String>,
    Query(q): Query<SessionIdQuery>,
    Json(body): Json<sessions::UpdateSessionRequest>,
) -> Result<Json<sessions::ChatSession>, (StatusCode, String)> {
    sessions::update_session(&q.project, &id, body)
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))
}

async fn delete_session_handler(
    Path(id): Path<String>,
    Query(q): Query<SessionIdQuery>,
) -> Result<StatusCode, (StatusCode, String)> {
    sessions::delete_session(&q.project, &id)
        .map(|_| StatusCode::NO_CONTENT)
        .map_err(|e| (StatusCode::NOT_FOUND, e))
}

async fn context_list_handler(Query(q): Query<ProjectQuery>) -> Json<Vec<context::ContextEntry>> {
    Json(context::list_context(&q.project))
}

async fn context_search_handler(
    Query(q): Query<ContextSearchQuery>,
) -> Json<Vec<context::ContextEntry>> {
    Json(context::search_context(&q.project, &q.q))
}

async fn context_read_handler(
    Query(q): Query<ContextReadQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let offset = q.offset.unwrap_or(0);
    let limit = q.limit.unwrap_or(context::DEFAULT_CHUNK_LIMIT);
    let text = context::read_context(&q.project, &q.kind, &q.name, offset, limit)
        .map_err(|e| (StatusCode::NOT_FOUND, e))?;
    Ok(Json(serde_json::json!({
        "kind": q.kind,
        "name": q.name,
        "offset": offset,
        "limit": limit,
        "text": text,
    })))
}

async fn context_pack_handler(Query(q): Query<ProjectQuery>) -> Json<context::ContextPack> {
    Json(context::pack_context(&q.project))
}
