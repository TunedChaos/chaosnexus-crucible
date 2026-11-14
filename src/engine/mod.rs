use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_new_tokens: Some(128),
            temperature: Some(0.7),
            top_p: Some(0.95),
            top_k: Some(40),
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub enum EngineError {
    InitializationFailed(String),
    GenerationFailed(String),
    NotImplemented,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::InitializationFailed(msg) => write!(f, "Engine Initialization Failed: {}", msg),
            EngineError::GenerationFailed(msg) => write!(f, "Engine Generation Failed: {}", msg),
            EngineError::NotImplemented => write!(f, "Feature Not Implemented for this backend"),
        }
    }
}

impl std::error::Error for EngineError {}

#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Initialize the backend (load model, weights, allocate buffers)
    async fn initialize(&mut self, model_path: &str) -> Result<(), EngineError>;
    
    /// Generate a complete response
    async fn generate(&self, prompt: &str, params: &GenerationParams) -> Result<String, EngineError>;
    
    // In the future:
    // async fn generate_stream(&self, prompt: &str, params: &GenerationParams) -> Result<BoxStream<String>, EngineError>;
}

pub mod candle_backend;
pub mod colibri_backend;
pub mod quantized_granite;
