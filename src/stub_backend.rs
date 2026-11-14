// chaosnexus-crucible/src/stub_backend.rs
//! Fallback inference backend used when the configured model cannot load.
//! Keeps `/health` and session/context HTTP APIs available for Forge supervision.

use async_trait::async_trait;

use crate::engine::{EngineError, GenerationParams, InferenceBackend};

/// Echo/stub backend that always fails generate with a clear message.
pub struct StubBackend {
    /// Why the real backend was not used.
    pub reason: String,
}

#[async_trait]
impl InferenceBackend for StubBackend {
    async fn initialize(&mut self, _model_path: &str) -> Result<(), EngineError> {
        Ok(())
    }

    async fn generate(
        &self,
        _prompt: &str,
        _params: &GenerationParams,
    ) -> Result<String, EngineError> {
        Err(EngineError::GenerationFailed(format!(
            "Crucible model unavailable: {}. Sessions and context APIs still work.",
            self.reason
        )))
    }
}
