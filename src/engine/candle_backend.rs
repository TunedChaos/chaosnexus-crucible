// chaosnexus-crucible/src/engine/candle_backend.rs
//! Candle GGUF inference backend (default: ChaosNexus Tuned v1 GGUF).

use async_trait::async_trait;
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights as LlamaModelWeights;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

use super::quantized_granite::{is_granite_gguf, GraniteModelWeights};
use super::{EngineError, GenerationParams, InferenceBackend};
use crate::config::CrucibleConfig;
use crate::model_store;

/// Loaded Candle weights (Llama-compatible GGUF or Granite GGUF).
enum LoadedWeights {
    Llama(LlamaModelWeights),
    Granite(GraniteModelWeights),
}

impl LoadedWeights {
    fn clear_kv_cache(&mut self) {
        match self {
            Self::Llama(m) => m.clear_kv_cache(),
            Self::Granite(m) => m.clear_kv_cache(),
        }
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor, candle_core::Error> {
        match self {
            Self::Llama(m) => m.forward(input, index_pos),
            Self::Granite(m) => m.forward(input, index_pos),
        }
    }
}

pub struct CandleBackend {
    device: Device,
    model: Option<Arc<Mutex<LoadedWeights>>>,
    tokenizer: Option<Tokenizer>,
    /// Retained config for ensure/pull during initialize.
    config: Option<CrucibleConfig>,
}

impl CandleBackend {
    pub fn new() -> Self {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        Self {
            device,
            model: None,
            tokenizer: None,
            config: None,
        }
    }

    /// Attach full Crucible config (models_dir, gguf_file, model_id).
    pub fn with_config(mut self, config: CrucibleConfig) -> Self {
        self.config = Some(config);
        self
    }

    fn load_tokenizer(gguf_path: &Path) -> Result<Tokenizer, EngineError> {
        let dir = gguf_path.parent().unwrap_or_else(|| Path::new("."));
        let candidates = [
            dir.join("tokenizer.json"),
            gguf_path.with_extension("tokenizer.json"),
        ];
        for c in &candidates {
            if c.is_file() {
                return Tokenizer::from_file(c)
                    .map_err(|e| EngineError::InitializationFailed(e.to_string()));
            }
        }
        Err(EngineError::InitializationFailed(
            "tokenizer.json not found beside GGUF (re-pull model or add tokenizer.json)".into(),
        ))
    }

    /// Prefer Granite / Llama chat EOS ids when present in the tokenizer vocab.
    fn eos_token_ids(tokenizer: &Tokenizer) -> Vec<u32> {
        let vocab = tokenizer.get_vocab(true);
        ["<|end_of_text|>", "</s>", "<eos>", "<|eot_id|>"]
            .iter()
            .filter_map(|t| vocab.get(*t).copied())
            .collect()
    }
}

#[async_trait]
impl InferenceBackend for CandleBackend {
    async fn initialize(&mut self, model_path: &str) -> Result<(), EngineError> {
        println!(
            "Initializing Candle GGUF backend on {:?} (model={})...",
            self.device, model_path
        );

        let cfg = self.config.clone().unwrap_or_default();
        let model_id = if model_path.trim().is_empty() {
            cfg.resolved_model_id().to_string()
        } else {
            model_path.to_string()
        };

        let gguf_path = if Path::new(&model_id).extension().and_then(|e| e.to_str()) == Some("gguf")
            && Path::new(&model_id).is_file()
        {
            Path::new(&model_id).to_path_buf()
        } else {
            model_store::ensure_gguf(
                Path::new(&cfg.models_dir),
                &model_id,
                cfg.gguf_file.as_deref(),
            )
            .await
            .map_err(EngineError::InitializationFailed)?
        };

        println!("Loading GGUF: {}", gguf_path.display());
        let mut file = File::open(&gguf_path).map_err(|e| {
            EngineError::InitializationFailed(format!("open {}: {e}", gguf_path.display()))
        })?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| EngineError::InitializationFailed(format!("gguf read: {e}")))?;

        let weights = if is_granite_gguf(&content) {
            println!("Detected Granite GGUF — using granite.* metadata + scales");
            LoadedWeights::Granite(
                GraniteModelWeights::from_gguf(content, &mut file, &self.device).map_err(|e| {
                    EngineError::InitializationFailed(format!("granite gguf load: {e}"))
                })?,
            )
        } else {
            LoadedWeights::Llama(
                LlamaModelWeights::from_gguf(content, &mut file, &self.device).map_err(|e| {
                    EngineError::InitializationFailed(format!("llama gguf load: {e}"))
                })?,
            )
        };

        let tokenizer = Self::load_tokenizer(&gguf_path)?;
        self.model = Some(Arc::new(Mutex::new(weights)));
        self.tokenizer = Some(tokenizer);
        Ok(())
    }

    async fn generate(&self, prompt: &str, params: &GenerationParams) -> Result<String, EngineError> {
        let Some(model) = &self.model else {
            return Err(EngineError::GenerationFailed(
                "Model not loaded yet".into(),
            ));
        };
        let Some(tokenizer) = &self.tokenizer else {
            return Err(EngineError::GenerationFailed(
                "Tokenizer not loaded yet".into(),
            ));
        };

        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| EngineError::GenerationFailed(e.to_string()))?;
        let tokens: Vec<u32> = encoding.get_ids().to_vec();
        let max_new = params.max_new_tokens.unwrap_or(128);
        let temperature = params.temperature.unwrap_or(0.7) as f64;
        let mut logits_processor = LogitsProcessor::new(299792458, Some(temperature), None);
        let eos_ids = Self::eos_token_ids(tokenizer);

        let mut model = model
            .lock()
            .map_err(|e| EngineError::GenerationFailed(e.to_string()))?;
        // Fresh KV cache per generate call (single-shot HTTP API).
        model.clear_kv_cache();

        let mut all_tokens = tokens.clone();
        let mut next_token = {
            let input = Tensor::new(tokens.as_slice(), &self.device)
                .map_err(|e| EngineError::GenerationFailed(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| EngineError::GenerationFailed(e.to_string()))?;
            let logits = model
                .forward(&input, 0)
                .map_err(|e| EngineError::GenerationFailed(e.to_string()))?;
            let logits = logits
                .squeeze(0)
                .map_err(|e| EngineError::GenerationFailed(e.to_string()))?;
            logits_processor
                .sample(&logits)
                .map_err(|e| EngineError::GenerationFailed(e.to_string()))?
        };
        all_tokens.push(next_token);

        for index in 0..max_new.saturating_sub(1) {
            if eos_ids.contains(&next_token) {
                break;
            }
            let input = Tensor::new(&[next_token], &self.device)
                .map_err(|e| EngineError::GenerationFailed(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| EngineError::GenerationFailed(e.to_string()))?;
            let logits = model
                .forward(&input, tokens.len() + index)
                .map_err(|e| EngineError::GenerationFailed(e.to_string()))?;
            let logits = logits
                .squeeze(0)
                .map_err(|e| EngineError::GenerationFailed(e.to_string()))?;
            next_token = logits_processor
                .sample(&logits)
                .map_err(|e| EngineError::GenerationFailed(e.to_string()))?;
            all_tokens.push(next_token);
        }

        let decoded = tokenizer
            .decode(&all_tokens[tokens.len()..], true)
            .map_err(|e| EngineError::GenerationFailed(e.to_string()))?;
        Ok(decoded)
    }
}
