// chaosnexus-crucible/src/config.rs
//! Crucible runtime configuration (`crucible.toml` + env).

use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

/// Default Hub id for ChaosNexus Tuned v1 Q4_K_M GGUF (Crucible default download).
pub const DEFAULT_MODEL_ID: &str = "TunedChaos/ChaosNexus_Tuned_v1-GGUF";

#[derive(Debug, Deserialize, Clone)]
pub struct CrucibleConfig {
    #[serde(default = "default_backend")]
    pub backend: String,

    /// Hugging Face Hub model id (or local path under models_dir). Prefer GGUF repos.
    #[serde(default = "default_model_id")]
    pub model_id: String,

    /// Legacy alias for `model_id` (supervisor / older configs).
    #[serde(default)]
    pub model_path: Option<String>,

    /// Optional explicit `.gguf` filename inside the Hub repo / cache folder.
    #[serde(default)]
    pub gguf_file: Option<String>,

    /// Local cache root for downloaded models.
    #[serde(default = "default_models_dir")]
    pub models_dir: String,

    #[serde(default = "default_port")]
    pub port: u16,
}

fn default_backend() -> String {
    "candle".to_string()
}

fn default_model_id() -> String {
    DEFAULT_MODEL_ID.to_string()
}

fn default_port() -> u16 {
    8080
}

/// Resolve `~/.chaosnexus/crucible/models`.
pub fn default_models_dir() -> String {
    directories::UserDirs::new()
        .map(|u| {
            u.home_dir()
                .join(".chaosnexus")
                .join("crucible")
                .join("models")
                .to_string_lossy()
                .to_string()
        })
        .unwrap_or_else(|| ".chaosnexus/crucible/models".to_string())
}

impl CrucibleConfig {
    pub fn load() -> Self {
        let config_path = Path::new("crucible.toml");
        let mut cfg = if config_path.exists() {
            let config_str = fs::read_to_string(config_path).unwrap_or_default();
            toml::from_str(&config_str).unwrap_or_else(|_| Self::default())
        } else {
            Self::default()
        };

        // Legacy `model_path` maps onto `model_id` when present.
        if let Some(legacy) = cfg.model_path.clone() {
            if !legacy.trim().is_empty()
                && (cfg.model_id == DEFAULT_MODEL_ID || cfg.model_id == default_model_id())
            {
                // Only override default when the legacy field looks intentional.
                if legacy != "models/granite-4.1-8b" {
                    cfg.model_id = legacy;
                }
            } else if !legacy.trim().is_empty() && cfg.model_id.is_empty() {
                cfg.model_id = legacy;
            }
        }

        // Expand tilde in models_dir.
        if cfg.models_dir.starts_with("~/")
            && let Some(home) = directories::UserDirs::new() {
                cfg.models_dir = home
                    .home_dir()
                    .join(cfg.models_dir.trim_start_matches("~/"))
                    .to_string_lossy()
                    .to_string();
            }

        let _ = fs::create_dir_all(&cfg.models_dir);
        // Point hf-hub cache at our models tree when not already set.
        if std::env::var_os("HF_HOME").is_none() {
            let hf_home = PathBuf::from(&cfg.models_dir).join(".hf");
            let _ = fs::create_dir_all(&hf_home);
            // SAFETY: single-threaded at startup before Hub clients spawn.
            unsafe {
                std::env::set_var("HF_HOME", &hf_home);
            }
        }
        cfg
    }

    /// Effective Hub / local model identifier.
    pub fn resolved_model_id(&self) -> &str {
        self.model_id.as_str()
    }
}

impl Default for CrucibleConfig {
    fn default() -> Self {
        Self {
            backend: default_backend(),
            model_id: default_model_id(),
            model_path: None,
            gguf_file: None,
            models_dir: default_models_dir(),
            port: default_port(),
        }
    }
}
