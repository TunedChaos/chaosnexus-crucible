// chaosnexus-crucible/src/model_store.rs
//! Download and locate GGUF (and related) model files under `models_dir`.

use hf_hub::{api::tokio::ApiBuilder, Repo, RepoType};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::config::CrucibleConfig;

static DOWNLOAD_LOCK: Mutex<()> = Mutex::new(());
static DOWNLOADING: AtomicBool = AtomicBool::new(false);

/// Snapshot of model readiness for `/models/status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    pub model_id: String,
    pub models_dir: String,
    pub gguf_path: Option<String>,
    pub present: bool,
    pub downloading: bool,
    pub error: Option<String>,
}

/// Body for `POST /models/pull`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PullRequest {
    pub model_id: Option<String>,
    pub gguf_file: Option<String>,
}

fn api_builder() -> ApiBuilder {
    let mut b = ApiBuilder::new();
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.trim().is_empty() {
            b = b.with_token(Some(token));
        }
    } else if let Ok(token) = std::env::var("HUGGING_FACE_HUB_TOKEN")
        && !token.trim().is_empty() {
            b = b.with_token(Some(token));
        }
    b
}

/// Cache subdirectory for a Hub repo id (`org--name`).
pub fn cache_subdir(models_dir: &Path, model_id: &str) -> PathBuf {
    let safe = model_id.replace('/', "--");
    models_dir.join(safe)
}

/// Find an existing `.gguf` under the model cache (prefer `preferred` name).
pub fn find_local_gguf(dir: &Path, preferred: Option<&str>) -> Option<PathBuf> {
    if let Some(name) = preferred {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return None;
    };
    let mut ggufs: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
        .collect();
    ggufs.sort();
    ggufs.into_iter().next()
}

/// List cached model folder names under `models_dir`.
pub fn list_cached(models_dir: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(models_dir) else {
        return Vec::new();
    };
    let mut out: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| !n.starts_with('.'))
        .collect();
    out.sort();
    out
}

/// Ensure a Hub model's GGUF is present locally; download when missing.
pub async fn ensure_gguf(
    models_dir: &Path,
    model_id: &str,
    preferred_gguf: Option<&str>,
) -> Result<PathBuf, String> {
    let dir = cache_subdir(models_dir, model_id);
    fs::create_dir_all(&dir).map_err(|e| format!("mkdir cache: {e}"))?;

    if let Some(existing) = find_local_gguf(&dir, preferred_gguf) {
        return Ok(existing);
    }

    // Serialize downloads without holding a std Mutex across `.await`.
    {
        let _guard = DOWNLOAD_LOCK
            .lock()
            .map_err(|e| format!("download lock: {e}"))?;
        if let Some(existing) = find_local_gguf(&dir, preferred_gguf) {
            return Ok(existing);
        }
        if DOWNLOADING.swap(true, Ordering::SeqCst) {
            return Err("another download is already in progress".into());
        }
    }

    let result = async {
        println!("[model_store] Pulling {model_id} into {} ...", dir.display());
        let api = api_builder()
            .build()
            .map_err(|e| format!("hf-hub api: {e}"))?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let info = repo
            .info()
            .await
            .map_err(|e| format!("repo info {model_id}: {e}"))?;
        let mut remote_ggufs: Vec<String> = info
            .siblings
            .iter()
            .map(|s| s.rfilename.clone())
            .filter(|n| n.ends_with(".gguf"))
            .collect();
        remote_ggufs.sort();
        if remote_ggufs.is_empty() {
            return Err(format!(
                "Hub repo `{model_id}` has no .gguf files (publish Tuned GGUF first)"
            ));
        }

        let target_name = preferred_gguf
            .map(|s| s.to_string())
            .or_else(|| {
                remote_ggufs
                    .iter()
                    .find(|n| {
                        let l = n.to_lowercase();
                        l.contains("q4_k_m") || l.contains("q4km")
                    })
                    .cloned()
            })
            .unwrap_or_else(|| remote_ggufs[0].clone());

        println!("[model_store] Downloading {target_name} ...");
        let downloaded = repo
            .get(&target_name)
            .await
            .map_err(|e| format!("download {target_name}: {e}"))?;

        let dest = dir.join(
            Path::new(&target_name)
                .file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("model.gguf")),
        );
        if downloaded != dest {
            fs::copy(&downloaded, &dest).map_err(|e| format!("copy gguf: {e}"))?;
        }

        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ] {
            if info.siblings.iter().any(|s| s.rfilename == name)
                && let Ok(src) = repo.get(name).await {
                    let _ = fs::copy(&src, dir.join(name));
                }
        }

        Ok(dest)
    }
    .await;
    DOWNLOADING.store(false, Ordering::SeqCst);
    result
}

/// Status for the active config model.
pub fn status_for(config: &CrucibleConfig) -> ModelStatus {
    let model_id = config.resolved_model_id().to_string();
    let models_dir = PathBuf::from(&config.models_dir);
    let dir = cache_subdir(&models_dir, &model_id);
    let gguf = find_local_gguf(&dir, config.gguf_file.as_deref());
    ModelStatus {
        model_id,
        models_dir: config.models_dir.clone(),
        gguf_path: gguf.as_ref().map(|p| p.to_string_lossy().to_string()),
        present: gguf.is_some(),
        downloading: DOWNLOADING.load(Ordering::SeqCst),
        error: None,
    }
}

/// Pull using config defaults overridden by request fields.
pub async fn pull(config: &CrucibleConfig, req: PullRequest) -> Result<ModelStatus, String> {
    let model_id = req
        .model_id
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| config.resolved_model_id().to_string());
    let gguf_file = req.gguf_file.or_else(|| config.gguf_file.clone());
    let path = ensure_gguf(
        Path::new(&config.models_dir),
        &model_id,
        gguf_file.as_deref(),
    )
    .await?;
    Ok(ModelStatus {
        model_id,
        models_dir: config.models_dir.clone(),
        gguf_path: Some(path.to_string_lossy().to_string()),
        present: true,
        downloading: false,
        error: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn find_local_gguf_prefers_named() {
        let tmp = std::env::temp_dir().join(format!("cn-ms-{}", Uuid::new_v4()));
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join("a.gguf"), b"a").unwrap();
        fs::write(tmp.join("prefer.gguf"), b"b").unwrap();
        let found = find_local_gguf(&tmp, Some("prefer.gguf")).unwrap();
        assert!(found.ends_with("prefer.gguf"));
        let _ = fs::remove_dir_all(&tmp);
    }
}
