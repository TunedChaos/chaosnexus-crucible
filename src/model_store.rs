// chaosnexus-crucible/src/model_store.rs
//! Download and locate GGUF (and related) model files under `models_dir`.

use futures_util::StreamExt;
use hf_hub::repository::RepoTreeEntry;
use hf_hub::HFClient;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::config::CrucibleConfig;

static DOWNLOAD_LOCK: Mutex<()> = Mutex::new(());
static DOWNLOADING: AtomicBool = AtomicBool::new(false);

/// Default Path-2 download: ChaosNexus Tuned v1 Q4_K_M (product GGUF).
pub const TUNED_V1_MODEL_ID: &str = "TunedChaos/ChaosNexus_Tuned_v1-GGUF";
pub const TUNED_V1_GGUF_FILE: &str = "ChaosNexus_Tuned_v1-Q4_K_M.gguf";
/// Git LFS SHA-256 of the Tuned Q4_K_M GGUF blob (Hugging Face tree API `lfs.oid`).
pub const TUNED_V1_GGUF_SHA256: &str =
    "b77b27c27b4b3970697ab91a51387d4eb5f81b67b15451113a7b4ca06a98ed79";
pub const TUNED_V1_GGUF_BYTES: u64 = 5_347_914_496;
const TUNED_V1_HF_RESOLVE: &str = concat!(
    "https://huggingface.co/TunedChaos/ChaosNexus_Tuned_v1-GGUF/resolve/main/",
    "ChaosNexus_Tuned_v1-Q4_K_M.gguf"
);

/// Optional stock IBM Granite Guardian pin (Apache-2.0; not the product default).
pub const GRANITE_GUARDIAN_MODEL_ID: &str = "ibm-research/granite-guardian-3.2-5b-GGUF";
pub const GRANITE_GUARDIAN_GGUF_FILE: &str = "granite-guardian-3.2-5b-Q4_K_M.gguf";
pub const GRANITE_GUARDIAN_GGUF_SHA256: &str =
    "0e5e40d2a4531baf6ca30a7b48a308c1dcff9e305f16d3299ac2982d9ab4894e";
pub const GRANITE_GUARDIAN_GGUF_BYTES: u64 = 3_510_079_712;
const GRANITE_GUARDIAN_HF_RESOLVE: &str = concat!(
    "https://huggingface.co/ibm-research/granite-guardian-3.2-5b-GGUF/resolve/main/",
    "granite-guardian-3.2-5b-Q4_K_M.gguf"
);


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

/// Split a Hub repo id (`owner/name`) into owner and name components.
fn split_repo_id(model_id: &str) -> Result<(&str, &str), String> {
    let (owner, name) = model_id
        .split_once('/')
        .ok_or_else(|| format!("Hub model id `{model_id}` must be `owner/name`"))?;
    if owner.is_empty() || name.is_empty() {
        return Err(format!("Hub model id `{model_id}` must be `owner/name`"));
    }
    Ok((owner, name))
}

/// Build an authenticated (when token present) Hugging Face Hub client.
fn hf_client() -> Result<HFClient, String> {
    let mut builder = HFClient::builder();
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.trim().is_empty() {
            builder = builder.token(token);
        }
    } else if let Ok(token) = std::env::var("HUGGING_FACE_HUB_TOKEN")
        && !token.trim().is_empty()
    {
        builder = builder.token(token);
    }
    builder.build().map_err(|e| format!("hf-hub client: {e}"))
}

/// Cache subdirectory for a Hub repo id (`org--name`).
pub fn cache_subdir(models_dir: &Path, model_id: &str) -> PathBuf {
    let safe = model_id.replace('/', "--");
    models_dir.join(safe)
}

/// True when `path` is a usable GGUF (min size and optional SHA-256).
pub fn gguf_file_ready(
    path: &Path,
    min_bytes: Option<u64>,
    expected_sha256: Option<&str>,
) -> bool {
    let Ok(meta) = fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    if let Some(min) = min_bytes {
        if meta.len() < min {
            return false;
        }
    }
    if let Some(want) = expected_sha256 {
        match sha256_file(path) {
            Ok(got) if got.eq_ignore_ascii_case(want) => {}
            _ => return false,
        }
    }
    true
}

fn sha256_file(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = fs::File::open(path).map_err(|e| format!("open for sha256: {e}"))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buf).map_err(|e| format!("read for sha256: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn pinned_gguf_specs(gguf_name: &str) -> (Option<u64>, Option<&'static str>) {
    if gguf_name == TUNED_V1_GGUF_FILE || gguf_name.ends_with(TUNED_V1_GGUF_FILE) {
        (
            Some(TUNED_V1_GGUF_BYTES.saturating_sub(1024 * 1024)),
            Some(TUNED_V1_GGUF_SHA256),
        )
    } else if gguf_name == GRANITE_GUARDIAN_GGUF_FILE
        || gguf_name.ends_with(GRANITE_GUARDIAN_GGUF_FILE)
    {
        (
            Some(GRANITE_GUARDIAN_GGUF_BYTES.saturating_sub(1024 * 1024)),
            Some(GRANITE_GUARDIAN_GGUF_SHA256),
        )
    } else {
        (None, None)
    }
}

fn verify_specs_for(model_id: &str, gguf_name: &str) -> (Option<u64>, Option<&'static str>) {
    let (min, sha) = pinned_gguf_specs(gguf_name);
    if min.is_some() {
        return (min, sha);
    }
    let _ = model_id;
    (None, None)
}

/// Find an existing `.gguf` under the model cache (prefer `preferred` name).
/// Incomplete / corrupt downloads (too small or wrong SHA) are treated as missing.
pub fn find_local_gguf(dir: &Path, preferred: Option<&str>) -> Option<PathBuf> {
    if let Some(name) = preferred {
        let p = dir.join(name);
        let (pin_min, sha) = pinned_gguf_specs(name);
        let min = pin_min.or(Some(1024 * 1024));
        if gguf_file_ready(&p, min, sha) {
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
        .filter(|p| {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let (pin_min, sha) = pinned_gguf_specs(name);
            let min = pin_min.or(Some(1024 * 1024));
            gguf_file_ready(p, min, sha)
        })
        .collect();
    ggufs.sort();
    ggufs.into_iter().next()
}

fn resolve_download_url(model_id: &str, filename: &str) -> String {
    if let Ok(base) = std::env::var("CHAOSNEXUS_MODEL_BASE_URL") {
        let base = base.trim_end_matches('/');
        if !base.is_empty() {
            return format!("{base}/{filename}");
        }
    }
    if model_id == TUNED_V1_MODEL_ID && filename == TUNED_V1_GGUF_FILE {
        return TUNED_V1_HF_RESOLVE.to_string();
    }
    if model_id == GRANITE_GUARDIAN_MODEL_ID && filename == GRANITE_GUARDIAN_GGUF_FILE {
        return GRANITE_GUARDIAN_HF_RESOLVE.to_string();
    }
    format!("https://huggingface.co/{model_id}/resolve/main/{filename}")
}

/// HTTP(S) download with optional Range resume into `dest.partial`, then rename.
async fn download_url_resumable(
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
    expected_bytes: Option<u64>,
) -> Result<(), String> {
    use futures_util::StreamExt;
    use sha2::{Digest, Sha256};
    use std::io::Write;

    let partial = dest.with_extension("gguf.partial");
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("http client: {e}"))?;

    let mut existing: u64 = 0;
    if partial.is_file() {
        existing = fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);
    }

    let mut hasher = Sha256::new();
    if existing > 0 {
        use std::io::Read;
        let mut prior = fs::File::open(&partial).map_err(|e| format!("open partial hash: {e}"))?;
        let mut buf = [0u8; 1024 * 1024];
        loop {
            let n = prior.read(&mut buf).map_err(|e| format!("hash prior: {e}"))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
    }

    let mut req = client.get(url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.trim().is_empty() {
            req = req.bearer_auth(token.trim());
        }
    }
    if existing > 0 {
        req = req.header(reqwest::header::RANGE, format!("bytes={existing}-"));
    }

    println!("[model_store] HTTP GET {url} (resume_from={existing})");
    let res = req.send().await.map_err(|e| format!("download GET: {e}"))?;
    let status = res.status();
    if !(status.is_success() || status.as_u16() == 206) {
        return Err(format!("download HTTP {status}"));
    }

    // 200 after a Range request means server ignored resume - restart file.
    let append = status.as_u16() == 206;
    if !append {
        existing = 0;
        hasher = Sha256::new();
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append)
        .open(&partial)
        .map_err(|e| format!("open partial: {e}"))?;

    let mut stream = res.bytes_stream();
    let mut written = existing;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("download stream: {e}"))?;
        file.write_all(&chunk)
            .map_err(|e| format!("write partial: {e}"))?;
        hasher.update(&chunk);
        written += chunk.len() as u64;
        if written % (64 * 1024 * 1024) < chunk.len() as u64 {
            println!("[model_store] downloaded {written} bytes…");
        }
    }
    file.flush().map_err(|e| format!("flush partial: {e}"))?;

    if let Some(exp) = expected_bytes {
        if written < exp.saturating_sub(1024 * 1024) {
            return Err(format!(
                "download incomplete: got {written} bytes, expected ~{exp}"
            ));
        }
    }

    let got = hex::encode(hasher.finalize());
    if let Some(want) = expected_sha256 {
        if !got.eq_ignore_ascii_case(want) {
            let _ = fs::remove_file(&partial);
            return Err(format!(
                "SHA-256 mismatch for {}: got {got}, expected {want}",
                dest.display()
            ));
        }
    }

    fs::rename(&partial, dest).map_err(|e| format!("rename partial→final: {e}"))?;
    println!(
        "[model_store] Saved {} ({written} bytes, sha256={got})",
        dest.display()
    );
    Ok(())
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
        let (owner, name) = split_repo_id(model_id)?;
        let client = hf_client()?;
        let repo = client.model(owner, name);

        let tree = repo
            .list_tree()
            .recursive(true)
            .send()
            .map_err(|e| format!("repo tree {model_id}: {e}"))?;
        let mut tree = std::pin::pin!(tree);
        let mut remote_files: Vec<String> = Vec::new();
        while let Some(entry) = tree.as_mut().next().await {
            let entry = entry.map_err(|e| format!("repo tree entry {model_id}: {e}"))?;
            if let RepoTreeEntry::File { path, .. } = entry {
                remote_files.push(path);
            }
        }

        let mut remote_ggufs: Vec<String> = remote_files
            .iter()
            .filter(|n| n.ends_with(".gguf"))
            .cloned()
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

        let dest = dir.join(
            Path::new(&target_name)
                .file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("model.gguf")),
        );
        let (min_b, sha) = verify_specs_for(model_id, &target_name);
        let expected_bytes = if target_name == TUNED_V1_GGUF_FILE
            || target_name.ends_with(TUNED_V1_GGUF_FILE)
        {
            Some(TUNED_V1_GGUF_BYTES)
        } else if target_name == GRANITE_GUARDIAN_GGUF_FILE
            || target_name.ends_with(GRANITE_GUARDIAN_GGUF_FILE)
        {
            Some(GRANITE_GUARDIAN_GGUF_BYTES)
        } else {
            min_b.map(|m| m + 1024 * 1024)
        };

        // Prefer direct HTTP(S) with Range resume (supports CHAOSNEXUS_MODEL_BASE_URL).
        let url = resolve_download_url(model_id, &target_name);
        println!("[model_store] Downloading {target_name} via {url} ...");
        match download_url_resumable(&url, &dest, sha, expected_bytes).await {
            Ok(()) => {}
            Err(direct_err) => {
                eprintln!(
                    "[model_store] Direct download failed ({direct_err}); falling back to hf-hub"
                );
                let downloaded = repo
                    .download_file()
                    .filename(target_name.clone())
                    .local_dir(dir.clone())
                    .send()
                    .await
                    .map_err(|e| {
                        format!("download {target_name}: direct={direct_err}; hf-hub={e}")
                    })?;
                if downloaded != dest {
                    fs::copy(&downloaded, &dest).map_err(|e| format!("copy gguf: {e}"))?;
                }
                if let Some(want) = sha {
                    let got = sha256_file(&dest)?;
                    if !got.eq_ignore_ascii_case(want) {
                        let _ = fs::remove_file(&dest);
                        return Err(format!(
                            "SHA-256 mismatch after hf-hub download: got {got}, expected {want}"
                        ));
                    }
                }
            }
        }

        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ] {
            if remote_files.iter().any(|p| p == name)
                && let Ok(src) = repo
                    .download_file()
                    .filename(name.to_string())
                    .local_dir(dir.clone())
                    .send()
                    .await
            {
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

    #[test]
    fn split_repo_id_requires_owner_name() {
        assert_eq!(split_repo_id("org/model").unwrap(), ("org", "model"));
        assert!(split_repo_id("noslash").is_err());
        assert!(split_repo_id("/name").is_err());
        assert!(split_repo_id("org/").is_err());
    }

    #[test]
    fn cache_subdir_flattens_slash() {
        let dir = PathBuf::from("/models");
        assert_eq!(
            cache_subdir(&dir, "TunedChaos/ChaosNexus_Tuned_v1"),
            PathBuf::from("/models/TunedChaos--ChaosNexus_Tuned_v1")
        );
    }

    #[test]
    fn list_cached_skips_dotdirs_and_files() {
        let tmp = std::env::temp_dir().join(format!("cn-ms-list-{}", Uuid::new_v4()));
        fs::create_dir_all(tmp.join("keep-me")).unwrap();
        fs::create_dir_all(tmp.join(".hidden")).unwrap();
        fs::write(tmp.join("file.txt"), b"x").unwrap();
        let listed = list_cached(&tmp);
        assert_eq!(listed, vec!["keep-me".to_string()]);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn find_local_gguf_falls_back_to_sorted_first() {
        let tmp = std::env::temp_dir().join(format!("cn-ms-fb-{}", Uuid::new_v4()));
        fs::create_dir_all(&tmp).unwrap();
        fs::write(tmp.join("z.gguf"), b"z").unwrap();
        fs::write(tmp.join("a.gguf"), b"a").unwrap();
        let found = find_local_gguf(&tmp, None).unwrap();
        assert!(found.ends_with("a.gguf"));
        let _ = fs::remove_dir_all(&tmp);
    }
}
