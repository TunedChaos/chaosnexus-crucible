// chaosnexus-crucible/src/sessions.rs
//
//! Per-project chat session store (SSOT for Forge and future crucible-cli).
//!
//! Sessions live under `<project>/.chaosnexus/crucible/sessions/<id>.json`.
//! Paths are normalized to absolute form so list queries stay consistent.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// A single chat message persisted with a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    /// Message author role (`user`, `agent`, `assistant`, `system`, `error`).
    pub role: String,
    /// Message body text.
    pub text: String,
    /// Optional diagnostic / stderr lines from CLI agents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<String>>,
}

/// Full session document on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatSession {
    /// Stable session identifier (UUID).
    pub id: String,
    /// Absolute project root this session belongs to.
    pub project_root: String,
    /// Short display title.
    pub title: String,
    /// RFC3339 creation timestamp.
    pub created_at: String,
    /// RFC3339 last-update timestamp.
    pub updated_at: String,
    /// Ordered conversation messages.
    pub messages: Vec<SessionMessage>,
}

/// Lightweight listing row (no message bodies).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionSummary {
    pub id: String,
    pub title: String,
    pub updated_at: String,
    pub created_at: String,
}

/// Body for `POST /sessions`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSessionRequest {
    /// Absolute (or resolvable) project root path.
    pub project: String,
    /// Optional title; defaults to "New session".
    #[serde(default)]
    pub title: Option<String>,
}

/// Body for `PUT /sessions/:id` (full replace of mutable fields).
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateSessionRequest {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub messages: Option<Vec<SessionMessage>>,
}

/// Normalize a project path to a canonical absolute string when possible.
pub fn normalize_project_root(project: &str) -> Result<String, String> {
    let trimmed = project.trim();
    if trimmed.is_empty() {
        return Err("project path is empty".to_string());
    }
    let path = PathBuf::from(trimmed);
    let abs = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .map_err(|e| format!("cwd: {e}"))?
            .join(path)
    };
    // Prefer canonicalize when the directory exists; otherwise keep absolute form.
    let normalized = fs::canonicalize(&abs).unwrap_or(abs);
    Ok(normalized.to_string_lossy().to_string())
}

fn sessions_dir(project_root: &str) -> PathBuf {
    Path::new(project_root)
        .join(".chaosnexus")
        .join("crucible")
        .join("sessions")
}

fn session_path(project_root: &str, id: &str) -> PathBuf {
    sessions_dir(project_root).join(format!("{id}.json"))
}

fn now_rfc3339() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Ensure the sessions directory exists for a project.
fn ensure_sessions_dir(project_root: &str) -> Result<PathBuf, String> {
    let dir = sessions_dir(project_root);
    fs::create_dir_all(&dir).map_err(|e| format!("create sessions dir: {e}"))?;
    Ok(dir)
}

/// Create a new empty session under the project.
pub fn create_session(req: CreateSessionRequest) -> Result<ChatSession, String> {
    let project_root = normalize_project_root(&req.project)?;
    ensure_sessions_dir(&project_root)?;
    let id = Uuid::new_v4().to_string();
    let stamp = now_rfc3339();
    let session = ChatSession {
        id: id.clone(),
        project_root: project_root.clone(),
        title: req
            .title
            .filter(|t| !t.trim().is_empty())
            .unwrap_or_else(|| "New session".to_string()),
        created_at: stamp.clone(),
        updated_at: stamp,
        messages: Vec::new(),
    };
    write_session(&session)?;
    Ok(session)
}

fn write_session(session: &ChatSession) -> Result<(), String> {
    ensure_sessions_dir(&session.project_root)?;
    let path = session_path(&session.project_root, &session.id);
    let body = serde_json::to_string_pretty(session).map_err(|e| format!("serialize: {e}"))?;
    fs::write(&path, body).map_err(|e| format!("write session: {e}"))
}

/// List session summaries for a project (newest `updated_at` first).
pub fn list_sessions(project: &str) -> Result<Vec<SessionSummary>, String> {
    let project_root = normalize_project_root(project)?;
    let dir = sessions_dir(&project_root);
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let entries = fs::read_dir(&dir).map_err(|e| format!("read sessions dir: {e}"))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let raw = match fs::read_to_string(&path) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let session: ChatSession = match serde_json::from_str(&raw) {
            Ok(s) => s,
            Err(_) => continue,
        };
        // Skip sessions that belong to a different project root (defense in depth).
        if session.project_root != project_root {
            continue;
        }
        out.push(SessionSummary {
            id: session.id,
            title: session.title,
            updated_at: session.updated_at,
            created_at: session.created_at,
        });
    }
    out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(out)
}

/// Load a session by id, searching under the given project root.
pub fn get_session(project: &str, id: &str) -> Result<ChatSession, String> {
    let project_root = normalize_project_root(project)?;
    let path = session_path(&project_root, id);
    if !path.exists() {
        return Err(format!("session `{id}` not found"));
    }
    let raw = fs::read_to_string(&path).map_err(|e| format!("read session: {e}"))?;
    let session: ChatSession =
        serde_json::from_str(&raw).map_err(|e| format!("parse session: {e}"))?;
    if session.project_root != project_root {
        return Err(format!("session `{id}` does not belong to this project"));
    }
    Ok(session)
}

/// Update title and/or messages for an existing session.
pub fn update_session(
    project: &str,
    id: &str,
    req: UpdateSessionRequest,
) -> Result<ChatSession, String> {
    let mut session = get_session(project, id)?;
    if let Some(title) = req.title
        && !title.trim().is_empty() {
            session.title = title;
        }
    if let Some(messages) = req.messages {
        session.messages = messages;
    }
    session.updated_at = now_rfc3339();
    write_session(&session)?;
    Ok(session)
}

/// Delete a session file.
pub fn delete_session(project: &str, id: &str) -> Result<(), String> {
    let project_root = normalize_project_root(project)?;
    let path = session_path(&project_root, id);
    if !path.exists() {
        return Err(format!("session `{id}` not found"));
    }
    fs::remove_file(&path).map_err(|e| format!("delete session: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn create_list_update_delete_roundtrip() {
        let tmp = env::temp_dir().join(format!("crucible-sess-{}", Uuid::new_v4()));
        fs::create_dir_all(&tmp).unwrap();
        let project = tmp.to_string_lossy().to_string();

        let created = create_session(CreateSessionRequest {
            project: project.clone(),
            title: Some("Test".into()),
        })
        .unwrap();
        assert_eq!(created.title, "Test");

        let listed = list_sessions(&project).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, created.id);

        let updated = update_session(
            &project,
            &created.id,
            UpdateSessionRequest {
                title: Some("Renamed".into()),
                messages: Some(vec![SessionMessage {
                    role: "user".into(),
                    text: "hi".into(),
                    diagnostics: None,
                }]),
            },
        )
        .unwrap();
        assert_eq!(updated.title, "Renamed");
        assert_eq!(updated.messages.len(), 1);

        delete_session(&project, &created.id).unwrap();
        assert!(list_sessions(&project).unwrap().is_empty());
        let _ = fs::remove_dir_all(&tmp);
    }
}
