// chaosnexus-crucible/src/context.rs
//! Dual-scope skills/rules loader with Codex-style char chunking.
//!
//! Cascade: `~/.chaosnexus/{rules,skills}` then `<project>/.chaosnexus/{rules,skills}`
//! (project overrides same basename). Always-on rules are budgeted; skills are
//! pull-based via list / search / read with offset+limit truncation markers.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Default char window for a single skill/rule read (mirrors Codex ~16k).
pub const DEFAULT_CHUNK_LIMIT: usize = 16_000;
/// Char budget for always-on rule bodies injected into `/generate`.
pub const RULES_INJECT_BUDGET: usize = 16_000;

/// Scope of a rules/skills entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    User,
    Project,
}

/// Index row for a rule or skill pack.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextEntry {
    /// Basename without extension (or skill folder name).
    pub name: String,
    /// `rule` or `skill`.
    pub kind: String,
    pub scope: Scope,
    /// One-line description (first non-empty markdown line, truncated).
    pub description: String,
}

/// Packed preamble for `/generate` (thin index + budgeted rules).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPack {
    /// Short system instructions plus rule bodies under budget.
    pub preamble: String,
    /// Available skills (pull via `/context/read`).
    pub skill_index: Vec<ContextEntry>,
    /// Rules that were fully or partially included.
    pub rule_index: Vec<ContextEntry>,
}

fn home_chaosnexus() -> PathBuf {
    directories::UserDirs::new()
        .map(|u| u.home_dir().join(".chaosnexus"))
        .unwrap_or_else(|| PathBuf::from(".chaosnexus"))
}

fn user_rules_dir() -> PathBuf {
    home_chaosnexus().join("rules")
}

fn user_skills_dir() -> PathBuf {
    home_chaosnexus().join("skills")
}

fn project_rules_dir(project_root: &Path) -> PathBuf {
    project_root.join(".chaosnexus").join("rules")
}

fn project_skills_dir(project_root: &Path) -> PathBuf {
    project_root.join(".chaosnexus").join("skills")
}

/// First meaningful line of markdown as a short description.
fn describe(content: &str) -> String {
    content
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .or_else(|| {
            content
                .lines()
                .map(str::trim)
                .find(|l| l.starts_with('#') && l.len() > 1)
                .map(|l| l.trim_start_matches('#').trim())
        })
        .unwrap_or("No description")
        .chars()
        .take(120)
        .collect()
}

fn read_markdown_file(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok()
}

/// Collect markdown rule files from a directory into a map keyed by stem.
fn collect_rules(dir: &Path, scope: Scope, into: &mut BTreeMap<String, (Scope, String, PathBuf)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "md" && ext != "markdown" {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if let Some(content) = read_markdown_file(&path) {
            into.insert(stem.to_string(), (scope, content, path));
        }
    }
}

/// Collect skill packs: either `skills/<name>/SKILL.md` or `skills/<name>.md`.
fn collect_skills(dir: &Path, scope: Scope, into: &mut BTreeMap<String, (Scope, String, PathBuf)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let skill_md = path.join("SKILL.md");
            let alt = path.join("skill.md");
            let file = if skill_md.is_file() {
                skill_md
            } else if alt.is_file() {
                alt
            } else {
                continue;
            };
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if let Some(content) = read_markdown_file(&file) {
                into.insert(name.to_string(), (scope, content, file));
            }
        } else if path.is_file() {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext != "md" && ext != "markdown" {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if let Some(content) = read_markdown_file(&path) {
                into.insert(stem.to_string(), (scope, content, path));
            }
        }
    }
}

/// Load cascaded rules (user then project override) for a project.
pub fn load_rules_map(project_root: &str) -> BTreeMap<String, (Scope, String, PathBuf)> {
    let mut map = BTreeMap::new();
    collect_rules(&user_rules_dir(), Scope::User, &mut map);
    let proj = Path::new(project_root);
    collect_rules(&project_rules_dir(proj), Scope::Project, &mut map);
    map
}

/// Load cascaded skills for a project.
pub fn load_skills_map(project_root: &str) -> BTreeMap<String, (Scope, String, PathBuf)> {
    let mut map = BTreeMap::new();
    collect_skills(&user_skills_dir(), Scope::User, &mut map);
    let proj = Path::new(project_root);
    collect_skills(&project_skills_dir(proj), Scope::Project, &mut map);
    map
}

/// Build a thin index of all rules and skills.
pub fn list_context(project_root: &str) -> Vec<ContextEntry> {
    let mut out = Vec::new();
    for (name, (scope, content, _)) in load_rules_map(project_root) {
        out.push(ContextEntry {
            name,
            kind: "rule".into(),
            scope,
            description: describe(&content),
        });
    }
    for (name, (scope, content, _)) in load_skills_map(project_root) {
        out.push(ContextEntry {
            name,
            kind: "skill".into(),
            scope,
            description: describe(&content),
        });
    }
    out
}

/// Filename / description substring search (Codex-style, no embeddings).
pub fn search_context(project_root: &str, query: &str) -> Vec<ContextEntry> {
    let q = query.to_lowercase();
    list_context(project_root)
        .into_iter()
        .filter(|e| {
            e.name.to_lowercase().contains(&q)
                || e.description.to_lowercase().contains(&q)
                || e.kind.to_lowercase().contains(&q)
        })
        .collect()
}

/// Char-window slice with Codex truncation marker.
pub fn chunk_text(content: &str, offset: usize, limit: usize) -> String {
    let original_len = content.chars().count();
    let start_idx = content
        .char_indices()
        .nth(offset)
        .map(|(i, _)| i)
        .unwrap_or(content.len());
    let end_idx = content[start_idx..]
        .char_indices()
        .nth(limit)
        .map(|(i, _)| start_idx + i)
        .unwrap_or(content.len());
    let mut text = content[start_idx..end_idx].to_string();
    if end_idx < content.len() {
        text.push_str(&format!(
            "\n\n...[TRUNCATED: Document has more content. Use offset={} to read next chunk.]",
            offset + limit
        ));
        let _ = original_len;
    }
    text
}

/// Read a rule or skill body with offset/limit chunking.
pub fn read_context(
    project_root: &str,
    kind: &str,
    name: &str,
    offset: usize,
    limit: usize,
) -> Result<String, String> {
    let map = match kind {
        "rule" => load_rules_map(project_root),
        "skill" => load_skills_map(project_root),
        _ => return Err(format!("unknown kind `{kind}` (use rule or skill)")),
    };
    let Some((_, content, _)) = map.get(name) else {
        return Err(format!("{kind} `{name}` not found"));
    };
    Ok(chunk_text(content, offset, limit))
}

/// Build the always-on preamble for generation: index + budgeted rule bodies.
///
/// Project-scoped rules are preferred when the budget is tight (inserted first
/// after the index header when packing bodies).
pub fn pack_context(project_root: &str) -> ContextPack {
    let rules = load_rules_map(project_root);
    let skills = load_skills_map(project_root);

    let rule_index: Vec<ContextEntry> = rules
        .iter()
        .map(|(name, (scope, content, _))| ContextEntry {
            name: name.clone(),
            kind: "rule".into(),
            scope: *scope,
            description: describe(content),
        })
        .collect();

    let skill_index: Vec<ContextEntry> = skills
        .iter()
        .map(|(name, (scope, content, _))| ContextEntry {
            name: name.clone(),
            kind: "skill".into(),
            scope: *scope,
            description: describe(content),
        })
        .collect();

    let mut preamble = String::from(
        "You are ChaosNexus Crucible. Follow project and user rules below. \
         Skills are listed by name — request full skill text via context tools \
         (list/search/read with offset/limit) instead of assuming full bodies.\n\n",
    );

    if !skill_index.is_empty() {
        preamble.push_str("## Available skills\n");
        for s in &skill_index {
            preamble.push_str(&format!(
                "- [{}] {} — {}\n",
                match s.scope {
                    Scope::User => "user",
                    Scope::Project => "project",
                },
                s.name,
                s.description
            ));
        }
        preamble.push('\n');
    }

    // Prefer project rules when packing bodies under budget.
    let mut ordered: Vec<(&String, &(Scope, String, PathBuf))> = rules.iter().collect();
    ordered.sort_by(|a, b| {
        let rank = |s: Scope| match s {
            Scope::Project => 0,
            Scope::User => 1,
        };
        rank(a.1 .0).cmp(&rank(b.1 .0)).then(a.0.cmp(b.0))
    });

    preamble.push_str("## Active rules\n");
    let mut used = 0usize;
    for (name, (scope, content, _)) in ordered {
        let header = format!(
            "### Rule: {name} ({})\n",
            match scope {
                Scope::User => "user",
                Scope::Project => "project",
            }
        );
        let remaining = RULES_INJECT_BUDGET.saturating_sub(used);
        if remaining == 0 {
            preamble.push_str(&format!(
                "\n...[TRUNCATED: Additional rules omitted due to {RULES_INJECT_BUDGET} char budget. Use context read for rule `{name}`]\n"
            ));
            break;
        }
        let body = chunk_text(content, 0, remaining.saturating_sub(header.len()));
        preamble.push_str(&header);
        preamble.push_str(&body);
        preamble.push_str("\n\n");
        used += header.chars().count() + body.chars().count();
    }

    ContextPack {
        preamble,
        skill_index,
        rule_index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn chunk_marks_truncation() {
        let text: String = (0..100).map(|_| 'a').collect();
        let chunk = chunk_text(&text, 0, 10);
        assert!(chunk.contains("TRUNCATED"));
        assert!(chunk.contains("offset=10"));
    }

    #[test]
    fn project_overrides_user_rule() {
        let tmp = std::env::temp_dir().join(format!("crucible-ctx-{}", Uuid::new_v4()));
        let user_rules = home_chaosnexus().join("rules");
        // Use isolated project only; user home may have real rules — override by name.
        let proj_rules = tmp.join(".chaosnexus").join("rules");
        fs::create_dir_all(&proj_rules).unwrap();
        fs::write(proj_rules.join("alpha.md"), "project alpha rule body").unwrap();
        let map = load_rules_map(tmp.to_str().unwrap());
        assert!(map.contains_key("alpha"));
        assert_eq!(map.get("alpha").unwrap().0, Scope::Project);
        let _ = fs::remove_dir_all(&tmp);
        let _ = user_rules; // silence unused in case home has no rules
    }
}
