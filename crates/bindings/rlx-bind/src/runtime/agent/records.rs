// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The run folder: where a run with an agent keeps its records, the notes the
//! person running it leaves for the agent, and what a finished run's program
//! was. Text that may carry credentials (the agent's answers, errors,
//! transcripts) is redacted before it is written.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use super::program::Files;
use crate::{ApiError, ApiResult};

/// Where you drop notes for the agent (`notes/*.txt`); read ones move to `notes/read/`.
pub const NOTES_DIR: &str = "notes";

/// The run folder and its records.
pub(crate) struct Run {
    pub dir: PathBuf,
    pub workspace: PathBuf,
}

pub(crate) fn unix_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

impl Run {
    pub fn create(root: &Path) -> ApiResult<Self> {
        let io = |e: std::io::Error| {
            ApiError::runtime(format!("agent: run folder under {}: {e}", root.display()))
        };
        fs::create_dir_all(root).map_err(io)?;
        let secs = unix_seconds() as u64;
        let mut n = 0u32;
        let dir = loop {
            let d = root.join(format!("{secs}-{}-{n}", std::process::id()));
            match fs::create_dir(&d) {
                Ok(()) => break d,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => n += 1,
                Err(e) => return Err(io(e)),
            }
        };
        Self::open(dir)
    }

    pub fn open(dir: PathBuf) -> ApiResult<Self> {
        for sub in [
            "workspace/inbox",
            "workspace/reference",
            "programs",
            "decisions",
            NOTES_DIR,
        ] {
            fs::create_dir_all(dir.join(sub)).map_err(|e| {
                ApiError::runtime(format!("agent: run folder {}: {e}", dir.display()))
            })?;
        }
        Ok(Self {
            workspace: dir.join("workspace"),
            dir,
        })
    }

    /// One line of `events.jsonl`, with the time.
    pub fn event(&self, kind: &str, fields: Value) {
        let mut rec = json!({"kind": kind, "t": (unix_seconds() * 1e3).round() / 1e3});
        if let (Value::Object(r), Value::Object(f)) = (&mut rec, fields) {
            r.extend(f);
        }
        self.append("events.jsonl", &redact(&rec.to_string()));
    }

    /// Append one line to a record file (created on first use).
    pub fn append(&self, rel: &str, line: &str) {
        if let Ok(mut f) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join(rel))
        {
            let _ = writeln!(f, "{line}");
        }
    }

    pub fn write(&self, rel: impl AsRef<Path>, text: &str) -> ApiResult<()> {
        let path = self.dir.join(rel);
        if let Some(p) = path.parent() {
            fs::create_dir_all(p).ok();
        }
        fs::write(&path, text)
            .map_err(|e| ApiError::runtime(format!("agent: writing {}: {e}", path.display())))
    }

    pub fn write_files(&self, rel: impl AsRef<Path>, files: &Files) -> ApiResult<()> {
        for (name, text) in files {
            self.write(rel.as_ref().join(name), text)?;
        }
        Ok(())
    }

    /// The notes left since the last call, oldest first; each moves to `notes/read/`.
    pub fn take_notes(&self) -> Vec<String> {
        let dir = self.dir.join(NOTES_DIR);
        let mut files: Vec<PathBuf> = fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "txt"))
            .collect();
        files.sort();
        let read = dir.join("read");
        let mut out = Vec::new();
        for f in files {
            if let Ok(text) = fs::read_to_string(&f) {
                let text = text.trim();
                if !text.is_empty() {
                    out.push(text.to_string());
                }
                fs::create_dir_all(&read).ok();
                let _ = fs::rename(&f, read.join(f.file_name().unwrap_or_default()));
            }
        }
        out
    }
}

/// Leave a note for the agent of the run in `run_dir`: it is read at the next
/// decision (`notes/<time>.txt`).
pub fn say(run_dir: &Path, text: &str) -> ApiResult<PathBuf> {
    let dir = run_dir.join(NOTES_DIR);
    if !run_dir.join("events.jsonl").exists() {
        return Err(ApiError::type_(format!(
            "{} is not a run folder (no events.jsonl)",
            run_dir.display()
        )));
    }
    fs::create_dir_all(&dir).map_err(|e| ApiError::runtime(e.to_string()))?;
    let path = dir.join(format!("{:.6}.txt", unix_seconds()));
    fs::write(&path, text).map_err(|e| ApiError::runtime(e.to_string()))?;
    Ok(path)
}

/// The program running at the end of the run in `run_dir` (its `FINAL`
/// version) → its version and files, to run again without an agent
/// ([`super::AgentOptimizer::replay`]). `path` may also be a program folder
/// (`programs/vNNN`).
pub fn final_program(path: &Path) -> ApiResult<(u32, Files)> {
    let (version, dir) = if path.join("FINAL").is_file() {
        let v = fs::read_to_string(path.join("FINAL"))
            .map_err(|e| ApiError::runtime(e.to_string()))?
            .trim()
            .to_string();
        (v.clone(), path.join("programs").join(v))
    } else {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        (name, path.to_path_buf())
    };
    let number = version
        .strip_prefix('v')
        .and_then(|n| n.parse::<u32>().ok())
        .ok_or_else(|| {
            ApiError::type_(format!(
                "{}: neither a run folder (with FINAL) nor a program folder (vNNN)",
                path.display()
            ))
        })?;
    let mut files = Files::new();
    for e in fs::read_dir(&dir)
        .map_err(|e| ApiError::runtime(format!("{}: {e}", dir.display())))?
        .flatten()
    {
        let p = e.path();
        if p.is_file()
            && let Ok(text) = fs::read_to_string(&p)
        {
            files.insert(e.file_name().to_string_lossy().into_owned(), text);
        }
    }
    if files.is_empty() {
        return Err(ApiError::type_(format!(
            "{}: no program files",
            dir.display()
        )));
    }
    Ok((number, files))
}

/// Credentials that look like these are replaced by `[redacted]` in records.
const TOKEN_PREFIXES: [&str; 9] = [
    "sk-ant-",
    "sk-proj-",
    "sk-",
    "ghp_",
    "gho_",
    "github_pat_",
    "xoxb-",
    "xoxp-",
    "hf_",
];

/// `text` with API keys, access tokens and private keys replaced.
pub fn redact(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    'scan: while !rest.is_empty() {
        if let Some(i) = rest.find("-----BEGIN ")
            && rest[i..].starts_with("-----BEGIN ")
            && rest[i..]
                .lines()
                .next()
                .is_some_and(|l| l.contains("PRIVATE KEY-----"))
        {
            out.push_str(&rest[..i]);
            match rest[i..].find("-----END ") {
                Some(j) => {
                    let end = rest[i + j..]
                        .find("KEY-----")
                        .map(|k| i + j + k + 8)
                        .unwrap_or(rest.len());
                    out.push_str("[redacted private key]");
                    rest = &rest[end..];
                }
                None => {
                    out.push_str("[redacted private key]");
                    rest = "";
                }
            }
            continue 'scan;
        }
        // Token prefixes and AWS access key ids, at a word start.
        let mut cut = None;
        let mut prev: Option<char> = None;
        for (i, c) in rest.char_indices() {
            let start_ok = prev.is_none_or(|p| !is_token_char(p));
            prev = Some(c);
            if !start_ok {
                continue;
            }
            let tail = &rest[i..];
            let prefix = TOKEN_PREFIXES
                .iter()
                .find(|p| tail.starts_with(**p))
                .map(|p| p.len());
            let aws = (tail.starts_with("AKIA") || tail.starts_with("ASIA")) && tail.len() >= 20;
            let bearer = tail.starts_with("Bearer ");
            let skip = if bearer {
                Some(7)
            } else if aws {
                Some(4)
            } else {
                prefix
            };
            if let Some(p) = skip {
                let len = tail[p..]
                    .chars()
                    .take_while(|c| is_token_char(*c))
                    .map(char::len_utf8)
                    .sum::<usize>();
                if len >= 16 {
                    cut = Some((i, i + p + len, bearer));
                    break;
                }
            }
        }
        match cut {
            Some((a, b, bearer)) => {
                out.push_str(&rest[..a]);
                out.push_str(if bearer {
                    "Bearer [redacted]"
                } else {
                    "[redacted]"
                });
                rest = &rest[b..];
            }
            None => {
                out.push_str(rest);
                rest = "";
            }
        }
    }
    out
}

fn is_token_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_and_tokens_are_redacted_and_prose_is_not() {
        let key = format!("sk-ant-api03-{}", "a".repeat(40));
        let t = format!("using {key} now; Bearer abcdefghijklmnopqrstu; AKIAABCDEFGHIJKLMNOP.");
        assert_eq!(
            redact(&t),
            "using [redacted] now; Bearer [redacted]; [redacted]."
        );
        let pem = "a\n-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----\nb";
        assert_eq!(redact(pem), "a\n[redacted private key]\nb");
        let prose = "risk-adjusted sk-short task-specific learning rate 3e-4";
        assert_eq!(redact(prose), prose);
    }
}
