// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo rlx doctor --report`: one paste-ready Markdown block for a GitHub
//! issue, so a bug report arrives with the host already described — and, when
//! a known issue matched, with its fix already named.

use super::{DeviceStatus, DoctorReport, EnvVarStatus};

impl DoctorReport {
    /// The report as Markdown, with the home directory and user name redacted.
    pub fn to_markdown(&self) -> String {
        let mut s = String::new();
        s.push_str("<details open><summary><code>cargo rlx doctor --report</code></summary>\n\n");
        s.push_str(&format!(
            "**rlx** {} · **host** {}-{} · **built with** `{}`\n\n",
            self.version,
            self.os,
            self.arch,
            if self.features.is_empty() {
                "(no backends)".to_string()
            } else {
                self.features.join(",")
            }
        ));

        s.push_str("| Device | Status | Check | Detail |\n|---|---|---|---|\n");
        for d in self
            .devices
            .iter()
            .filter(|d| d.status != DeviceStatus::NotCompiled)
        {
            let check = match &d.check {
                Some(c) if c.passed() => format!("ok, rel err {:.1e}", c.rel_err.unwrap_or(0.0)),
                Some(c) => c
                    .error
                    .clone()
                    .unwrap_or_else(|| format!("rel err {:.1e}", c.rel_err.unwrap_or(f64::NAN))),
                None => "—".into(),
            };
            let detail = [d.detail.clone(), d.fix.clone(), d.toolchain_warning.clone()]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(" · ");
            s.push_str(&format!(
                "| {} | {} | {} | {} |\n",
                d.device,
                d.status.label(),
                cell(&check),
                cell(&detail)
            ));
        }

        if !self.versions.is_empty() {
            s.push_str("\n**Toolchains:** ");
            let v: Vec<String> = self
                .versions
                .iter()
                .map(|(k, (a, b, c))| format!("{k} {a}.{b}.{c}"))
                .collect();
            s.push_str(&v.join(", "));
            s.push('\n');
        }

        if !self.issues.is_empty() {
            s.push_str("\n**Known issues matched:**\n\n");
            for i in &self.issues {
                s.push_str(&format!(
                    "- **{}** ({}) {} — fix: {}\n",
                    i.id,
                    i.severity.label(),
                    i.title,
                    i.fix
                ));
            }
        }

        if !self.env.is_empty() || !self.implied.is_empty() {
            s.push_str("\n**RLX_\\* settings:**\n\n```\n");
            for e in &self.env {
                let tag = match &e.status {
                    EnvVarStatus::Known { .. } => String::new(),
                    EnvVarStatus::Deprecated { use_instead } => {
                        format!("   # deprecated, use {use_instead}")
                    }
                    EnvVarStatus::Unknown { did_you_mean } => format!(
                        "   # unknown{}",
                        did_you_mean
                            .map(|s| format!(", did you mean {s}?"))
                            .unwrap_or_default()
                    ),
                };
                s.push_str(&format!("{}={}{tag}\n", e.name, e.value));
            }
            for (n, v, src) in &self.implied {
                s.push_str(&format!("{n}={v}   # from {src}\n"));
            }
            s.push_str("```\n");
        }
        s.push_str("\n</details>\n");
        redact(&s)
    }
}

/// A table cell: no pipes, no newlines, bounded length.
fn cell(s: &str) -> String {
    let one_line: String = s.replace('|', "\\|").replace('\n', " ");
    if one_line.chars().count() > 240 {
        let cut: String = one_line.chars().take(240).collect();
        format!("{cut}…")
    } else {
        one_line
    }
}

/// Replace the home directory with `~` and the user name with `<user>`.
pub(super) fn redact(s: &str) -> String {
    let mut out = s.to_string();
    if let Some(home) = std::env::var_os("HOME").map(|h| h.to_string_lossy().into_owned())
        && home.len() > 1
    {
        out = out.replace(&home, "~");
    }
    if let Ok(user) = std::env::var("USER")
        && user.len() > 2
    {
        out = out.replace(&format!("/{user}/"), "/<user>/");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_cannot_break_the_table() {
        assert_eq!(cell("a|b\nc"), "a\\|b c");
        assert!(cell(&"x".repeat(500)).ends_with('…'));
    }

    #[test]
    fn home_is_redacted() {
        if let Some(home) = std::env::var_os("HOME") {
            let h = home.to_string_lossy();
            assert_eq!(
                redact(&format!("AIECC={h}/venv/bin/aiecc")),
                "AIECC=~/venv/bin/aiecc"
            );
        }
    }
}
