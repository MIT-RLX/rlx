// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The workspace MCP server: reads anywhere in the folder, writes only the
//! writable names, never leaves the folder; and the `rlx-agent workspace-mcp`
//! binary answering a CLI's JSON-RPC over stdio.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use rlx_agent::workspace::Workspace;
use serde_json::{Value, json};

fn folder(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rlx-agent-ws-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.join("inbox")).unwrap();
    std::fs::write(d.join("updater.json"), "{\"lr\": 1}\n").unwrap();
    std::fs::write(d.join("inbox/decision-001.json"), "{}").unwrap();
    std::fs::write(d.join(".hidden"), "x").unwrap();
    d
}

#[test]
fn reads_anywhere_writes_only_the_writable_names() {
    let d = folder("tools");
    let ws = Workspace::new(&d, ["updater.json", "NOTES.md"]);
    let list = ws.call("list_files", &json!({})).unwrap();
    assert_eq!(
        list,
        "inbox/decision-001.json (2 bytes)\nupdater.json (10 bytes), writable"
    );
    assert_eq!(
        ws.call("read_file", &json!({"path": "inbox/decision-001.json"}))
            .unwrap(),
        "{}"
    );

    ws.call(
        "edit_file",
        &json!({"path": "updater.json", "old": "1", "new": "2"}),
    )
    .unwrap();
    assert_eq!(
        std::fs::read_to_string(d.join("updater.json")).unwrap(),
        "{\"lr\": 2}\n"
    );
    ws.call(
        "write_file",
        &json!({"path": "NOTES.md", "content": "tried lr 2"}),
    )
    .unwrap();
    assert!(
        d.join("NOTES.md").exists(),
        "a writable name may be created"
    );

    let refused = |tool: &str, args: Value| ws.call(tool, &args).unwrap_err();
    assert!(
        refused(
            "write_file",
            json!({"path": "inbox/decision-001.json", "content": ""})
        )
        .contains("read-only")
    );
    assert!(refused("write_file", json!({"path": "evil.py", "content": ""})).contains("read-only"));
    assert!(refused("read_file", json!({"path": "../outside"})).contains("no `..`"));
    assert!(refused("read_file", json!({"path": "/etc/hosts"})).contains("no absolute paths"));
    assert!(
        refused(
            "edit_file",
            json!({"path": "updater.json", "old": "zz", "new": "y"})
        )
        .contains("does not occur")
    );
    assert!(
        refused(
            "edit_file",
            json!({"path": "NOTES.md", "old": "", "new": "y"})
        )
        .contains("occurs")
    );
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn the_binary_serves_json_rpc_over_stdio() {
    let d = folder("bin");
    let mut child = Command::new(env!("CARGO_BIN_EXE_rlx-agent"))
        .args([
            "workspace-mcp",
            "--dir",
            &d.display().to_string(),
            "--write",
            "updater.json",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap()).lines();
    let mut ask = |msg: Value| -> Value {
        writeln!(stdin, "{msg}").unwrap();
        serde_json::from_str(&out.next().unwrap().unwrap()).unwrap()
    };
    let init = ask(
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}),
    );
    assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
    let tools = ask(json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}));
    let names: Vec<&str> = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "list_files",
            "read_file",
            "search",
            "write_file",
            "edit_file"
        ]
    );
    let wrote = ask(json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                           "params": {"name": "write_file", "arguments": {"path": "updater.json", "content": "{}"}}}));
    assert_eq!(wrote["result"]["isError"], false);
    let refused = ask(json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                             "params": {"name": "write_file", "arguments": {"path": "x.txt", "content": "{}"}}}));
    assert_eq!(refused["result"]["isError"], true);
    drop(stdin);
    assert!(child.wait().unwrap().success());
    assert_eq!(
        std::fs::read_to_string(d.join("updater.json")).unwrap(),
        "{}"
    );
    assert!(!d.join("x.txt").exists());
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn the_server_entry_carries_the_folder_and_its_writable_names() {
    let ws = Workspace::new("/runs/1/workspace", ["updater.json", "NOTES.md"]);
    let s = ws.server(
        "workspace",
        "/bin/rlx-agent",
        &["workspace-mcp".to_string()],
    );
    assert_eq!(
        s.args,
        [
            "workspace-mcp",
            "--dir",
            "/runs/1/workspace",
            "--write",
            "updater.json",
            "--write",
            "NOTES.md"
        ]
    );
    assert_eq!(Workspace::from_args(&s.args[1..]).unwrap(), ws);
    assert!(Workspace::from_args(&["--write".into()]).is_err());
}

#[test]
fn reads_line_ranges_and_searches_the_files() {
    let d = folder("search");
    std::fs::write(d.join("losses.csv"), "update,loss\n1,2.5\n2,2.25\n3,2.0\n").unwrap();
    let ws = Workspace::new(&d, ["updater.json"]);
    assert_eq!(
        ws.call(
            "read_file",
            &json!({"path": "losses.csv", "start_line": 2, "lines": 2})
        )
        .unwrap(),
        "     2  1,2.5\n     3  2,2.25\n[lines 2–3 of 4]"
    );
    assert!(
        ws.call("read_file", &json!({"path": "losses.csv", "start_line": 9}))
            .unwrap_err()
            .contains("has 4 lines")
    );
    assert_eq!(
        ws.call("search", &json!({"text": "LR", "ignore_case": true}))
            .unwrap(),
        "updater.json:1: {\"lr\": 1}"
    );
    assert_eq!(
        ws.call("search", &json!({"text": "2.25"})).unwrap(),
        "losses.csv:3: 2,2.25"
    );
    assert!(
        ws.call("search", &json!({"text": "lr", "path": "inbox"}))
            .unwrap()
            .starts_with("no line contains")
    );
}

#[test]
fn a_json_file_written_with_a_syntax_error_says_so() {
    let d = folder("json");
    let ws = Workspace::new(&d, ["updater.json"]);
    let reply = ws
        .call(
            "write_file",
            &json!({"path": "updater.json", "content": "{\"lr\": }"}),
        )
        .unwrap();
    assert!(reply.contains("not valid JSON yet"), "{reply}");
    let reply = ws
        .call(
            "edit_file",
            &json!({"path": "updater.json", "old": "}", "new": "1}"}),
        )
        .unwrap();
    assert_eq!(reply, "edited updater.json", "valid again");
}

#[cfg(unix)]
#[test]
fn links_out_of_the_folder_are_not_followed() {
    let d = folder("links");
    let outside = folder("links-outside");
    std::fs::write(outside.join("secret.txt"), "key").unwrap();
    std::os::unix::fs::symlink(outside.join("secret.txt"), d.join("leak.txt")).unwrap();
    std::os::unix::fs::symlink(&outside, d.join("leakdir")).unwrap();
    let ws = Workspace::new(&d, ["updater.json"]);
    assert!(
        ws.call("read_file", &json!({"path": "leak.txt"}))
            .unwrap_err()
            .contains("leads outside the workspace")
    );
    let list = ws.call("list_files", &json!({})).unwrap();
    assert!(!list.contains("leak"), "{list}");
    assert!(
        ws.call("search", &json!({"text": "key"}))
            .unwrap()
            .starts_with("no line")
    );
}

#[cfg(unix)]
#[test]
fn check_program_runs_its_command_on_the_folder() {
    let d = folder("check");
    let ws = Workspace::new(&d, ["updater.json"]);
    assert!(
        !ws.tools().to_string().contains("check_program"),
        "no command, no tool"
    );
    assert!(ws.call("check_program", &json!({})).is_err());

    let ok = ws.clone().with_check(vec![
        "sh".into(),
        "-c".into(),
        "echo checked \"$0\"; cat updater.json".into(),
    ]);
    assert!(ok.tools().to_string().contains("check_program"));
    let out = ok.call("check_program", &json!({})).unwrap();
    assert_eq!(out, format!("checked {}\n{{\"lr\": 1}}", d.display()));

    let fails = ws.clone().with_check(vec![
        "sh".into(),
        "-c".into(),
        "echo 'update 1: not finite' >&2; exit 3".into(),
    ]);
    let err = fails.call("check_program", &json!({})).unwrap_err();
    assert!(
        err.contains("the check failed") && err.contains("update 1: not finite"),
        "{err}"
    );

    // `--check` words reach the served copy.
    let server = ok.server("rlx_files", "rlx-agent", &["workspace-mcp".to_string()]);
    let served = Workspace::from_args(&server.args[1..]).unwrap();
    assert_eq!(served, ok);
}
