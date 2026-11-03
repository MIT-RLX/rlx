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
        ["list_files", "read_file", "write_file", "edit_file"]
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
