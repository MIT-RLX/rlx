// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Command lines, MCP configs, and event parsing for each agent CLI. The
//! event fixtures are trimmed copies of real lines from each CLI's output.

use rlx_agent::{AgentCli, AgentRun, Event, McpServer, Summary, parse_event};
use serde_json::{Value, json};

fn run(cli: AgentCli, dir: &std::path::Path) -> AgentRun {
    let mut r = AgentRun::new(
        cli,
        dir,
        vec![
            McpServer::new("wsp", "rlx-wsp", ["mcp"]),
            McpServer::new("aux", "aux-server", Vec::<String>::new()),
        ],
    );
    r.model = Some("m".into());
    r.effort = Some("high".into());
    r.max_turns = Some(7);
    r.max_budget_usd = Some(2.5);
    r.extra_args = vec!["--extra".into()];
    r
}

fn after(a: &[String], flag: &str) -> Vec<String> {
    a.windows(2)
        .filter(|w| w[0] == flag)
        .map(|w| w[1].clone())
        .collect()
}

#[test]
fn claude_has_no_builtin_tools_and_only_the_listed_servers() {
    let (prog, a) = run(AgentCli::Claude, "/tmp/x".as_ref()).command_line("P");
    assert_eq!(prog, "claude");
    assert_eq!(after(&a, "-p"), ["P"]);
    assert_eq!(after(&a, "--tools"), [""]);
    assert_eq!(after(&a, "--allowedTools"), ["mcp__wsp", "mcp__aux"]);
    assert_eq!(after(&a, "--permission-mode"), ["dontAsk"]);
    assert_eq!(after(&a, "--max-turns"), ["7"]);
    assert_eq!(after(&a, "--effort"), ["high"]);
    assert_eq!(after(&a, "--max-budget-usd"), ["2.5"]);
    assert_eq!(after(&a, "--model"), ["m"]);
    assert!(a.contains(&"--strict-mcp-config".to_string()));
    assert_eq!(a.last().unwrap(), "--extra");
}

#[test]
fn cursor_and_copilot_flags() {
    let (p, a) = run(AgentCli::Cursor, "/tmp/x".as_ref()).command_line("P");
    assert_eq!(p, "agent");
    assert!(a.contains(&"--approve-mcps".to_string()));
    assert_eq!(after(&a, "--workspace"), ["/tmp/x/workspace"]);
    // Claude-only options are not passed to other CLIs.
    assert!(after(&a, "--effort").is_empty() && after(&a, "--max-turns").is_empty());
    let (p, a) = run(AgentCli::Copilot, "/tmp/x".as_ref()).command_line("P");
    assert_eq!(p, "copilot");
    assert_eq!(after(&a, "--additional-mcp-config"), ["@/tmp/x/mcp.json"]);
    assert_eq!(after(&a, "--allow-tool"), ["wsp", "aux"]);
    assert_eq!(after(&a, "--deny-tool"), ["shell", "write"]);
}

#[test]
fn prepare_writes_each_clis_config() {
    let root = std::env::temp_dir().join(format!("rlx_agent_prep_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for cli in AgentCli::ALL {
        let dir = root.join(cli.program());
        let r = run(cli, &dir);
        r.prepare().unwrap();
        let cfg: Value =
            serde_json::from_str(&std::fs::read_to_string(r.mcp_config_path()).unwrap()).unwrap();
        assert_eq!(cfg["mcpServers"]["wsp"]["command"], "rlx-wsp");
        assert_eq!(cfg["mcpServers"]["wsp"]["args"], json!(["mcp"]));
        assert_eq!(
            cfg["mcpServers"]["wsp"]["type"] == "local",
            cli == AgentCli::Copilot
        );
        assert!(r.workspace.is_dir());
        assert_eq!(
            dir.join("workspace/.cursor/mcp.json").is_file(),
            cli == AgentCli::Cursor
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

fn summarise(cli: AgentCli, lines: &[Value]) -> Summary {
    let mut s = Summary::default();
    for l in lines {
        for e in parse_event(cli, l, &["wsp"]) {
            s.record(e);
        }
    }
    s
}

#[test]
fn claude_events() {
    let s = summarise(
        AgentCli::Claude,
        &[
            json!({"type":"system","subtype":"init","model":"opus"}),
            json!({"type":"system","subtype":"thinking_tokens","estimated_tokens":100}),
            json!({"type":"assistant","message":{"content":[
                {"type":"tool_use","name":"mcp__wsp__describe","input":{}},
                {"type":"text","text":"Reading the rules."}]}}),
            json!({"type":"assistant","message":{"content":[{"type":"text","text":"Done: 22."}]}}),
            json!({"type":"result","subtype":"success","num_turns":12,
                   "total_cost_usd":0.19,"duration_ms":53285,"result":"Done: 22."}),
        ],
    );
    assert_eq!(s.models, ["opus"]);
    assert_eq!(s.tool_calls["describe"], 1);
    assert_eq!(s.foreign_tool_calls, 0);
    assert_eq!(s.final_text.as_deref(), Some("Done: 22."));
    assert_eq!(
        (s.num_turns, s.cost_usd, s.duration_ms),
        (Some(12), Some(0.19), Some(53285))
    );
}

#[test]
fn cursor_events() {
    let s = summarise(
        AgentCli::Cursor,
        &[
            json!({"type":"system","subtype":"init","model":"Muse Wonder 300K Medium"}),
            json!({"type":"tool_call","subtype":"started","tool_call":{
                "getMcpToolsToolCall":{"args":{"server":"wsp"}},"toolCallId":"a"}}),
            json!({"type":"tool_call","subtype":"started","tool_call":{
                "mcpToolCall":{"args":{"name":"wsp-list_params","args":{}}},"toolCallId":"b"}}),
            json!({"type":"tool_call","subtype":"completed","tool_call":{
                "mcpToolCall":{"args":{"name":"wsp-list_params","args":{}}}}}),
            json!({"type":"assistant","message":{"content":[{"type":"text","text":"first"}]}}),
            json!({"type":"assistant","message":{"content":[{"type":"text","text":"Program written."}]}}),
            json!({"type":"result","subtype":"success","duration_ms":453340,
                   "result":"firstProgram written."}),
        ],
    );
    assert_eq!(s.models, ["Muse Wonder 300K Medium"]);
    assert_eq!(s.tool_calls["list_params"], 1);
    assert_eq!(s.tool_calls["getMcpTools"], 1);
    assert_eq!(s.foreign_tool_calls, 1);
    // Cursor's `result` concatenates every message; the last message wins.
    assert_eq!(s.final_text.as_deref(), Some("Program written."));
    assert_eq!(s.duration_ms, Some(453340));
}

#[test]
fn copilot_events() {
    let s = summarise(
        AgentCli::Copilot,
        &[
            json!({"type":"model.call_start","data":{"model":"claude-sonnet-5.5"}}),
            json!({"type":"assistant.tool_call_delta","data":{"toolName":"wsp-set","inputDelta":"{"}}),
            json!({"type":"tool.execution_start","data":{"toolName":"wsp-describe","arguments":{}}}),
            json!({"type":"tool.execution_start","data":{"toolName":"bash","arguments":{"command":"ls"}}}),
            json!({"type":"model.call_start","data":{"model":"claude-sonnet-5.5"}}),
            json!({"type":"assistant.message","data":{"content":""}}),
            json!({"type":"assistant.message","data":{"content":"Mean 7.6."}}),
            json!({"type":"result","exitCode":0,"usage":{"sessionDurationMs":132679}}),
        ],
    );
    assert_eq!(s.models, ["claude-sonnet-5.5"]);
    assert_eq!(s.tool_calls["describe"], 1);
    assert_eq!(s.tool_calls["bash"], 1);
    assert_eq!(s.foreign_tool_calls, 1);
    assert_eq!(s.final_text.as_deref(), Some("Mean 7.6."));
    assert_eq!(s.duration_ms, Some(132679));
}

#[test]
fn event_lines_are_short() {
    let long = Event::ToolCall {
        server: Some("wsp".into()),
        tool: "set_col".into(),
        input: json!({ "values": vec![1.25f32; 500] }),
    };
    let line = long.to_string();
    assert!(line.starts_with("→ set_col {"));
    assert!(line.chars().count() < 140, "{line}");
}

#[test]
fn codex_gets_its_servers_as_overrides_a_read_only_sandbox_and_the_prompt_last() {
    let dir = std::env::temp_dir().join(format!("rlx-agent-codex-{}", std::process::id()));
    let r = run(AgentCli::Codex, &dir);
    let (program, a) = r.command_line("PROMPT");
    assert_eq!(program, "codex");
    assert_eq!(&a[..3], ["exec", "--json", "--skip-git-repo-check"]);
    assert_eq!(after(&a, "--sandbox"), ["read-only"]);
    assert_eq!(after(&a, "--cd"), [r.workspace.display().to_string()]);
    assert_eq!(
        after(&a, "-c"),
        [
            "mcp_servers.wsp.command=\"rlx-wsp\"",
            "mcp_servers.wsp.args=[\"mcp\"]",
            "mcp_servers.aux.command=\"aux-server\"",
            "mcp_servers.aux.args=[]",
        ]
    );
    assert_eq!(after(&a, "--model"), ["m"]);
    assert_eq!(a.last().unwrap(), "PROMPT");
    assert!(!a.contains(&"-p".to_string()));
}

#[test]
fn codex_events() {
    let lines = [
        json!({"type": "thread.started", "thread_id": "t"}),
        json!({"type": "item.started", "item": {"id": "1", "type": "mcp_tool_call", "server": "wsp",
               "tool": "describe", "arguments": {"x": 1}}}),
        json!({"type": "item.started", "item": {"id": "2", "type": "command_execution", "command": "ls"}}),
        json!({"type": "item.completed", "item": {"id": "3", "type": "agent_message", "text": "Kept it."}}),
        json!({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 3}}),
    ];
    let mut s = Summary::default();
    for l in &lines {
        for e in parse_event(AgentCli::Codex, l, &["wsp"]) {
            s.record(e);
        }
    }
    assert_eq!(s.tool_calls.get("describe"), Some(&1));
    assert_eq!(
        s.foreign_tool_calls, 1,
        "its own shell is outside the servers"
    );
    assert_eq!(s.final_text.as_deref(), Some("Kept it."));
    assert_eq!(s.num_turns, Some(1));
}

#[cfg(unix)]
#[test]
fn a_cli_past_its_timeout_is_stopped() {
    use std::os::unix::fs::PermissionsExt;
    let dir = std::env::temp_dir().join(format!("rlx-agent-timeout-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let fake = dir.join("slow-cli");
    std::fs::write(&fake, "#!/bin/sh\nsleep 60\n").unwrap();
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut r = AgentRun::new(AgentCli::Claude, &dir, vec![]);
    r.program = Some(fake);
    r.timeout = Some(std::time::Duration::from_millis(300));
    r.prepare().unwrap();
    let t0 = std::time::Instant::now();
    let out = r.run("hello", |_| {}).unwrap();
    assert!(out.timed_out);
    assert!(
        t0.elapsed() < std::time::Duration::from_secs(10),
        "{:?}",
        t0.elapsed()
    );
    std::fs::remove_dir_all(&dir).ok();
}
