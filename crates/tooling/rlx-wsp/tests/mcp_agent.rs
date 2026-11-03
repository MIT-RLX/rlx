// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The MCP server answers the protocol an agent CLI speaks, and each agent
//! CLI gets a command line that exposes only the `wsp` tools.

use rlx_wsp::agent::{AgentCli, RunSpec, command_line, setup};
use rlx_wsp::mcp::handle_message;
use rlx_wsp::{Act, EnvKind, WeightSpace, tool_schemas};
use serde_json::{Value, json};

fn space() -> WeightSpace {
    WeightSpace::new(EnvKind::Bounce, &[32, 32], Act::Tanh, 2)
}

fn rpc(ws: &mut WeightSpace, id: i64, method: &str, params: Value) -> Value {
    handle_message(
        ws,
        &json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }),
    )
    .expect("request gets a reply")
}

#[test]
fn handshake_and_tool_list() {
    let mut ws = space();
    let init = rpc(
        &mut ws,
        1,
        "initialize",
        json!({ "protocolVersion": "2025-06-18", "capabilities": {} }),
    );
    assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
    assert!(init["result"]["capabilities"]["tools"].is_object());
    // Notifications get no reply.
    assert!(
        handle_message(
            &mut ws,
            &json!({ "jsonrpc": "2.0", "method": "notifications/initialized" })
        )
        .is_none()
    );
    let list = rpc(&mut ws, 2, "tools/list", json!({}));
    let tools = list["result"]["tools"].as_array().unwrap();
    assert_eq!(tools.len(), tool_schemas().as_array().unwrap().len());
    assert!(tools.iter().all(|t| t["inputSchema"]["type"] == "object"));
    let unknown = rpc(&mut ws, 3, "resources/list", json!({}));
    assert_eq!(unknown["error"]["code"], -32601);
}

#[test]
fn tool_calls_spend_the_budget_and_report_errors_in_band() {
    let mut ws = space();
    let call = |ws: &mut WeightSpace, id, name: &str, args: Value| {
        rpc(
            ws,
            id,
            "tools/call",
            json!({ "name": name, "arguments": args }),
        )["result"]
            .clone()
    };
    let r = call(&mut ws, 1, "rollout", json!({ "episodes": 32 }));
    assert_eq!(r["isError"], false);
    let report: Value = serde_json::from_str(r["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(report["episodes"], 32);
    assert_eq!(ws.state.used, 1);
    // A failing tool is a result the model reads, not a protocol error.
    let bad = call(
        &mut ws,
        2,
        "set",
        json!({ "name": "l0.w", "entries": [{ "row": "nope", "col": 0, "value": 1 }] }),
    );
    assert_eq!(bad["isError"], true);
    assert!(bad["content"][0]["text"].as_str().unwrap().contains("nope"));
    call(&mut ws, 3, "rollout", json!({}));
    let spent = call(&mut ws, 4, "rollout", json!({}));
    assert_eq!(spent["isError"], true);
}

fn spec(cli: AgentCli, out: &std::path::Path) -> RunSpec {
    RunSpec {
        cli,
        out: out.to_path_buf(),
        wsp_exe: "rlx-wsp".into(),
        model: Some("m".into()),
        effort: Some("high".into()),
        max_turns: 7,
        max_budget_usd: Some(2.5),
        extra_args: vec!["--extra".into()],
    }
}

#[test]
fn run_spec_wires_the_wsp_server_and_options() {
    // CLI-specific flags are tested in rlx-agent; here: the wsp server and
    // the run options reach the agent invocation.
    let out = std::env::temp_dir().join("rlx_wsp_cmd");
    let s = spec(AgentCli::Claude, &out);
    let run = s.agent_run();
    assert_eq!(run.servers.len(), 1);
    assert_eq!(run.servers[0].name, "wsp");
    assert_eq!(run.servers[0].args[..2], ["mcp", "--state"]);
    assert!(run.servers[0].args[2].ends_with("state.json"));
    let (prog, a) = command_line(&s, "P");
    assert_eq!(prog, "claude");
    let after = |flag: &str| a[a.iter().position(|x| x == flag).unwrap() + 1].clone();
    assert_eq!(after("--allowedTools"), "mcp__wsp");
    assert_eq!(after("--max-turns"), "7");
    assert_eq!(after("--model"), "m");
    assert_eq!(a.last().unwrap(), "--extra");
}

#[test]
fn setup_writes_a_fresh_run_and_refuses_to_reuse_it() {
    let out = std::env::temp_dir().join(format!("rlx_wsp_setup_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
    for cli in [AgentCli::Claude, AgentCli::Cursor, AgentCli::Copilot] {
        let dir = out.join(cli.program());
        let s = spec(cli, &dir);
        let prompt = setup(&s, &space()).unwrap();
        assert!(prompt.contains("budget of 2"));
        let cfg: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("mcp.json")).unwrap()).unwrap();
        assert_eq!(cfg["mcpServers"]["wsp"]["args"][0], "mcp");
        assert!(dir.join("workspace").is_dir());
        if cli == AgentCli::Cursor {
            assert!(dir.join("workspace/.cursor/mcp.json").is_file());
        }
        assert!(setup(&s, &space()).is_err());
    }
    let _ = std::fs::remove_dir_all(&out);
}
