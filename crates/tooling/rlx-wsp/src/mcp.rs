// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The weight-space tools as a **stdio MCP server** (`rlx-wsp mcp`).
//!
//! Newline-delimited JSON-RPC 2.0 on stdin/stdout, the transport every agent
//! CLI speaks (Claude Code `--mcp-config`, Cursor `.cursor/mcp.json`, Copilot
//! `--additional-mcp-config`). Implements `initialize`, `tools/list`,
//! `tools/call` and `ping`; notifications are accepted and ignored. Each tool
//! call is the same dispatch as [`handle_json`], and the state is saved after
//! every call so the driver can score it once the agent exits.

use crate::space::WeightSpace;
use crate::tools::{handle_json, tool_schemas};
use serde_json::{Value, json};
use std::io::{BufRead, Write};

const PROTOCOL_VERSION: &str = "2025-06-18";

const INSTRUCTIONS: &str = "Weight-space programming: read and write a neural network's \
weights directly. Call `describe` first. Reading and editing are free; `probe` and `rollout` \
each spend one forward pass from a fixed budget.";

/// Tools in MCP shape (`inputSchema`, camelCase).
pub fn mcp_tools() -> Value {
    let tools = tool_schemas();
    Value::Array(
        tools
            .as_array()
            .into_iter()
            .flatten()
            .map(|t| {
                json!({
                    "name": t["name"],
                    "description": t["description"],
                    "inputSchema": t["input_schema"],
                })
            })
            .collect(),
    )
}

/// Answer one JSON-RPC message; `None` for notifications.
pub fn handle_message(ws: &mut WeightSpace, msg: &Value) -> Option<Value> {
    let id = msg.get("id")?.clone();
    let method = msg["method"].as_str().unwrap_or("");
    let params = &msg["params"];
    let result = match method {
        "initialize" => Ok(json!({
            "protocolVersion": params["protocolVersion"].as_str().unwrap_or(PROTOCOL_VERSION),
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": { "name": "rlx-wsp", "version": env!("CARGO_PKG_VERSION") },
            "instructions": INSTRUCTIONS,
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": match &ws.state.tools {
            Some(list) => Value::Array(
                mcp_tools()
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|t| list.iter().any(|n| t["name"] == n.as_str()))
                    .cloned()
                    .collect(),
            ),
            None => mcp_tools(),
        } })),
        "tools/call" => call_tool(ws, params),
        _ => Err((-32601, format!("method not found: {method}"))),
    };
    Some(match result {
        Ok(r) => json!({ "jsonrpc": "2.0", "id": id, "result": r }),
        Err((code, message)) => json!({
            "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message }
        }),
    })
}

fn call_tool(ws: &mut WeightSpace, params: &Value) -> Result<Value, (i64, String)> {
    let name = params["name"]
        .as_str()
        .ok_or((-32602, "tools/call needs `name`".to_string()))?;
    let mut call = match &params["arguments"] {
        Value::Object(m) => m.clone(),
        Value::Null => Default::default(),
        _ => return Err((-32602, "`arguments` must be an object".into())),
    };
    call.insert("tool".into(), json!(name));
    let reply = handle_json(ws, &Value::Object(call).to_string());
    // Tool failures are results the model reads, not protocol errors.
    let (text, is_error) = match (reply.get("ok"), reply.get("error")) {
        (Some(ok), _) => (ok.to_string(), false),
        (_, Some(e)) => (e.as_str().unwrap_or("error").to_string(), true),
        _ => (reply.to_string(), true),
    };
    Ok(json!({ "content": [{ "type": "text", "text": text }], "isError": is_error }))
}

/// Serve until stdin closes; `save` runs after every tools/call.
pub fn serve(
    ws: &mut WeightSpace,
    mut save: impl FnMut(&WeightSpace) -> Result<(), String>,
) -> Result<(), String> {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => {
                let is_call = msg["method"] == "tools/call";
                let r = handle_message(ws, &msg);
                if is_call {
                    save(ws)?;
                }
                r
            }
            Err(e) => Some(json!({
                "jsonrpc": "2.0", "id": null,
                "error": { "code": -32700, "message": format!("parse error: {e}") }
            })),
        };
        if let Some(r) = reply {
            let mut out = stdout.lock();
            writeln!(out, "{r}").map_err(|e| e.to_string())?;
            out.flush().map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
