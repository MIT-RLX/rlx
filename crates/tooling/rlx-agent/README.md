# rlx-agent

Run an installed **agent CLI** against MCP servers, with one event shape for
all of them. The LLM is the CLI; this crate makes no API requests.

```rust
use rlx_agent::{AgentCli, AgentRun, McpServer};

let mut run = AgentRun::new(AgentCli::Claude, "runs/1", vec![
    McpServer::new("wsp", "rlx-wsp", ["mcp", "--state", "runs/1/state.json"]),
]);
run.max_turns = Some(80);
run.prepare()?;                                   // run dir, workspace, MCP config
let out = run.run(prompt, |e| eprintln!("{e}"))?; // → describe {} · text … done in 53.2s
println!("{:?} {:?}", out.summary.tool_calls, out.summary.final_text);
```

| `AgentCli` | program | MCP config | tool restriction | output |
|---|---|---|---|---|
| `Claude` | `claude -p` | `--mcp-config` + `--strict-mcp-config` | `--tools ""` (no built-ins), `--allowedTools mcp__<server>`, `dontAsk` | stream-json |
| `Cursor` | `agent -p` | `<workspace>/.cursor/mcp.json`, `--approve-mcps` | shell stays available, sandboxed in the empty workspace | stream-json |
| `Copilot` | `copilot -p` | `--additional-mcp-config @mcp.json` | `--allow-tool <server>`, `shell` and `write` denied | JSONL |
| `Codex` | `codex exec --json` | `-c mcp_servers.<name>.command=… -c …args=[…]` | `--sandbox read-only`: its shell and patches cannot write | JSONL |

`model` goes to every CLI; `effort`, `max_turns` and `max_budget_usd` only to
Claude; `extra_args` are appended verbatim; `program` replaces the CLI's
executable (default: its name on `PATH`); `env` adds to its environment (for
example Codex's `CODEX_HOME`).

`run` launches the CLI in `<run_dir>/workspace`, saves stdout to
`<run_dir>/transcript.jsonl`, and returns a `Summary`: models used, calls per
tool (server prefixes stripped), calls to tools outside the listed servers
(attempted; the CLI may have denied them), the agent's last message, and
turns / cost / duration where the CLI reports them. With `timeout` set, a CLI
still running then is stopped together with what it started (on Unix it runs
in its own process group), and the outcome says `timed_out`.

The Cursor, Copilot and Codex launch lines and event parsing are checked by
tests (`tests/cli.rs`) but have not been run live; Claude has.

## The workspace server

`workspace::Workspace` is a stdio MCP server over one folder — the whole file
surface of an agent whose CLI runs without built-in tools: `list_files`,
`read_file`, `write_file`, `edit_file`. Every file in the folder can be read;
only the names given as writable can be written. Absolute paths and `..` are
refused.

```text
rlx-agent workspace-mcp --dir runs/1/workspace --write updater.json --write NOTES.md
```

`Workspace::server(name, command, prefix)` builds the `McpServer` entry that
launches it. `rlx-bind`'s agent optimizer (`runtime::agent`) uses it for every
decision; `pyrlx` serves the same server itself (`python -m pyrlx.workspace_mcp`).

Adding a CLI: one `AgentCli` variant, its arm in `command_line` (and in
`mcp_config` / `prepare` if its config shape differs), and its arm in
`events::parse_event`, with a fixture test in `tests/cli.rs`.

## License

MIT OR Apache-2.0.
