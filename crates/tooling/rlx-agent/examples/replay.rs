// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Re-summarise a saved transcript:
//! `cargo run -p rlx-agent --example replay -- <claude|agent|copilot> <transcript.jsonl> [server..]`

use rlx_agent::{AgentCli, Summary, parse_event};
use std::io::BufRead;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(cli), Some(path)) = (args.first().and_then(|c| AgentCli::parse(c)), args.get(1))
    else {
        eprintln!("usage: replay <claude|agent|copilot> <transcript.jsonl> [server..]");
        std::process::exit(2);
    };
    let servers: Vec<&str> = args[2..].iter().map(String::as_str).collect();
    let file = std::fs::File::open(path).expect("open transcript");
    let mut summary = Summary::default();
    for line in std::io::BufReader::new(file).lines() {
        let Ok(v) = serde_json::from_str(&line.expect("read line")) else {
            continue;
        };
        for e in parse_event(cli, &v, &servers) {
            println!("{e}");
            summary.record(e);
        }
    }
    println!(
        "\nmodels {:?}\ntool calls {} {:?}\nforeign tool calls {}\nturns {:?} cost {:?} duration_ms {:?}",
        summary.models,
        summary.total_tool_calls(),
        summary.tool_calls,
        summary.foreign_tool_calls,
        summary.num_turns,
        summary.cost_usd,
        summary.duration_ms
    );
}
