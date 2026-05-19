// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Demonstrates plugging a new model family into the `rlx-run`
// dispatch surface from a downstream crate, without modifying
// anything in `rlx-models`. The pattern:
//
//   1. Implement `ModelRunner` for your model type.
//   2. `register_runner(Box::new(YourRunner));` early in `main`.
//   3. Call `dispatch(&argv)` — your subcommand shows up in
//      `dispatch_help()` and routes to `YourRunner::run` when the
//      first argv element matches your `name()`.
//
// Running:
//   cargo run --release -p rlx-models --example register_custom_runner -- echo hello world
//   cargo run --release -p rlx-models --example register_custom_runner -- help
//   cargo run --release -p rlx-models --example register_custom_runner -- adder 3 4
//
// The example registers two trivial runners (`echo`, `adder`) plus
// uses `dispatch` directly — no `rlx-run` binary involved. A real
// downstream tool would mix `register_runner` calls for its own
// model alongside `rlx_models::run::*` calls if it wants the
// built-in qwen3 / sam / dinov2 subcommands too (and would link
// against `rlx_models::bin::rlx_run`'s `register_builtins`, or
// just call the registration helpers manually).

use anyhow::{Context, Result};
use rlx_models::run::{ModelRunner, dispatch, dispatch_help, register_runner};

/// `echo <args...>` — prints its arguments back. Useful as a
/// no-deps sanity test of the dispatch path.
struct EchoRunner;
impl ModelRunner for EchoRunner {
    fn name(&self) -> &'static str {
        "echo"
    }
    fn description(&self) -> &'static str {
        "Print the supplied arguments back to stdout"
    }
    fn run(&self, args: &[String]) -> Result<()> {
        println!("echo: {}", args.join(" "));
        Ok(())
    }
}

/// `adder <a> <b>` — parses two integers and prints the sum.
/// Shows how a real runner does its own arg parsing inside `run`.
struct AdderRunner;
impl ModelRunner for AdderRunner {
    fn name(&self) -> &'static str {
        "adder"
    }
    fn description(&self) -> &'static str {
        "Add two integers (demonstrates per-runner arg parsing)"
    }
    fn run(&self, args: &[String]) -> Result<()> {
        let a: i64 = args
            .first()
            .context("adder: missing first integer")?
            .parse()
            .context("adder: first arg not an integer")?;
        let b: i64 = args
            .get(1)
            .context("adder: missing second integer")?
            .parse()
            .context("adder: second arg not an integer")?;
        println!("{a} + {b} = {}", a + b);
        Ok(())
    }
}

fn main() -> Result<()> {
    register_runner(Box::new(EchoRunner));
    register_runner(Box::new(AdderRunner));

    eprintln!("[register_custom_runner] registered subcommands:");
    eprintln!("{}", dispatch_help());

    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        eprintln!(
            "[register_custom_runner] no subcommand given; try `echo hi` or `adder 3 4` or `help`"
        );
        return Ok(());
    }
    dispatch(&argv)
}
