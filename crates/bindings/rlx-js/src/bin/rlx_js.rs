// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx-js` — run a JavaScript file against the RLX API.
//!
//! ```text
//! rlx-js script.js [args...]     run a file
//! rlx-js -e 'rlx.devices()'      run an expression, print its value
//! rlx-js -                       read the script from stdin
//! ```

use std::io::Read;
use std::process::ExitCode;

fn usage() -> ExitCode {
    eprintln!(
        "rlx-js {} — run JavaScript against the RLX API\n\
         \n\
         usage:\n  \
           rlx-js <script.js> [args...]   run a file\n  \
           rlx-js -e <source>             evaluate and print the result\n  \
           rlx-js -                       read the script from stdin\n\
         \n\
         `scriptArgs` holds [args...] in every mode (never the script path).\n\
         \n\
         backends in this build: {}",
        env!("CARGO_PKG_VERSION"),
        rlx_runtime::available_devices()
            .into_iter()
            .map(rlx_runtime::device_label)
            .collect::<Vec<_>>()
            .join(", "),
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(first) = args.first() else {
        return usage();
    };

    let mut runtime = rlx_js::Runtime::new();
    // A script run from a shell already has the user's authority, so the CLI
    // hands over file access that `Runtime::new` withholds by default.
    runtime.allow_filesystem();

    // `-e` prints the completion value; a file does not, so a script that
    // ends in an expression doesn't spray its last value over real output.
    let (source, print_result, script_args) = match first.as_str() {
        "-h" | "--help" => return usage(),
        "-e" | "--eval" => match args.get(1) {
            Some(src) => (src.clone(), true, &args[2.min(args.len())..]),
            None => return usage(),
        },
        "-" => {
            let mut buf = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
                eprintln!("rlx-js: reading stdin: {e}");
                return ExitCode::FAILURE;
            }
            (buf, false, &args[1..])
        }
        path => match std::fs::read_to_string(path) {
            Ok(src) => (src, false, &args[1..]),
            Err(e) => {
                eprintln!("rlx-js: {path}: {e}");
                return ExitCode::FAILURE;
            }
        },
    };

    bind_script_args(&mut runtime, script_args);

    match runtime.eval(&source) {
        Ok(value) => {
            if print_result {
                println!("{value}");
            }
            ExitCode::SUCCESS
        }
        Err(report) => {
            eprintln!("{report}");
            ExitCode::FAILURE
        }
    }
}

/// `scriptArgs` — the arguments the script was given, script path already
/// removed, identically in `-e`, `-` and file mode.
///
/// `qjs` puts the script path at index 0 and `-e` has nothing to put there, so
/// matching it would make index 0 mean two different things depending on how
/// the script was launched. One rule is easier to write against.
fn bind_script_args(runtime: &mut rlx_js::Runtime, args: &[String]) {
    use quickrs_core::object::PropFlags;
    use quickrs_core::value::Value;

    let ctx = runtime.context_mut();
    let values: Vec<Value> = args.iter().map(|a| Value::Str(ctx.intern(a))).collect();
    let array = ctx.new_array_from(values);
    let global = ctx.global();
    ctx.define_value(&global, "scriptArgs", Value::Object(array), PropFlags::C_W);
}
