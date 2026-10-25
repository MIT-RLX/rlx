// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Compiled.inputs()` / `params()` / `outputs()` / `outputIndex()` /
//! `checkBindings()` / `allowUnbound()` and the named errors for a
//! misspelled or wrong-length input, from JS.

use rlx_js::Runtime;

fn eval(source: &str) -> String {
    Runtime::new()
        .eval(source)
        .unwrap_or_else(|e| panic!("script failed:\n{e}"))
}

const SETUP: &str = r#"
    const g = new rlx.Graph("mlp");
    const x = g.input("tokens", [2, 4], "f32");
    const w = g.param("w", [4, 3], "f32");
    g.setOutputs([g.relu(g.matmul(x, w))]);
    const c = new rlx.Session("cpu").compile(g);
"#;

#[test]
fn describes_inputs_and_params() {
    let out = eval(&format!("{SETUP} JSON.stringify([c.inputs(), c.params()])"));
    assert_eq!(
        out,
        r#"[[{"dtype":"f32","name":"tokens","shape":[2,4]}],[{"dtype":"f32","name":"w","shape":[4,3]}]]"#
    );
}

#[test]
fn mistakes_throw_with_the_name() {
    let out = eval(&format!(
        r#"{SETUP}
        const msgs = [];
        try {{ c.checkBindings(["tokens"]); }} catch (e) {{ msgs.push(String(e.message)); }}
        c.setParam("w", new Float32Array(12).fill(1));
        try {{ c.run({{ token: new Float32Array(8) }}); }} catch (e) {{ msgs.push(String(e.message)); }}
        try {{ c.run({{ tokens: new Float32Array(3) }}); }} catch (e) {{ msgs.push(String(e.message)); }}
        c.allowUnbound(["tokens"]);
        c.checkBindings([]);
        msgs.join("\n")"#
    ));
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 3, "{out}");
    assert!(lines[0].contains("unbound params `w`"), "{out}");
    assert!(lines[1].contains("did you mean `tokens`"), "{out}");
    assert!(lines[2].contains("holds 8"), "{out}");
}

#[test]
fn outputs_are_found_by_name() {
    let out = eval(
        r#"
        const g = new rlx.Graph("two");
        const x = g.input("tokens", [2, 4], "f32");
        const w = g.param("w", [4, 3], "f32");
        const h = g.matmul(x, w);
        g.setOutputs([g.relu(h), h]);
        g.nameOutput(0, "probs");
        g.nameOutput(1, "hidden");
        const names = g.outputNames();
        const c = new rlx.Session("cpu").compile(g);
        let miss = "";
        try { c.outputIndex("hiden"); } catch (e) { miss = String(e.message); }
        JSON.stringify([names, c.outputs().map(o => o.name), c.outputIndex("hidden"), miss])"#,
    );
    assert_eq!(
        out,
        r#"[["probs","hidden"],["probs","hidden"],1,"rlx: the graph has no output named `hiden` (did you mean `hidden`?)"]"#
    );
}
