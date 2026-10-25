// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `rlx.splice` / `parallel` / `sequential` / `extract` / `rlx.Partition`
//! from script: combined and split graphs compute what the originals do.

use rlx_js::Runtime;

fn eval(source: &str) -> String {
    Runtime::new()
        .eval(source)
        .unwrap_or_else(|e| panic!("script failed:\n{e}"))
}

fn eval_err(source: &str) -> String {
    match Runtime::new().eval(source) {
        Ok(v) => panic!("expected a throw, got {v}"),
        Err(e) => e,
    }
}

/// `layer()` builds `y = x * w`; `run(g, feeds)` compiles `g` on CPU, sets
/// the params it has from `feeds`, and feeds the given inputs.
const PRELUDE: &str = r#"
function layer() {
  const g = new rlx.Graph("layer");
  const x = g.input("x", [3], "f32");
  const w = g.param("w", [3], "f32");
  g.setOutputs([g.mul(x, w)]);
  return g;
}
function run(g, params, inputs) {
  const c = new rlx.Session("cpu").compile(g.clone());
  for (const p of g.paramNames()) c.setParam(p, new Float32Array(params[p]));
  const feed = {};
  for (const k of Object.keys(inputs)) feed[k] = new Float32Array(inputs[k]);
  return c.run(feed).map((o) => Array.from(o));
}
"#;

fn script(body: &str) -> String {
    format!("{PRELUDE}\n{body}")
}

#[test]
fn splice_prefix_and_bind_stack_layers() {
    let out = eval(&script(
        r#"
        const g = new rlx.Graph("stack");
        let [h] = rlx.splice(g, layer(), { prefix: "l0." });
        [h] = rlx.splice(g, layer(), { prefix: "l1.", bind: { x: h } });
        g.setOutputs([h]);
        const [y] = run(g, { "l0.w": [2, 2, 2], "l1.w": [1, 2, 3] }, { "l0.x": [1, 2, 3] });
        g.paramNames().join(",") + "|" + y.join(",");
        "#,
    ));
    assert_eq!(out, "l0.w,l1.w|2,8,18");
}

#[test]
fn sequential_shares_and_parallel_concatenates() {
    let out = eval(&script(
        r#"
        // Same `w` twice: x * w * w.
        const s = rlx.sequential(layer(), layer());
        const [y] = run(s, { w: [2, 3, 4] }, { x: [1, 1, 1] });
        const p = rlx.parallel(layer(), s, "both");
        const [a, b] = run(p, { w: [2, 3, 4] }, { x: [1, 1, 1] });
        y.join(",") + "|" + a.join(",") + "|" + b.join(",") + "|" + p.name();
        "#,
    ));
    assert_eq!(out, "4,9,16|2,3,4|4,9,16|both");
}

#[test]
fn split_parts_run_in_order_and_merge_back() {
    let out = eval(&script(
        r#"
        const g = new rlx.Graph("stack");
        const [h0] = rlx.splice(g, layer(), { prefix: "l0." });
        const [h1] = rlx.splice(g, layer(), { prefix: "l1.", bind: { x: h0 } });
        g.setOutputs([h1]);
        const params = { "l0.w": [2, 2, 2], "l1.w": [1, 2, 3] };
        const [whole] = run(g, params, { "l0.x": [1, 2, 3] });

        const p = rlx.Partition.splitAt(g, [h0]);
        const env = { "l0.x": [1, 2, 3] };
        const outs = [];
        for (let i = 0; i < p.numParts(); i++) {
          const feed = {};
          for (const n of p.imports(i)) feed[n] = env[n];
          if (i === 0) feed["l0.x"] = env["l0.x"];
          const o = run(p.graph(i), params, feed);
          p.exports(i).forEach((n, j) => (env[n] = o[j]));
          outs.push(o);
        }
        const [[pi, j]] = p.outputs();
        const [merged] = run(p.merge(), params, { "l0.x": [1, 2, 3] });
        [whole.join(","), outs[pi][j].join(","), merged.join(","), p.numParts()].join("|");
        "#,
    ));
    assert_eq!(out, "2,8,18|2,8,18|2,8,18|2");
}

#[test]
fn components_and_extract_split_independent_outputs() {
    let out = eval(&script(
        r#"
        const g = new rlx.Graph("two");
        const x = g.input("x", [3], "f32");
        const a = g.exp(x);
        const b = g.mul(x, x);
        g.setOutputs([a, b]);
        const c = rlx.Partition.components(g);
        const sq = rlx.extract(g, [b]);
        [c.numParts(), c.imports(1).length, JSON.stringify(c.outputs()),
         run(sq, {}, { x: [1, 2, 3] })[0].join(",")].join("|");
        "#,
    ));
    assert_eq!(out, "2|0|[[0,0],[1,0]]|1,4,9");
}

#[test]
fn errors_name_the_problem() {
    let e = eval_err(&script(
        r#"const t = new rlx.Graph("t");
        rlx.splice(t, layer(), { bind: { y: t.input("z", [3], "f32") } })"#,
    ));
    assert!(e.contains("\"y\""), "{e}");
    let e = eval_err(&script(
        r#"rlx.splice(new rlx.Graph("t"), layer(), { prefx: "a." })"#,
    ));
    assert!(e.contains("prefx"), "{e}");
    let e = eval_err(&script(r#"rlx.Partition.components(layer()).graph(5)"#));
    assert!(e.contains("part 5"), "{e}");
    let e = eval_err(&script(
        r#"rlx.sequential(rlx.parallel(layer(), layer()), layer())"#,
    ));
    assert!(e.contains("free inputs"), "{e}");
}
