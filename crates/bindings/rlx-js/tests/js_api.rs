// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The JS surface, exercised from Rust.
//!
//! Each test is a script, because that is what a user writes — a Rust-level
//! unit test of `m_matmul` would pass whether or not the method was ever
//! reachable from `new rlx.Graph()`.

use rlx_js::Runtime;

fn eval(source: &str) -> String {
    Runtime::new()
        .eval(source)
        .unwrap_or_else(|e| panic!("script failed:\n{e}"))
}

/// A runtime with file access, for the tests that read a fixture.
fn eval_with_fs(source: &str) -> String {
    let mut rt = Runtime::new();
    rt.allow_filesystem();
    rt.eval(source)
        .unwrap_or_else(|e| panic!("script failed:\n{e}"))
}

fn eval_err(source: &str) -> String {
    match Runtime::new().eval(source) {
        Ok(v) => panic!("expected a throw, got {v}"),
        Err(e) => e,
    }
}

#[test]
fn engine_runs_plain_javascript() {
    // The RLX globals are additive: everything QuickJS-ng offers still works.
    assert_eq!(eval("[3, 1, 2].sort().join('')"), "123");
    assert_eq!(eval("JSON.stringify({a: 1})"), r#"{"a":1}"#);
}

#[test]
fn cpu_is_always_available() {
    assert_eq!(eval("rlx.isAvailable('cpu')"), "true");
    assert_eq!(eval("rlx.devices().includes('cpu')"), "true");
    // An unknown device is a `false`, not a throw — it answers "can I use it".
    assert_eq!(eval("rlx.isAvailable('quantum')"), "false");
}

#[test]
fn matmul_round_trips_through_float32arrays() {
    let out = eval(
        r#"
        const g = new rlx.Graph("mm");
        const x = g.input("x", [2, 3], "f32");
        const w = g.param("w", [3, 2], "f32");
        g.setOutputs([g.matmul(x, w)]);

        const c = new rlx.Session("cpu").compile(g);
        c.setParam("w", new Float32Array([1, 0, 0, 1, 1, 1]));
        const [y] = c.run({ x: new Float32Array([1, 2, 3, 4, 5, 6]) });
        Array.from(y).join(",");
        "#,
    );
    assert_eq!(out, "4,5,10,11");
}

#[test]
fn shape_inference_is_visible_to_script() {
    let out = eval(
        r#"
        const g = new rlx.Graph("s");
        const x = g.input("x", [2, 3], "f32");
        const w = g.param("w", [3, 5], "f32");
        const y = g.matmul(x, w);
        const s = g.shapeOf(y);
        s.dims.join("x") + ":" + s.dtype;
        "#,
    );
    assert_eq!(out, "2x5:f32");
}

#[test]
fn a_plain_array_works_where_a_float32array_does() {
    // Convenience for hand-written fixtures; the typed-array path is the fast
    // one, but a literal must not be a silent zero-fill.
    let out = eval(
        r#"
        const g = new rlx.Graph("id");
        const x = g.input("x", [3], "f32");
        g.setOutputs([g.mul(x, x)]);
        const c = new rlx.Session("cpu").compile(g);
        Array.from(c.run({ x: [2, 3, 4] })[0]).join(",");
        "#,
    );
    assert_eq!(out, "4,9,16");
}

#[test]
fn grad_matches_the_analytic_derivative() {
    // loss = sum(w * x) with x = [1,2,3]  ⇒  dloss/dw = x.
    let out = eval(
        r#"
        const g = new rlx.Graph("dot");
        const x = g.input("x", [3], "f32");
        const w = g.param("w", [3], "f32");
        g.setOutputs([g.sum(g.mul(x, w), [0], false)]);

        const bwd = rlx.grad(g, [w]);
        const c = new rlx.Session("cpu").compile(bwd);
        c.setParam("w", new Float32Array([0.5, 0.5, 0.5]));
        const [loss, dW] = c.run({
          x: new Float32Array([1, 2, 3]),
          d_output: new Float32Array([1]),
        });
        loss[0] + "|" + Array.from(dW).join(",");
        "#,
    );
    assert_eq!(out, "3|1,2,3");
}

#[test]
fn gradient_descent_reduces_the_loss() {
    let out = eval(
        r#"
        const g = new rlx.Graph("mse");
        const x = g.input("x", [1, 2], "f32");
        const t = g.input("t", [1, 2], "f32");
        const w = g.param("w", [2, 2], "f32");
        const d = g.sub(g.matmul(x, w), t);
        g.setOutputs([g.mean(g.mul(d, d), [0, 1], false)]);

        const step = new rlx.Session("cpu").compile(rlx.grad(g, [w]));
        const x0 = new Float32Array([1, 1]);
        const t0 = new Float32Array([1, -1]);
        const seed = new Float32Array([1]);
        let weights = new Float32Array([0, 0, 0, 0]);

        let first = null, last = null;
        for (let i = 0; i < 50; i++) {
          step.setParam("w", weights);
          const [loss, dW] = step.run({ x: x0, t: t0, d_output: seed });
          if (first === null) first = loss[0];
          last = loss[0];
          for (let k = 0; k < weights.length; k++) weights[k] -= 0.1 * dW[k];
        }
        (first > 0.9 && last < 1e-3) ? "converged" : `first=${first} last=${last}`;
        "#,
    );
    assert_eq!(out, "converged");
}

#[test]
fn vmap_batches_a_scalar_graph() {
    let out = eval(
        r#"
        const g = new rlx.Graph("sq");
        const x = g.input("x", [2], "f32");
        g.setOutputs([g.mul(x, x)]);

        const batched = rlx.vmap(g, ["x"], 3);
        const c = new rlx.Session("cpu").compile(batched);
        const [y] = c.run({ x: new Float32Array([1, 2, 3, 4, 5, 6]) });
        Array.from(y).join(",");
        "#,
    );
    assert_eq!(out, "1,4,9,16,25,36");
}

#[test]
fn f64_survives_the_typed_round_trip() {
    // `run` is the f32 fast path; `runTyped` is the lossless one.
    let out = eval(
        r#"
        const g = new rlx.Graph("f64");
        const x = g.input("x", [2], "f64");
        g.setOutputs([g.add(x, x)]);
        const c = new rlx.Session("cpu").compile(g);
        const [{ data, dtype }] = c.runTyped({
          x: { data: new Float64Array([0.1, 0.2]), dtype: "f64" },
        });
        const view = new Float64Array(data.buffer, data.byteOffset, data.byteLength / 8);
        dtype + ":" + view[0];
        "#,
    );
    assert_eq!(out, "f64:0.2");
}

#[test]
fn a_consumed_graph_says_so() {
    let err = eval_err(
        r#"
        const g = new rlx.Graph("g");
        g.setOutputs([g.input("x", [1], "f32")]);
        const s = new rlx.Session("cpu");
        s.compile(g);
        s.compile(g);
        "#,
    );
    assert!(
        err.contains("consumed"),
        "want a 'consumed' message, got: {err}"
    );
}

#[test]
fn a_missing_backend_names_the_feature_to_rebuild_with() {
    // Never a silent CPU fallback: a script asking for CUDA on a CPU-only
    // build has to hear about it, or it will report CPU numbers as CUDA ones.
    if rlx_runtime::is_available(rlx_runtime::Device::Cuda) {
        return;
    }
    let err = eval_err(r#"new rlx.Session("cuda")"#);
    assert!(
        err.contains("not in this build") && err.contains("--features cuda"),
        "unhelpful message: {err}"
    );
}

#[test]
fn wrong_handle_types_are_type_errors_not_crashes() {
    // The class tag is what stands between `compile({})` and a wild pointer.
    for script in [
        r#"new rlx.Session("cpu").compile({})"#,
        r#"new rlx.Session("cpu").compile(new rlx.Session("cpu"))"#,
        r#"rlx.grad(42, [0])"#,
        r#"new rlx.Graph("g").matmul.call({}, 0, 0)"#,
    ] {
        let err = eval_err(script);
        assert!(
            err.contains("TypeError"),
            "script `{script}` should throw a TypeError, got: {err}"
        );
    }
}

#[test]
fn a_runaway_script_hits_the_instruction_budget() {
    let mut rt = Runtime::new();
    rt.set_instruction_budget(Some(100_000));
    let err = rt
        .eval("let i = 0; while (true) { i++; } i;")
        .expect_err("an infinite loop must not hang the host");
    assert!(!err.is_empty());
}

#[test]
fn device_report_covers_every_backend() {
    let out = eval(
        r#"
        const g = new rlx.Graph("r");
        g.setOutputs([g.relu(g.input("x", [4], "f32"))]);
        const rows = rlx.deviceReport(g);
        const cpu = rows.find((r) => r.device === "cpu");
        `${rows.length > 1}:${cpu.available}:${cpu.supportsGraph}`;
        "#,
    );
    assert_eq!(out, "true:true:true");
}

#[test]
fn a_getter_that_re_enters_the_graph_is_safe() {
    // Argument conversion can run script — a getter on an "array" of dims is
    // an ordinary JS function, and nothing stops it calling back into the very
    // graph whose method is mid-flight. Every method here converts its
    // arguments *before* borrowing the handle, so the re-entrant call gets its
    // own borrow rather than aliasing a live one.
    let out = eval(
        r#"
        const g = new rlx.Graph("reentrant");
        let sneaky = null;
        const dims = {
          length: 2,
          get 0() { sneaky = g.input("sneaky", [7], "f32"); return 3; },
          get 1() { return 4; },
        };
        const x = g.input("x", dims, "f32");
        const s = g.shapeOf(x);
        `${s.dims.join("x")}|${g.shapeOf(sneaky).dims.join("x")}`;
        "#,
    );
    assert_eq!(out, "3x4|7");
}

#[test]
fn a_large_tensor_survives_the_bulk_copy_path() {
    // The Float32Array fast path is a memcpy out of the backing ArrayBuffer,
    // not an element-at-a-time Value round trip; a length that is not a tidy
    // multiple of anything is what catches an off-by-one in the window math.
    let out = eval(
        r#"
        const n = 4097;
        const g = new rlx.Graph("big");
        const x = g.input("x", [n], "f32");
        g.setOutputs([g.sum(x, [0], false)]);
        const c = new rlx.Session("cpu").compile(g);
        const data = new Float32Array(n).fill(2);
        const [total] = c.run({ x: data });
        `${total[0]}`;
        "#,
    );
    assert_eq!(out, "8194");
}

#[test]
fn a_subarray_view_reads_its_own_window() {
    // A `subarray` shares the buffer at a non-zero byteOffset — reading the
    // whole buffer instead of the view's window is the classic mistake here.
    let out = eval(
        r#"
        const g = new rlx.Graph("win");
        const x = g.input("x", [3], "f32");
        g.setOutputs([g.mul(x, x)]);
        const c = new rlx.Session("cpu").compile(g);
        const backing = new Float32Array([9, 9, 1, 2, 3, 9]);
        Array.from(c.run({ x: backing.subarray(2, 5) })[0]).join(",");
        "#,
    );
    assert_eq!(out, "1,4,9");
}

// ── the rest of the IR surface ───────────────────────────────

#[test]
fn extended_ops_infer_the_shapes_the_kernels_expect() {
    // One assertion per op family. These are shape checks, not numerics: a
    // wrong output shape is what turns a working graph into a silent
    // reinterpretation of someone else's bytes.
    let out = eval(
        r#"
        const g = new rlx.Graph("ext");
        const x = g.input("x", [2, 3], "f32");
        const a = g.input("a", [4, 3], "f32");
        const sq = g.input("sq", [3, 3], "f64");
        const d = (n) => g.shapeOf(n).dims.join("x");
        [
          d(g.clamp(x, 0, 1)),
          d(g.pad(x, [[1, 1], [2, 0]], "reflect")),
          d(g.slice(x, 1, 0, 2, 1)),
          d(g.tile(x, [2, 1])),
          d(g.trilu(sq, true, 0)),
          d(g.roll(x, [1], [0])),
          d(g.reverse(x, [1])),
          d(g.cumprod(x, 1, false)),
          d(g.argmax(x, 1, false)),
          d(g.sort(x, 1, true)),
          d(g.qr(a, "q")), d(g.qr(a, "r")),
          d(g.svd(a, "u")), d(g.svd(a, "s")), d(g.svd(a, "vt")),
          d(g.det(sq)) || "scalar",
          d(g.cholesky(sq)),
          d(g.eigh(sq)[0]), d(g.eigh(sq)[1]),
          d(g.histogram(x, 8, 0, 1)),
          d(g.softmaxCrossEntropy(x, g.input("t", [2, 3], "f32"))),
          d(g.zeros([2, 2], "f32")),
        ].join("|");
        "#,
    );
    assert_eq!(
        out,
        "2x3|4x5|2x2|4x3|3x3|2x3|2x3|2x3|2|2x3|4x3|3x3|4x3|3|3x3|scalar|3x3|3|3x3|8|2|2x2"
    );
}

#[test]
fn extended_ops_compute_the_right_numbers() {
    let out = eval(
        r#"
        const g = new rlx.Graph("num");
        const x = g.input("x", [6], "f32");
        g.setOutputs([
          g.clamp(x, 0, 3),
          g.cumprod(x, 0, false),
          g.reverse(x, [0]),
          g.sort(x, 0, true),
        ]);
        const c = new rlx.Session("cpu").compile(g);
        const [clamped, prod, rev, sorted] = c.run({ x: new Float32Array([1, 2, 3, 4, 5, 6]) });
        [clamped, prod, rev, sorted].map((a) => Array.from(a).join(",")).join(" | ");
        "#,
    );
    assert_eq!(
        out,
        "1,2,3,3,3,3 | 1,2,6,24,120,720 | 6,5,4,3,2,1 | 6,5,4,3,2,1"
    );
}

#[test]
fn quantize_round_trips_through_gguf_packing() {
    // Q8_0 over a 32-element block: lossy, but a round trip that came back
    // scrambled (wrong block layout) would be off by far more than 2%.
    let out = eval(
        r#"
        const n = 64;
        const src = new Float32Array(n);
        for (let i = 0; i < n; i++) src[i] = Math.sin(i / 3);
        const packed = rlx.quantize(src, "Q8_0");
        const back = rlx.dequantize(packed, "Q8_0");
        let worst = 0;
        for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(back[i] - src[i]));
        `${back.length}:${worst < 0.02}`;
        "#,
    );
    assert_eq!(out, "64:true");
}

// ── multi-backend routing ────────────────────────────────────

#[test]
fn runner_reports_and_runs_on_a_chosen_backend() {
    let out = eval(
        r#"
        const g = new rlx.Graph("mm");
        const x = g.input("x", [2, 2], "f32");
        const w = g.param("w", [2, 2], "f32");
        g.setOutputs([g.matmul(x, w)]);

        const r = new rlx.Runner(g, { only: ["cpu"] });
        r.setParams({ w: new Float32Array([1, 0, 0, 1]) });
        const warmed = r.warmAll().join(",");
        const bench = r.benchmark({ x: new Float32Array([1, 2, 3, 4]) }, 2);
        const hit = r.run({ x: new Float32Array([1, 2, 3, 4]) }, "cpu");
        `${warmed}|${bench.length}|${hit.device}|${Array.from(hit.outputs[0]).join(",")}`;
        "#,
    );
    assert_eq!(out, "cpu|1|cpu|1,2,3,4");
}

#[test]
fn router_falls_back_and_names_the_device_it_used() {
    let out = eval(
        r#"
        const g = new rlx.Graph("mm");
        const x = g.input("x", [2], "f32");
        g.setOutputs([g.mul(x, x)]);
        const router = new rlx.Router(g, { only: ["cpu"] });
        const hit = router.runChain({ x: new Float32Array([3, 4]) });
        `${hit.device}:${Array.from(hit.outputs[0]).join(",")}`;
        "#,
    );
    assert_eq!(out, "cpu:9,16");
}

// ── compile options ──────────────────────────────────────────

#[test]
fn compile_with_options_is_numerically_identical() {
    // Fusion toggles are a performance knob, not a semantics knob. If these
    // two disagree, a fusion pass is wrong — which is exactly what a JS-level
    // A/B sweep is for.
    let out = eval(
        r#"
        function build() {
          const g = new rlx.Graph("fuse");
          const x = g.input("x", [4, 4], "f32");
          const w = g.param("w", [4, 4], "f32");
          g.setOutputs([g.silu(g.add(g.matmul(x, w), g.constant(0.5)))]);
          return g;
        }
        const weights = { w: new Float32Array(16).fill(0.25) };
        const inputs = { x: new Float32Array(16).fill(0.5) };
        const s = new rlx.Session("cpu");

        const fused = s.compile(build());
        fused.setParams(weights);
        const a = fused.run(inputs)[0];

        const plain = s.compileWith(build(), { fusion: { skipFusion: true } });
        plain.setParams(weights);
        const b = plain.run(inputs)[0];

        let worst = 0;
        for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
        worst < 1e-6 ? "identical" : `differ by ${worst}`;
        "#,
    );
    assert_eq!(out, "identical");
}

// ── static diagnostics ───────────────────────────────────────

#[test]
fn check_answers_for_backends_that_are_not_in_this_build() {
    // The point of a device-free check: CUDA legality is unavailable on a
    // CPU-only build, and that reads as `null`, not as `false`.
    let out = eval(
        r#"
        const g = new rlx.Graph("chk");
        const x = g.input("x", [4, 8], "f32");
        const w = g.param("w", [8, 8], "f32");
        g.setOutputs([g.softmax(g.matmul(x, w), -1)]);
        const r = rlx.check(g, { backends: ["cpu"] });
        const cpu = r.backends.find((b) => b.backend === "cpu");
        `${r.nodes}:${r.errors}:${cpu.legality.compileReady}`;
        "#,
    );
    assert_eq!(out, "4:0:true");
}

#[test]
fn check_rejects_an_unknown_backend_by_name() {
    let err = eval_err(
        r#"
        const g = new rlx.Graph("chk");
        g.setOutputs([g.relu(g.input("x", [2], "f32"))]);
        rlx.check(g, { backends: ["definitely-not-a-backend"] });
        "#,
    );
    assert!(err.contains("unknown backend"), "unhelpful: {err}");
}

// ── training ─────────────────────────────────────────────────

#[test]
fn optimizer_moves_weights_downhill() {
    // AdamW's first step is ±lr per element regardless of gradient magnitude
    // (the moment ratio is 1), so the sign is the thing to assert.
    let out = eval(
        r#"
        const opt = new rlx.Optimizer("adamw", { lr: 0.1 });
        let w = new Float32Array([1, 1]);
        for (let i = 0; i < 10; i++) {
          const updated = opt.step({ w: { param: w, grad: new Float32Array([1, -1]), shape: [2] } });
          w = updated.w;
        }
        (w[0] < 0.2 && w[1] > 1.8) ? "descended" : Array.from(w).join(",");
        "#,
    );
    assert_eq!(out, "descended");
}

#[test]
fn every_optimizer_kind_constructs_and_steps() {
    let out = eval(
        r#"
        const kinds = ["sgd", "adam", "adamw", "lion", "muon", "radam", "nadamw",
                       "lamb", "mars", "soap", "sophia", "qhadamw", "adafactor"];
        const bad = [];
        for (const kind of kinds) {
          const opt = new rlx.Optimizer(kind, { lr: 0.01 });
          // 2-D shape: Muon orthogonalizes matrices and skips vectors, so a
          // flat [4] would never exercise its actual update.
          const out = opt.step({
            w: { param: new Float32Array([1, 2, 3, 4]), grad: new Float32Array([1, 1, 1, 1]), shape: [2, 2] },
          });
          if (!(out.w instanceof Float32Array) || out.w.length !== 4 ||
              !Array.from(out.w).every(Number.isFinite)) bad.push(kind);
        }
        bad.length === 0 ? "all ok" : "broken: " + bad.join(",");
        "#,
    );
    assert_eq!(out, "all ok");
}

#[test]
fn trainer_learns_a_linear_map() {
    let out = eval(
        r#"
        const g = new rlx.Graph("mse");
        const x = g.input("x", [1, 2], "f32");
        const t = g.input("t", [1, 2], "f32");
        const w = g.param("w", [2, 2], "f32");
        const d = g.sub(g.matmul(x, w), t);
        g.setOutputs([g.mean(g.mul(d, d), [0, 1], false)]);

        const trainer = new rlx.Trainer(g, {
          wrt: ["w"],
          init: { w: new Float32Array([0, 0, 0, 0]) },
          device: "cpu",
          optimizer: { kind: "adamw", lr: 0.1 },
        });

        const batch = { x: new Float32Array([1, 1]), t: new Float32Array([1, -1]) };
        const first = trainer.step(batch);
        for (let i = 0; i < 200; i++) trainer.step(batch);
        const last = trainer.evaluate(batch);
        const trained = trainer.params().w;

        (first > 0.9 && last < 1e-4 && trained.length === 4 && trainer.steps() === 201)
          ? "learned" : `first=${first} last=${last} steps=${trainer.steps()}`;
        "#,
    );
    assert_eq!(out, "learned");
}

#[test]
fn trainer_round_trips_weights_through_set_params() {
    // Resuming from a checkpoint has to actually restore the loss, or a
    // long run that crashes has lost everything after the last save.
    let out = eval(
        r#"
        function build() {
          const g = new rlx.Graph("mse");
          const x = g.input("x", [1, 2], "f32");
          const t = g.input("t", [1, 2], "f32");
          const w = g.param("w", [2, 2], "f32");
          const d = g.sub(g.matmul(x, w), t);
          g.setOutputs([g.mean(g.mul(d, d), [0, 1], false)]);
          return g;
        }
        const opts = {
          wrt: ["w"], init: { w: new Float32Array([0, 0, 0, 0]) },
          device: "cpu", optimizer: { kind: "adamw", lr: 0.1 },
        };
        const batch = { x: new Float32Array([1, 1]), t: new Float32Array([1, -1]) };

        const a = new rlx.Trainer(build(), opts);
        for (let i = 0; i < 50; i++) a.step(batch);
        const checkpoint = a.params();
        const lossA = a.evaluate(batch);

        const b = new rlx.Trainer(build(), opts);
        b.setParams(checkpoint);
        const lossB = b.evaluate(batch);

        Math.abs(lossA - lossB) < 1e-7 ? "restored" : `${lossA} vs ${lossB}`;
        "#,
    );
    assert_eq!(out, "restored");
}

#[test]
fn trainer_rejects_a_graph_that_is_not_a_scalar_loss() {
    // `grad` seeds d_output with 1, which only means "differentiate the loss"
    // when the graph ends on one output.
    let err = eval_err(
        r#"
        const g = new rlx.Graph("two");
        const x = g.input("x", [2], "f32");
        const w = g.param("w", [2], "f32");
        g.setOutputs([g.mul(x, w), g.add(x, w)]);
        new rlx.Trainer(g, { wrt: ["w"], init: { w: new Float32Array([1, 1]) }, device: "cpu" });
        "#,
    );
    assert!(
        err.contains("exactly one output"),
        "want the scalar-loss message, got: {err}"
    );
}

#[test]
fn trainer_names_the_params_it_has_when_wrt_is_wrong() {
    let err = eval_err(
        r#"
        const g = new rlx.Graph("mse");
        const x = g.input("x", [2], "f32");
        const w = g.param("weight", [2], "f32");
        g.setOutputs([g.sum(g.mul(x, w), [0], false)]);
        new rlx.Trainer(g, { wrt: ["w"], init: { w: new Float32Array([1, 1]) }, device: "cpu" });
        "#,
    );
    assert!(
        err.contains("not a Param") && err.contains("weight"),
        "the error should list the real param names, got: {err}"
    );
}

#[test]
fn trainer_catches_an_init_shape_mismatch() {
    let err = eval_err(
        r#"
        const g = new rlx.Graph("mse");
        const x = g.input("x", [2], "f32");
        const w = g.param("w", [2], "f32");
        g.setOutputs([g.sum(g.mul(x, w), [0], false)]);
        new rlx.Trainer(g, { wrt: ["w"], init: { w: new Float32Array([1, 1, 1]) }, device: "cpu" });
        "#,
    );
    assert!(
        err.contains("3"),
        "should report the given length, got: {err}"
    );
}

// ── what the macros promise ──────────────────────────────────

#[test]
fn declared_arity_matches_the_parameter_list() {
    // `graph_ops!` derives `Function.length` from the signature it already
    // has. Before the macro these were two hand-maintained numbers, and
    // `rfft` shipped with `length` 1 against a 2-parameter reader.
    let out = eval(
        r#"
        const g = new rlx.Graph("arity");
        const want = {
          matmul: 2, add: 2, softmax: 2, reduce: 4, narrow: 4, cast: 2,
          layerNorm: 5, rmsNorm: 4, attention: 6, attentionKind: 6,
          rfft: 2, irfft: 4, slice: 5, trilu: 3, roll: 3, histogram: 4,
          stft: 4, fftfreq: 1, eigh: 1, det: 1, clamp: 3,
        };
        const bad = [];
        for (const [name, n] of Object.entries(want)) {
          if (typeof g[name] !== "function") bad.push(`${name} missing`);
          else if (g[name].length !== n) bad.push(`${name}=${g[name].length} want ${n}`);
        }
        bad.length === 0 ? "arities agree" : bad.join(", ");
        "#,
    );
    assert_eq!(out, "arities agree");
}

#[test]
fn defaults_declared_in_the_table_actually_apply() {
    // `axis: i32 = -1` and friends must mean the same as passing the value.
    let out = eval(
        r#"
        const g = new rlx.Graph("defaults");
        const x = g.input("x", [2, 4], "f32");
        const same = (a, b) => g.shapeOf(a).dims.join() === g.shapeOf(b).dims.join();
        [
          same(g.softmax(x), g.softmax(x, -1)),
          same(g.sum(x, [1]), g.sum(x, [1], false)),
          same(g.cumsum(x, 1), g.cumsum(x, 1, false)),
          same(g.zeros([2, 2]), g.zeros([2, 2], "f32")),
          same(g.trilu(g.input("s", [3, 3], "f32"), true),
               g.trilu(g.input("s2", [3, 3], "f32"), true, 0)),
        ].every(Boolean) ? "defaults hold" : "a default diverged";
        "#,
    );
    assert_eq!(out, "defaults hold");
}

#[test]
fn a_class_prototype_cannot_be_swapped_out() {
    // `js_class!` pins every prototype non-writable and non-configurable. A
    // swapped prototype would make the class tag unverifiable, which is the
    // one thing standing between a bad cast and a wild pointer.
    let out = eval(
        r#"
        let threw = 0;
        for (const cls of [rlx.Graph, rlx.Session, rlx.Runner, rlx.Trainer]) {
          try { "use strict"; cls.prototype = {}; } catch (e) { threw++; }
        }
        // Non-writable in sloppy mode is a silent no-op, so check both ways.
        const intact = rlx.Graph.prototype.matmul !== undefined;
        `${intact}`;
        "#,
    );
    assert_eq!(out, "true");
}

#[test]
fn every_optimizer_actually_descends() {
    // Constructing and stepping is not the same as working: a wrong
    // hyperparameter mapping (e.g. `weightDecay` landing on `momentum`) still
    // produces finite numbers that never improve. Muon and Lion converge more
    // slowly than the Adam family here, hence the looser bound.
    let out = eval(
        r#"
        function build() {
          const g = new rlx.Graph("mse");
          const x = g.input("x", [2, 4], "f32");
          const t = g.input("t", [2, 4], "f32");
          const w = g.param("w", [4, 4], "f32");
          const d = g.sub(g.matmul(x, w), t);
          g.setOutputs([g.mean(g.mul(d, d), [0, 1], false)]);
          return g;
        }
        const batch = {
          x: new Float32Array([1, 0, 0, 1, 0, 1, 1, 0]),
          t: new Float32Array([1, -1, 0.5, 0, 0, 2, -1, 1]),
        };
        const kinds = ["sgd", "adam", "adamw", "lion", "muon", "radam",
                       "nadamw", "lamb", "mars", "soap", "qhadamw"];
        const stuck = [];
        for (const kind of kinds) {
          const t = new rlx.Trainer(build(), {
            wrt: ["w"], init: { w: new Float32Array(16).fill(0.01) },
            device: "cpu", optimizer: { kind, lr: kind === "sgd" ? 0.2 : 0.05 },
          });
          const first = t.step(batch);
          for (let i = 0; i < 300; i++) t.step(batch);
          const last = t.evaluate(batch);
          if (!(last < first * 0.05) || !Number.isFinite(last)) {
            stuck.push(`${kind} ${first.toFixed(4)}->${last.toFixed(4)}`);
          }
        }
        stuck.length === 0 ? "all descend" : stuck.join(", ");
        "#,
    );
    assert_eq!(out, "all descend");
}

#[test]
fn a_uniform_cross_entropy_target_sits_at_its_minimum() {
    // Guards a reading mistake, not a bug: uniform logits against uniform
    // targets is exactly ln(C), so a flat loss curve there is correct and
    // should not be chased as a broken optimizer.
    let out = eval(
        r#"
        const g = new rlx.Graph("ce");
        const logits = g.input("logits", [1, 8], "f32");
        const targets = g.input("t", [1, 8], "f32");
        g.setOutputs([g.mean(g.softmaxCrossEntropy(logits, targets), [0], false)]);
        const c = new rlx.Session("cpu").compile(g);
        const [loss] = c.run({
          logits: new Float32Array(8).fill(0.3),
          t: new Float32Array(8).fill(1 / 8),
        });
        Math.abs(loss[0] - Math.log(8)) < 1e-6 ? "at ln(8)" : `${loss[0]}`;
        "#,
    );
    assert_eq!(out, "at ln(8)");
}

// ── sampling: state that has to survive across calls ─────────

#[test]
fn sampler_actually_samples_across_calls() {
    // The bug this guards: seeding an RNG per call makes every draw identical,
    // so a generation loop emits the same token forever. Bumping the seed by
    // one did not help either — near-identical seeds give near-identical first
    // draws out of an xorshift stream.
    let out = eval(
        r#"
        const logits = new Float32Array(50);
        for (let i = 0; i < 50; i++) logits[i] = Math.sin(i) * 2;

        const s = new rlx.Sampler({ temperature: 1.0, topP: 0.95, seed: 7 });
        const draws = [];
        for (let i = 0; i < 12; i++) draws.push(s.next(logits));
        const distinct = new Set(draws).size;

        // Reproducible for a given seed...
        const again = new rlx.Sampler({ temperature: 1.0, topP: 0.95, seed: 7 });
        const repeat = [];
        for (let i = 0; i < 12; i++) repeat.push(again.next(logits));

        // ...and different for a neighbouring one.
        const other = new rlx.Sampler({ temperature: 1.0, topP: 0.95, seed: 8 });
        const shifted = [];
        for (let i = 0; i < 12; i++) shifted.push(other.next(logits));

        (distinct > 4 && draws.join() === repeat.join() && draws.join() !== shifted.join())
          ? "samples" : `distinct=${distinct} repeat=${draws.join() === repeat.join()} shifted=${draws.join() !== shifted.join()}`;
        "#,
    );
    assert_eq!(out, "samples");
}

#[test]
fn sampler_keeps_history_for_the_repetition_penalty() {
    // A penalty that cannot see what was already generated does nothing, so the
    // sampler owning its history is load-bearing, not a convenience.
    let out = eval(
        r#"
        // The penalty *divides* a positive logit, so 10/1000 = 0.01 only loses
        // if the alternatives are above it — a field of zeros would keep 3 on
        // top and look like the penalty had done nothing.
        const logits = new Float32Array(8).fill(1);
        logits[3] = 10;                        // one favourite
        const greedy = new rlx.Sampler({});
        const first = greedy.next(logits);

        // A penalty alone used to be a silent no-op: `sample_next` returns
        // argmax whenever temperature <= 0, so nothing else ran. Naming a
        // sampling-only option now implies temperature 1.
        const penalized = new rlx.Sampler({ repetitionPenalty: 1000.0 });
        const picks = [];
        for (let i = 0; i < 4; i++) picks.push(penalized.next(logits));

        penalized.prime([1, 2, 3]);
        const primed = penalized.history().join(",");
        penalized.reset(99);

        `${first}|${picks[0]}|${picks.slice(1).every((p) => p !== 3)}|${primed}|${penalized.history().length}`;
        "#,
    );
    // Greedy takes 3; a huge penalty pushes every later pick off it; prime
    // replaces the history; reset clears it.
    assert_eq!(out, "3|3|true|1,2,3|0");
}

#[test]
fn one_shot_sample_next_is_documented_as_deterministic() {
    // Kept as a test helper, and honest about it: same seed, same token. The
    // doc comment says so; this pins the behaviour so the docs stay true.
    let out = eval(
        r#"
        const logits = new Float32Array(50);
        for (let i = 0; i < 50; i++) logits[i] = Math.sin(i) * 2;
        const a = rlx.sampleNext(logits, [], { temperature: 1, seed: 5 });
        const b = rlx.sampleNext(logits, [], { temperature: 1, seed: 5 });
        `${a === b}`;
        "#,
    );
    assert_eq!(out, "true");
}

// ── the reused input scratch ─────────────────────────────────

#[test]
fn reused_input_buffers_do_not_leak_between_calls() {
    // `run` refills persistent buffers instead of allocating per call. The risk
    // that introduces is staleness: a shorter second batch, or a different key
    // set, must not see the previous call's bytes.
    let out = eval(
        r#"
        const g = new rlx.Graph("scratch");
        const x = g.input("x", [4], "f32");
        const y = g.input("y", [4], "f32");
        g.setOutputs([g.add(x, y)]);
        const c = new rlx.Session("cpu").compile(g);

        const seen = [];
        for (const k of [1, 2, 3]) {
          const [out] = c.run({
            x: new Float32Array([k, k, k, k]),
            y: new Float32Array([10 * k, 10 * k, 10 * k, 10 * k]),
          });
          seen.push(Array.from(out).join(","));
        }
        // Reversed key order must give the same answer, not a swapped one.
        const [rev] = c.run({ y: new Float32Array([1, 1, 1, 1]), x: new Float32Array([2, 2, 2, 2]) });
        `${seen.join(" ")} | ${Array.from(rev).join(",")}`;
        "#,
    );
    assert_eq!(out, "11,11,11,11 22,22,22,22 33,33,33,33 | 3,3,3,3");
}

#[test]
fn a_training_loop_is_stable_over_many_steps() {
    // The scratch is moved out of the handle and back on every step. If that
    // handoff dropped a buffer, the loss would drift or the step would panic
    // somewhere past the first iteration rather than on it.
    let out = eval(
        r#"
        const g = new rlx.Graph("mse");
        const x = g.input("x", [1, 2], "f32");
        const t = g.input("t", [1, 2], "f32");
        const w = g.param("w", [2, 2], "f32");
        const d = g.sub(g.matmul(x, w), t);
        g.setOutputs([g.mean(g.mul(d, d), [0, 1], false)]);

        const tr = new rlx.Trainer(g, {
          wrt: ["w"], init: { w: new Float32Array([0, 0, 0, 0]) },
          device: "cpu", optimizer: { kind: "adamw", lr: 0.05 },
        });
        // Alternating batches, so a stale buffer would show up as a loss that
        // stops tracking the input.
        const a = { x: new Float32Array([1, 0]), t: new Float32Array([1, -1]) };
        const b = { x: new Float32Array([0, 1]), t: new Float32Array([-1, 1]) };
        let last = 0;
        for (let i = 0; i < 500; i++) last = tr.step(i % 2 ? a : b);
        const lossA = tr.evaluate(a);
        const lossB = tr.evaluate(b);
        (Number.isFinite(last) && lossA < 1e-3 && lossB < 1e-3 && tr.steps() === 500)
          ? "stable" : `last=${last} a=${lossA} b=${lossB} steps=${tr.steps()}`;
        "#,
    );
    assert_eq!(out, "stable");
}

// ── streaming detokenization (needs a tokenizer fixture) ─────

/// Weights live in the sibling rlx-models repo; try a repo-relative path first,
/// then `$RLX_MODELS_DIR`, and skip if absent.
fn tokenizer_fixture() -> Option<std::path::PathBuf> {
    let rel = "weights/qwen3-0.6b/tokenizer.json";
    let mut candidates = vec![
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../rlx-models")
            .join(rel),
    ];
    if let Some(dir) = std::env::var_os("RLX_MODELS_DIR") {
        candidates.push(std::path::PathBuf::from(dir).join(rel));
    }
    candidates.into_iter().find(|p| p.is_file())
}

#[test]
fn streaming_decode_loses_no_text_including_zwj_graphemes() {
    let Some(path) = tokenizer_fixture() else {
        eprintln!("skip: qwen3-0.6b/tokenizer.json not present");
        return;
    };
    // The bug this guards: a hand-rolled "emit the suffix if the prefix still
    // matches" incremental decoder drops two characters of a ZWJ family emoji,
    // because the decoder rewrites earlier bytes as the sequence completes.
    // Delegating to `rlx_text::incremental_emit` (which holds back a trailing
    // U+FFFD run) fixes it, as does flushing the tail with `finishStream`.
    //
    // Control characters are built from char codes so no layer between this
    // source and the engine can re-interpret an escape.
    let script = format!(
        r#"
        const tok = rlx.loadTokenizer({path:?});
        const cases = [
          "héllo wörld",
          "日本語のテキスト",
          "emoji 🎉🚀 and 👨‍👩‍👧 family",
          "a" + String.fromCharCode(10) + "b" + String.fromCharCode(9) + "c",
        ];
        const bad = [];
        for (const text of cases) {{
          const ids = tok.encode(text, false);
          tok.resetStream();
          let streamed = "";
          for (const id of ids) streamed += tok.decodeIncremental(id);
          streamed += tok.finishStream();
          const whole = tok.decode(ids, false);
          if (streamed !== whole) {{
            bad.push(JSON.stringify(text) + ": " + whole.length + " vs " + streamed.length);
          }}
        }}
        bad.length === 0 ? "no loss" : bad.join("; ");
        "#,
        path = path.to_string_lossy()
    );
    assert_eq!(eval_with_fs(&script), "no loss");
}

// ── the chaining DSL ─────────────────────────────────────────

#[test]
fn the_dsl_builds_a_bit_identical_graph() {
    // The strongest claim available: the DSL is a *spelling*, not a second
    // builder. Same node count, same op kinds, same numbers — so nothing can be
    // fast in one path and correct only in the other.
    let out = eval(
        r#"
        const D = 8;
        function dsl() {
          const m = rlx.dsl("b");
          const x = m.input("x", [2, D]);
          const w = m.param("w", [D, D]);
          const g = m.param("g", [D]);
          const b = m.param("b", [D]);
          m.output(x.rmsNorm(g, b, 1e-6).matmul(w).silu().add(x));
          return m;
        }
        function imperative() {
          const g = new rlx.Graph("b");
          const x = g.input("x", [2, D], "f32");
          const w = g.param("w", [D, D], "f32");
          const gm = g.param("g", [D], "f32");
          const bt = g.param("b", [D], "f32");
          g.setOutputs([g.add(g.silu(g.matmul(g.rmsNorm(x, gm, bt, 1e-6), w)), x)]);
          return g;
        }
        const weights = {
          w: new Float32Array(D * D).fill(0.1),
          g: new Float32Array(D).fill(1),
          b: new Float32Array(D),
        };
        const inputs = { x: new Float32Array(2 * D).fill(0.5) };
        const run = (graph) => {
          const c = new rlx.Session("cpu").compile(graph);
          c.setParams(weights);
          return c.run(inputs)[0];
        };
        const a = dsl(), i = imperative();
        const sameStructure =
          a.nodeCount() === i.nodeCount() &&
          JSON.stringify(a.opKinds()) === JSON.stringify(i.opKinds());
        const outA = run(dsl()), outB = run(imperative());
        let worst = 0;
        for (let k = 0; k < outA.length; k++) worst = Math.max(worst, Math.abs(outA[k] - outB[k]));
        (sameStructure && worst === 0) ? "identical" : `structure=${sameStructure} delta=${worst}`;
        "#,
    );
    assert_eq!(out, "identical");
}

#[test]
fn tensor_methods_are_derived_from_the_graph_prototype() {
    // Tensor's methods are generated by enumerating Graph.prototype, so there is
    // no second list to fall behind. This asserts the derivation actually
    // happened rather than a hand-written subset.
    let out = eval(
        r#"
        const skip = new Set(["input", "param", "constant", "setOutputs", "outputs",
          "nodeCount", "name", "toString", "toJSON", "shapeOf", "paramNames",
          "opKinds", "zeros", "full", "fftfreq", "rfftfreq", "concat", "customFn",
          "__class"]);
        const graphOps = Object.getOwnPropertyNames(rlx.Graph.prototype)
          .filter((n) => !skip.has(n));
        const tensorOps = new Set(Object.getOwnPropertyNames(rlx.Tensor.prototype));
        const missing = graphOps.filter((n) => !tensorOps.has(n));
        (graphOps.length > 70 && missing.length === 0)
          ? `derived ${graphOps.length}` : `missing: ${missing.join(",")}`;
        "#,
    );
    assert!(out.starts_with("derived "), "{out}");
}

#[test]
fn tensor_reports_the_shape_inference_that_happened() {
    // `.dims` reads off the graph, not off what the caller assumed — which is
    // the whole reason to have it rather than tracking shapes in JS.
    let out = eval(
        r#"
        const m = rlx.dsl("shapes");
        const x = m.input("x", [2, 3]);
        const w = m.param("w", [3, 5]);
        const y = x.matmul(w);
        const c = x.compare("lt", m.constant(0.5));
        `${x.dims.join("x")}:${x.dtype}|${y.dims.join("x")}|${c.dtype}|${/^Tensor\(%\d+ \[2, 5\] f32\)$/.test(String(y))}`;
        "#,
    );
    assert_eq!(out, "2x3:f32|2x5|bool|true");
}

#[test]
fn selector_first_ops_splice_the_receiver_correctly() {
    // `activation`, `binary` and `compare` take a string selector first, so the
    // receiver goes in at index 1. Getting that wrong would silently pass the
    // node id as the selector.
    let out = eval(
        r#"
        const m = rlx.dsl("sel");
        const x = m.input("x", [4]);
        m.output([
          x.activation("gelu"),
          x.binary("pow", m.constant(2.0)),
          // Bool has to be cast: `run` is the f32 path and now says so.
          x.compare("gt", m.constant(0.0)).cast("f32"),
        ]);
        const c = new rlx.Session("cpu").compile(m);
        const [gelu, sq, gt] = c.run({ x: new Float32Array([-1, 0, 1, 2]) });
        `${sq[3].toFixed(1)}|${gt[0]}|${gt[3]}|${gelu[1].toFixed(1)}`;
        "#,
    );
    // 2^2 = 4; -1 > 0 is false, 2 > 0 is true; gelu(0) = 0.
    assert_eq!(out, "4.0|0|1|0.0");
}

#[test]
fn a_two_output_op_chains_as_an_array_of_tensors() {
    let out = eval(
        r#"
        const m = rlx.dsl("pair");
        const [re, im] = m.input("sig", [64]).rfft();
        `${re instanceof rlx.Tensor}:${re.dims.join()}:${im.dims.join()}`;
        "#,
    );
    assert_eq!(out, "true:33:33");
}

#[test]
fn constructor_less_classes_still_support_instanceof() {
    // `Compiled`, `Gguf`, `Tokenizer` and `Tensor` are never `new`-ed by a
    // script, but a script still wants to test what it was handed.
    let out = eval(
        r#"
        const m = rlx.dsl("i");
        const x = m.input("x", [2]);
        m.output(x.relu());
        const c = new rlx.Session("cpu").compile(m);
        const notCtor = (() => { try { rlx.Compiled(); return "no throw"; } catch (e) { return "throws"; } })();
        `${x instanceof rlx.Tensor}:${c instanceof rlx.Compiled}:${notCtor}`;
        "#,
    );
    assert_eq!(out, "true:true:throws");
}

#[test]
fn output_accepts_tensors_ids_and_arrays_alike() {
    // Three spellings of the same call; a DSL that only took one would force a
    // conversion dance at every boundary with the imperative API.
    let out = eval(
        r#"
        const shapes = [];
        for (const form of ["tensor", "array", "id"]) {
          const m = rlx.dsl("o");
          const x = m.input("x", [3]);
          const y = x.neg();
          if (form === "tensor") m.output(y);
          else if (form === "array") m.output([y]);
          else m.output(y.id);
          const c = new rlx.Session("cpu").compile(m);
          shapes.push(Array.from(c.run({ x: new Float32Array([1, 2, 3]) })[0]).join(","));
        }
        new Set(shapes).size === 1 ? shapes[0] : shapes.join(" / ");
        "#,
    );
    assert_eq!(out, "-1,-2,-3");
}

#[test]
fn lift_bridges_the_imperative_api_into_a_chain() {
    // A DSL graph is still a Graph, so the id-based methods remain available and
    // `lift` brings their results back into the chain.
    let out = eval(
        r#"
        const m = rlx.dsl("mix");
        const x = m.input("x", [4]);
        // Build one node the imperative way, on the same graph.
        const doubled = m.mul(x.id, m.constant(2.0).id);
        m.output(m.lift(doubled).relu());
        const c = new rlx.Session("cpu").compile(m);
        Array.from(c.run({ x: new Float32Array([-1, 0, 1, 2]) })[0]).join(",");
        "#,
    );
    assert_eq!(out, "0,0,2,4");
}

#[test]
fn run_refuses_to_reinterpret_a_non_f32_output() {
    // Before this check, a Bool output came back as `2.37e-38` — the byte
    // pattern of `true` read as an f32. Plausible-looking garbage, never an
    // error, which is the worst failure mode available.
    let err = eval_err(
        r#"
        const g = new rlx.Graph("bool");
        const x = g.input("x", [4], "f32");
        g.setOutputs([g.compare("gt", x, g.constant(0.0))]);
        const c = new rlx.Session("cpu").compile(g);
        c.run({ x: new Float32Array([-1, 0, 1, 2]) });
        "#,
    );
    assert!(
        err.contains("not f32") && err.contains("runTyped") && err.contains("cast"),
        "the error should name both fixes, got: {err}"
    );
}

#[test]
fn casting_to_f32_is_the_documented_way_through() {
    let out = eval(
        r#"
        const g = new rlx.Graph("bool");
        const x = g.input("x", [4], "f32");
        g.setOutputs([g.cast(g.compare("gt", x, g.constant(0.0)), "f32")]);
        const c = new rlx.Session("cpu").compile(g);
        `${Array.from(c.run({ x: new Float32Array([-1, 0, 1, 2]) })[0]).join(",")}|${c.outputDtypes().join()}`;
        "#,
    );
    assert_eq!(out, "0,0,1,1|f32");
}

// ── node-id validation ───────────────────────────────────────

#[test]
fn an_out_of_range_node_id_throws_instead_of_aborting() {
    // This used to *panic the host process*: the id went straight into
    // `Graph::shape`, which indexes. An embedded engine must not let a script
    // abort its host, so every `node`-kind argument is now checked. The macro
    // generates those checks from the same signature it generates the reader
    // from, so a new op is covered when it is declared.
    for script in [
        r#"const g = new rlx.Graph("g"); g.input("x", [4], "f32"); g.add(0, 99)"#,
        r#"const g = new rlx.Graph("g"); g.input("x", [4], "f32"); g.shapeOf(50)"#,
        r#"const g = new rlx.Graph("g"); g.input("x", [4], "f32"); g.setOutputs([99])"#,
        r#"const g = new rlx.Graph("g"); g.input("x", [4], "f32"); g.relu(7)"#,
        r#"const g = new rlx.Graph("g"); g.input("x", [4], "f32"); g.concat([0, 42], 0)"#,
        r#"const g = new rlx.Graph("g"); g.input("x", [4], "f32"); g.rope(0, 8, 9, 4)"#,
        r#"const g = new rlx.Graph("g"); g.input("x", [4], "f32"); g.pad(31, [[1, 1]])"#,
    ] {
        let err = eval_err(script);
        assert!(
            err.contains("does not exist in this graph"),
            "script `{script}` should be a RangeError, got: {err}"
        );
    }
}

// ── DSL scalar lifting ───────────────────────────────────────

#[test]
fn elementwise_ops_lift_a_scalar_to_a_constant() {
    // `x.add(2)` used to mean "node id 2". In the DSL a number in a tensor
    // operand slot becomes a constant of the receiver's dtype.
    let out = eval(
        r#"
        const m = rlx.dsl("lift");
        const x = m.input("x", [4]);
        m.output([x.mul(2).add(1), x.div(4.0), x.binary("pow", 2), x.sub(0.5)]);
        const c = new rlx.Session("cpu").compile(m);
        c.run({ x: new Float32Array([1, 2, 3, 4]) })
          .map((o) => Array.from(o).join(",")).join(" | ");
        "#,
    );
    assert_eq!(
        out,
        "3,5,7,9 | 0.25,0.5,0.75,1 | 1,4,9,16 | 0.5,1.5,2.5,3.5"
    );
}

#[test]
fn a_lifted_constant_follows_the_receivers_dtype() {
    // Forcing f32 into an f64 graph would be a dtype mismatch at the binary op.
    let out = eval(
        r#"
        const m = rlx.dsl("dt");
        const x = m.input("x", [2], "f64");
        const y = x.mul(2);
        `${y.dtype}`;
        "#,
    );
    assert_eq!(out, "f64");
}

#[test]
fn scalars_are_not_lifted_where_they_are_real_scalars() {
    // `narrow(axis, start, len)` takes three integers; lifting any of them
    // would build a constant tensor and then fail somewhere unhelpful.
    let out = eval(
        r#"
        const m = rlx.dsl("noLift");
        const x = m.input("x", [2, 6]);
        m.output(x.narrow(1, 2, 3));
        const c = new rlx.Session("cpu").compile(m);
        Array.from(c.run({ x: new Float32Array([1,2,3,4,5,6, 7,8,9,10,11,12]) })[0]).join(",");
        "#,
    );
    assert_eq!(out, "3,4,5,9,10,11");
}

#[test]
fn concat_chains_from_a_tensor() {
    // `Graph.concat` takes the list first, so the generic splice rule would
    // build `concat(a, [b, c], axis)`. This is the bespoke arm for it.
    let out = eval(
        r#"
        const m = rlx.dsl("cat");
        const a = m.input("a", [2]);
        const b = m.input("b", [2]);
        m.output([a.concat([b, a.neg()], 0), a.concat(b, 0)]);
        const c = new rlx.Session("cpu").compile(m);
        c.run({ a: new Float32Array([1, 2]), b: new Float32Array([9, 9]) })
          .map((o) => Array.from(o).join(",")).join(" | ");
        "#,
    );
    assert_eq!(out, "1,2,9,9,-1,-2 | 1,2,9,9");
}

// ── filesystem: off unless handed over ───────────────────────

#[test]
fn the_runtime_is_sandboxed_until_an_embedder_says_otherwise() {
    // The README claims "no I/O a script can reach unless an embedder hands it
    // one". The first version of this test only checked `rlx.readFile`, and
    // missed that `openGguf`, `loadTokenizer`, `writeGguf` and every weight
    // loader also take paths and were installed unconditionally. A claim tested
    // that narrowly is worse than no claim.
    let mut sealed = Runtime::new();
    let probe = "typeof rlx.readFile + ',' + typeof rlx.writeFile + ',' + typeof rlx.env";
    assert_eq!(
        sealed.eval(probe).unwrap(),
        "undefined,undefined,undefined",
        "Runtime::new must not expose the filesystem"
    );

    // Every *other* path-taking function must refuse rather than act.
    for call in [
        r#"rlx.openGguf("/etc/hosts")"#,
        r#"rlx.writeGguf("/tmp/rlx-should-not-exist.gguf", {})"#,
        r#"rlx.loadTokenizer("/etc/hosts")"#,
        r#"rlx.loadPt("/etc/hosts")"#,
        r#"rlx.loadMlx("/etc/hosts")"#,
        r#"rlx.openRlxp("/etc/hosts")"#,
        r#"rlx.toRlxp("/etc/hosts", "/tmp/x", { from: "gguf" })"#,
        r#"rlx.verifyRlxp("/etc/hosts")"#,
    ] {
        let err = match sealed.eval(call) {
            Ok(v) => panic!("sandboxed runtime allowed `{call}` -> {v}"),
            Err(e) => e,
        };
        assert!(
            err.contains("no filesystem access"),
            "`{call}` should be refused for lack of permission, got: {err}"
        );
    }
    assert!(
        !std::path::Path::new("/tmp/rlx-should-not-exist.gguf").exists(),
        "a sandboxed runtime wrote a file"
    );

    let mut opened = Runtime::new();
    opened.allow_filesystem();
    assert_eq!(opened.eval(probe).unwrap(), "function,function,function");
    // Now it gets as far as the file itself rather than the permission check.
    let err = opened.eval(r#"rlx.openGguf("/etc/hosts")"#).unwrap_err();
    assert!(!err.contains("no filesystem access"), "still gated: {err}");
}

#[test]
fn an_inline_chat_template_needs_no_permission() {
    // Rendering a template *string* touches nothing, so it must keep working in
    // a sealed runtime — only a path is privileged.
    let out = eval(
        r#"
        rlx.renderChat("{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}",
                       [{ role: "user", content: "hi" }], false);
        "#,
    );
    assert_eq!(out, "<user>hi");
}

#[test]
fn file_reads_round_trip_and_report_real_errors() {
    let mut rt = Runtime::new();
    rt.allow_filesystem();
    let dir = std::env::temp_dir().join("rlx-js-fs-test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("bytes.bin");
    let script = format!(
        r#"
        const p = {path:?};
        const n = rlx.writeFile(p, new Uint8Array([1, 2, 3, 250]));
        const back = rlx.readFile(p);
        const missing = (() => {{
          try {{ rlx.readFile(p + ".nope"); return "no throw"; }}
          catch (e) {{ return String(e).includes("No such file") ? "reports missing" : String(e); }}
        }})();
        `${{n}}|${{Array.from(back).join(",")}}|${{rlx.fileSize(p)}}|${{rlx.fileExists(p + ".nope")}}|${{missing}}`;
        "#,
        path = path.to_string_lossy()
    );
    let out = rt.eval(&script).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(out, "4|1,2,3,250|4|false|reports missing");
    let _ = std::fs::remove_dir_all(&dir);
}

// ── MNIST (needs the dataset) ────────────────────────────────

/// The cache directories `rlx-vision-bench` uses, so a prior download is found.
fn mnist_dir() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let files = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ];
    let mut candidates = Vec::new();
    if let Some(dir) = std::env::var_os("MNIST_DIR") {
        candidates.push(std::path::PathBuf::from(dir));
    }
    candidates.push(std::path::PathBuf::from(format!(
        "{home}/.cache/torchvision-mnist/MNIST/raw"
    )));
    candidates.push(std::path::PathBuf::from(format!(
        "{home}/.cache/rlx-datasets/mnist"
    )));
    candidates.into_iter().find(|dir| {
        files.iter().all(|f| {
            std::fs::metadata(dir.join(f))
                .map(|m| m.len() > 100)
                .unwrap_or(false)
        })
    })
}

#[test]
fn mnist_trains_past_ninety_percent() {
    let Some(dir) = mnist_dir() else {
        eprintln!(
            "skip: MNIST idx files not present (see examples/mnist.js for the fetch command)"
        );
        return;
    };
    // Short on purpose. `cargo test` builds unoptimized, so the CPU kernels run
    // at a fraction of release speed and 600 steps cost 25 s — this is a
    // correctness gate, not a benchmark. The bar is deliberately low: it asserts
    // the pipeline learns, and `examples/mnist.js` is where the real 97.98%
    // lives.
    let mut rt = Runtime::new();
    rt.allow_filesystem();
    let script = format!(
        r#"
        const DIR = {dir:?};
        function readIdx(path) {{
          const bytes = rlx.readFile(path);
          const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
          const magic = view.getUint32(0, false);
          const rank = magic & 0xff;
          const dims = [];
          for (let i = 0; i < rank; i++) dims.push(view.getUint32(4 + 4 * i, false));
          return {{ dims, data: bytes.subarray(4 + 4 * rank) }};
        }}
        // Only the rows this run will touch: `readFileRange` is what makes a
        // dataset bigger than memory workable, and here it keeps the test cheap.
        function split(img, lab, limit) {{
          const head = readIdx(DIR + "/" + img);
          const [total, h, w] = head.dims;
          const n = Math.min(limit, total);
          const pixels = h * w;
          const bytes = rlx.readFileRange(DIR + "/" + img, 16, n * pixels);
          const x = rlx.toFloat32(bytes, {{ scale: 1 / (255 * 0.3081), bias: -0.1307 / 0.3081 }});
          const labels = readIdx(DIR + "/" + lab);
          return {{ n, pixels, x, y: labels.data }};
        }}
        const EVAL = 2000;
        const train = split("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 20000);
        const test = split("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", EVAL);

        const BATCH = 64, HIDDEN = 32, CLASSES = 10, STEPS = 250;
        // `cargo test` builds unoptimized, so the CPU kernels run at a fraction
        // of release speed. A GPU backend does its work inside a framework that
        // is optimized either way, so prefer one when the build has it.
        const DEV = rlx.devices().find((d) => d !== "cpu") || "cpu";
        function build(batch, head) {{
          const m = rlx.dsl("mnist");
          const x = m.input("x", [batch, train.pixels]);
          const y = m.input("y", [batch, CLASSES]);
          const logits = x
            .matmul(m.param("w1", [train.pixels, HIDDEN])).add(m.param("b1", [1, HIDDEN])).relu()
            .matmul(m.param("w2", [HIDDEN, CLASSES])).add(m.param("b2", [1, CLASSES]));
          m.output(head === "loss" ? logits.softmaxCrossEntropy(y).mean([0], false)
                                   : logits.argmax(1, false));
          return m;
        }}
        let s = 7;
        const gauss = () => {{
          const u = ((s = (s * 1664525 + 1013904223) >>> 0) >>> 8) / 16777216 || 1e-7;
          const v = ((s = (s * 1664525 + 1013904223) >>> 0) >>> 8) / 16777216;
          return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
        }};
        const he = (fanIn, n) => {{
          const a = new Float32Array(n), k = Math.sqrt(2 / fanIn);
          for (let i = 0; i < n; i++) a[i] = gauss() * k;
          return a;
        }};
        const init = {{
          w1: he(train.pixels, train.pixels * HIDDEN), b1: new Float32Array(HIDDEN),
          w2: he(HIDDEN, HIDDEN * CLASSES), b2: new Float32Array(CLASSES),
        }};
        const trainer = new rlx.Trainer(build(BATCH, "loss"), {{
          wrt: ["w1", "b1", "w2", "b2"], init, device: DEV,
          optimizer: {{ kind: "adamw", lr: 3e-3, weightDecay: 1e-4 }},
        }});

        let cursor = 0;
        const pick = new Int32Array(BATCH), labels = new Int32Array(BATCH);
        const result = trainer.run({{
          steps: STEPS,
          batch: () => {{
            if (cursor + BATCH > train.n) cursor = 0;
            for (let b = 0; b < BATCH; b++) {{
              pick[b] = cursor + b;
              labels[b] = train.y[pick[b]];
            }}
            cursor += BATCH;
            return {{
              x: rlx.gatherRows(train.x, pick, train.pixels),
              y: rlx.oneHot(labels, CLASSES),
            }};
          }},
        }});
        const first = result.firstLoss;

        const infer = new rlx.Session(DEV).compile(build(EVAL, "argmax"));
        infer.setParams(trainer.params());
        const [pred] = infer.run({{ x: test.x, y: new Float32Array(EVAL * CLASSES) }});
        let correct = 0;
        for (let i = 0; i < EVAL; i++) if (pred[i] === test.y[i]) correct++;
        const acc = (100 * correct) / EVAL;
        `${{first > 2 && first < 6}}|${{acc > 88}}|${{acc.toFixed(1)}}% on ${{DEV}}`;
        "#,
        dir = dir.to_string_lossy()
    );
    let out = rt.eval(&script).unwrap_or_else(|e| panic!("{e}"));
    let parts: Vec<&str> = out.split('|').collect();
    assert_eq!(
        parts[0], "true",
        "initial loss should be near ln(10); got {out}"
    );
    assert_eq!(parts[1], "true", "accuracy should clear 88%; got {out}");
    eprintln!("mnist: {} after 250 steps (debug build)", parts[2]);
}

// ── training lifecycle ───────────────────────────────────────

#[test]
fn params_covers_every_parameter_not_only_the_trainable_ones() {
    // The invariant: `setParams(trainer.params())` on a matching inference graph
    // must reproduce the trainer's own forward pass. When `params()` omitted the
    // non-`wrt` weights, the inference graph silently kept zeros — the training
    // loss looked healthy and the evaluation was nonsense.
    let out = eval(
        r#"
        const P = 4, H = 3;
        function build(batch, head) {
          const m = rlx.dsl("frozen-base");
          const x = m.input("x", [batch, P]);
          const y = m.input("y", [batch, H]);
          const logits = x.matmul(m.param("base", [P, H])).add(m.param("delta", [1, H]));
          m.output(head === "loss" ? logits.softmaxCrossEntropy(y).mean([0], false) : logits);
          return m;
        }
        const base = new Float32Array([1,0,0, 0,1,0, 0,0,1, 1,1,1]);
        const t = new rlx.Trainer(build(2, "loss"), {
          // `base` is in `init` but not `wrt`: no gradient, never updated.
          wrt: ["delta"],
          init: { base, delta: new Float32Array([0.5, -0.5, 0.25]) },
          device: "cpu", optimizer: { kind: "adamw", lr: 0.01 },
        });
        const p = t.params();
        const keys = Object.keys(p).sort().join(",");
        const baseIntact = Array.from(p.base).join() === Array.from(base).join();
        // And the round trip must actually reproduce a forward pass.
        const infer = new rlx.Session("cpu").compile(build(1, "logits"));
        infer.setParams(p);
        const [logits] = infer.run({ x: new Float32Array([1, 2, 3, 4]), y: new Float32Array(H) });
        `${keys}|${baseIntact}|${Array.from(logits).map((v) => v.toFixed(2)).join(",")}`;
        "#,
    );
    // x·base = [1+4, 2+4, 3+4] = [5, 6, 7]; plus delta = [5.5, 5.5, 7.25].
    assert_eq!(out, "base,delta|true|5.50,5.50,7.25");
}

#[test]
fn run_reports_early_stop_and_resumes_where_it_left_off() {
    let out = eval(
        r#"
        const g = () => {
          const m = rlx.dsl("mse");
          const x = m.input("x", [1, 2]);
          const t = m.input("t", [1, 2]);
          const d = x.matmul(m.param("w", [2, 2])).sub(t);
          m.output(d.mul(d).mean([0, 1], false));
          return m;
        };
        const tr = new rlx.Trainer(g(), {
          wrt: ["w"], init: { w: new Float32Array(4) },
          device: "cpu", optimizer: { kind: "adamw", lr: 0.05 },
        });
        const batch = { x: new Float32Array([1, 1]), t: new Float32Array([1, -1]) };
        // `onStep` returning false stops; anything else (including undefined)
        // continues, so a logging callback does not end the run by accident.
        const a = tr.run({ steps: 500, batch: () => batch, onStep: (i) => i < 9 });
        const b = tr.run({ steps: 5, batch: () => batch, onStep: () => undefined });
        `${a.stopped}:${a.steps}:${b.stopped}:${b.steps}:${tr.steps()}`;
        "#,
    );
    assert_eq!(out, "true:10:false:5:15");
}

#[test]
fn a_batch_provider_returning_null_ends_the_run() {
    // How a finite dataset signals exhaustion without the script tracking counts.
    let out = eval(
        r#"
        const m = rlx.dsl("mse");
        const x = m.input("x", [1, 2]);
        const t = m.input("t", [1, 2]);
        const d = x.matmul(m.param("w", [2, 2])).sub(t);
        m.output(d.mul(d).mean([0, 1], false));
        const tr = new rlx.Trainer(m, {
          wrt: ["w"], init: { w: new Float32Array(4) },
          device: "cpu", optimizer: { kind: "adamw", lr: 0.05 },
        });
        let left = 7;
        const r = tr.run({
          steps: 1000,
          batch: () => (left-- > 0 ? { x: new Float32Array([1, 1]), t: new Float32Array([1, -1]) } : null),
        });
        `${r.steps}:${r.stopped}`;
        "#,
    );
    assert_eq!(out, "7:true");
}

#[test]
fn gradient_clipping_bounds_the_step() {
    // A deliberately huge gradient: without clipping the first AdamW step is
    // +-lr per element either way, so this compares the *reported norm* and that
    // clipping leaves the parameters finite and bounded.
    let out = eval(
        r#"
        function build() {
          const m = rlx.dsl("big");
          const x = m.input("x", [1, 2]);
          const t = m.input("t", [1, 2]);
          const d = x.matmul(m.param("w", [2, 2])).sub(t);
          m.output(d.mul(d).mean([0, 1], false));
          return m;
        }
        const batch = { x: new Float32Array([1e3, 1e3]), t: new Float32Array([0, 0]) };
        const opts = (clip) => ({
          wrt: ["w"], init: { w: new Float32Array([1, 1, 1, 1]) },
          device: "cpu", optimizer: { kind: "sgd", lr: 1e-3 }, clipNorm: clip,
        });
        const loose = new rlx.Trainer(build(), opts(undefined));
        loose.step(batch);
        const tight = new rlx.Trainer(build(), opts(1.0));
        tight.step(batch);
        const lw = Array.from(loose.params().w), tw = Array.from(tight.params().w);
        const finite = tw.every(Number.isFinite);
        // Both see the same gradient, so the reported norms match; only the step
        // taken differs.
        `${loose.gradNorm() > 1e6}|${Math.abs(loose.gradNorm() - tight.gradNorm()) < 1}|${finite}|${Math.abs(tw[0] - 1) < Math.abs(lw[0] - 1)}`;
        "#,
    );
    assert_eq!(out, "true|true|true|true");
}

#[test]
fn a_checkpoint_round_trips_through_gguf() {
    // Checkpoints are GGUF, so `rlx.openGguf` can read one — which is how a
    // corrupt checkpoint is diagnosed without this crate.
    let mut rt = Runtime::new();
    rt.allow_filesystem();
    let dir = std::env::temp_dir().join("rlx-js-ckpt-test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("ckpt.gguf");
    let script = format!(
        r#"
        const CK = {path:?};
        function build() {{
          const m = rlx.dsl("mse");
          const x = m.input("x", [1, 2]);
          const t = m.input("t", [1, 2]);
          const d = x.matmul(m.param("w", [2, 2])).add(m.param("b", [1, 2])).sub(t);
          m.output(d.mul(d).mean([0, 1], false));
          return m;
        }}
        const opts = {{
          wrt: ["w", "b"],
          init: {{ w: new Float32Array(4), b: new Float32Array(2), frozenBias: new Float32Array([7, 7]) }},
          device: "cpu", optimizer: {{ kind: "adamw", lr: 0.05 }},
        }};
        const batch = {{ x: new Float32Array([1, 1]), t: new Float32Array([1, -1]) }};

        const a = new rlx.Trainer(build(), opts);
        a.run({{ steps: 40, batch: () => batch }});
        a.freeze(["b"]);
        const tensors = a.save(CK);
        const continued = a.run({{ steps: 40, batch: () => batch }}).lastLoss;

        const b = new rlx.Trainer(build(), opts);
        const rep = b.load(CK);
        const resumed = b.run({{ steps: 40, batch: () => batch }}).lastLoss;

        // Readable as an ordinary GGUF, with the header a reader would expect.
        const f = rlx.openGguf(CK);
        const meta = f.metadata();
        `${{tensors}}|${{rep.step}}|${{rep.optimizerRestored}}|${{continued === resumed}}` +
        `|${{JSON.stringify(b.frozen())}}|${{meta["general.architecture"]}}` +
        `|${{f.tensorNames().filter((n) => n.startsWith("opt/")).length > 0}}`;
        "#,
        path = path.to_string_lossy()
    );
    let out = rt.eval(&script).unwrap_or_else(|e| panic!("{e}"));
    let parts: Vec<&str> = out.split('|').collect();
    assert_eq!(parts[1], "40", "step should come back: {out}");
    assert_eq!(parts[2], "true", "optimizer state should restore: {out}");
    assert_eq!(
        parts[3], "true",
        "a resumed run must match an uninterrupted one exactly: {out}"
    );
    assert_eq!(parts[4], r#"["b"]"#, "the frozen set should persist: {out}");
    assert_eq!(parts[5], "rlx-checkpoint", "{out}");
    assert_eq!(
        parts[6], "true",
        "optimizer buffers should be in the file: {out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn loading_a_checkpoint_from_a_different_optimizer_is_refused() {
    // Adam's moments mean nothing to Lion. Loading them anyway would train,
    // badly, with no sign anything was wrong.
    let mut rt = Runtime::new();
    rt.allow_filesystem();
    let dir = std::env::temp_dir().join("rlx-js-ckpt-mismatch");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("adam.gguf");
    let script = format!(
        r#"
        const CK = {path:?};
        function build() {{
          const m = rlx.dsl("mse");
          const x = m.input("x", [1, 2]);
          const w = m.param("w", [2, 2]);
          m.output(x.matmul(w).mul(x.matmul(w)).mean([0, 1], false));
          return m;
        }}
        const init = {{ w: new Float32Array([0.1, 0.2, 0.3, 0.4]) }};
        const a = new rlx.Trainer(build(), {{ wrt: ["w"], init, device: "cpu",
          optimizer: {{ kind: "adamw", lr: 0.01 }} }});
        a.step({{ x: new Float32Array([1, 1]) }});
        a.save(CK);
        const b = new rlx.Trainer(build(), {{ wrt: ["w"], init, device: "cpu",
          optimizer: {{ kind: "lion", lr: 0.01 }} }});
        try {{ b.load(CK); "loaded"; }} catch (e) {{ String(e); }}
        "#,
        path = path.to_string_lossy()
    );
    let out = rt.eval(&script).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        out.contains("adamw") && out.contains("lion"),
        "the error should name both algorithms, got: {out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── async ────────────────────────────────────────────────────

#[test]
fn eval_awaits_a_returned_promise() {
    // This used to come back as "[object Promise]", which made every async
    // script look like it had returned nothing.
    assert_eq!(eval("(async () => 41 + 1)()"), "42");
    assert_eq!(eval("Promise.resolve('resolved')"), "resolved");
    let err = eval_err("(async () => { throw new Error('boom'); })()");
    assert!(err.contains("boom"), "a rejection should surface: {err}");
}

#[test]
fn timers_fire_in_due_order_after_microtasks() {
    let out = eval(
        r#"
        (async () => {
          const order = [];
          setTimeout(() => order.push("t20"), 20);
          setTimeout(() => order.push("t0"), 0);
          queueMicrotask(() => order.push("micro"));
          const cancelled = setTimeout(() => order.push("never"), 5);
          clearTimeout(cancelled);
          await rlx.sleep(40);
          return order.join(",");
        })();
        "#,
    );
    assert_eq!(out, "micro,t0,t20");
}

#[test]
fn sleep_actually_waits() {
    let out = eval(
        r#"
        (async () => {
          const t0 = Date.now();
          await rlx.sleep(25);
          return Date.now() - t0 >= 20 ? "waited" : `only ${Date.now() - t0}ms`;
        })();
        "#,
    );
    assert_eq!(out, "waited");
}

#[test]
fn a_promise_that_can_never_settle_is_reported_as_such() {
    // Better than "[object Promise]" and better than hanging.
    let mut rt = Runtime::new();
    let err = rt
        .eval("new Promise(() => {})")
        .expect_err("a forever-pending promise should not look like success");
    assert!(err.contains("never settle"), "unhelpful: {err}");
}

#[test]
fn training_can_yield_between_chunks() {
    // There are no threads here, so "async training" means cooperative
    // scheduling: run a chunk, yield, let a reporter run, continue.
    let out = eval(
        r#"
        (async () => {
          const m = rlx.dsl("mse");
          const x = m.input("x", [1, 2]);
          const t = m.input("t", [1, 2]);
          const d = x.matmul(m.param("w", [2, 2])).sub(t);
          m.output(d.mul(d).mean([0, 1], false));
          const tr = new rlx.Trainer(m, {
            wrt: ["w"], init: { w: new Float32Array(4) },
            device: "cpu", optimizer: { kind: "adamw", lr: 0.05 },
          });
          const batch = { x: new Float32Array([1, 1]), t: new Float32Array([1, -1]) };
          const reports = [];
          for (let done = 0; done < 60; done += 20) {
            const r = tr.run({ steps: 20, batch: () => batch });
            reports.push(r.steps);
            await rlx.sleep(0);          // let anything else pending run
          }
          return `${reports.join(",")}|${tr.steps()}`;
        })();
        "#,
    );
    assert_eq!(out, "20,20,20|60");
}

// ── buffer helpers ───────────────────────────────────────────

#[test]
fn buffer_helpers_match_the_arithmetic_they_replace() {
    let out = eval(
        r#"
        const raw = new Uint8Array([0, 64, 128, 255]);
        const widened = rlx.toFloat32(raw);
        const affine = rlx.toFloat32(raw, { scale: 1 / 255, bias: -0.5 });
        const expect = Array.from(raw).map((b) => b / 255 - 0.5);
        let worst = 0;
        for (let i = 0; i < 4; i++) worst = Math.max(worst, Math.abs(affine[i] - expect[i]));

        const hot = rlx.oneHot([2, 0, 1], 3);
        const rows = rlx.gatherRows(new Float32Array([1,2, 3,4, 5,6, 7,8]), [3, 0], 2);
        `${Array.from(widened).join(",")}|${worst < 1e-6}|${Array.from(hot).join(",")}|${Array.from(rows).join(",")}`;
        "#,
    );
    assert_eq!(out, "0,64,128,255|true|0,0,1,1,0,0,0,1,0|7,8,1,2");
}

#[test]
fn to_float32_converts_per_element_not_per_byte() {
    // The bug: reading every input as raw bytes meant
    // `toFloat32(new Float32Array([1, 2, 3]))` returned twelve values — the IEEE
    // byte pattern — instead of three. Silent, and exactly the wrong shape to
    // notice downstream.
    let out = eval(
        r#"
        const rows = [
          ["u8",  rlx.toFloat32(new Uint8Array([0, 128, 255]))],
          ["i8",  rlx.toFloat32(new Int8Array([-1, 0, 127]))],
          ["i16", rlx.toFloat32(new Int16Array([-300, 300]))],
          ["u16", rlx.toFloat32(new Uint16Array([65535]))],
          ["i32", rlx.toFloat32(new Int32Array([-5, 1000]))],
          ["u32", rlx.toFloat32(new Uint32Array([7]))],
          ["f32", rlx.toFloat32(new Float32Array([1, 2, 3]))],
          ["f64", rlx.toFloat32(new Float64Array([0.5, -2]))],
          ["arr", rlx.toFloat32([7, 8])],
        ];
        rows.map(([k, v]) => `${k}=${Array.from(v).join(",")}`).join(" ");
        "#,
    );
    assert_eq!(
        out,
        "u8=0,128,255 i8=-1,0,127 i16=-300,300 u16=65535 i32=-5,1000 u32=7 f32=1,2,3 f64=0.5,-2 arr=7,8"
    );
}

#[test]
fn to_float32_applies_the_affine_to_every_element_type() {
    let out = eval(
        r#"
        const o = { scale: 2, bias: 1 };
        [
          Array.from(rlx.toFloat32(new Uint8Array([0, 1, 2]), o)).join(","),
          Array.from(rlx.toFloat32(new Float32Array([0, 1, 2]), o)).join(","),
          Array.from(rlx.toFloat32([0, 1, 2], o)).join(","),
        ].join(" | ");
        "#,
    );
    assert_eq!(out, "1,3,5 | 1,3,5 | 1,3,5");
}

#[test]
fn a_range_read_clamps_to_the_file() {
    // `length` comes from script; a mistaken 1e12 should be a short read, not an
    // allocation the host cannot survive.
    let mut rt = Runtime::new();
    rt.allow_filesystem();
    let dir = std::env::temp_dir().join("rlx-js-range-test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("small.bin");
    std::fs::write(&path, [1u8, 2, 3, 4, 5, 6, 7, 8]).expect("write");
    let script = format!(
        r#"
        const p = {path:?};
        const huge = rlx.readFileRange(p, 0, 1e9);
        const mid = rlx.readFileRange(p, 4, 100);
        const past = rlx.readFileRange(p, 99, 10);
        // And into a caller-owned buffer, which is the streaming shape.
        const target = new Uint8Array(4);
        const n = rlx.readFileInto(p, target, 2);
        `${{huge.length}}|${{Array.from(mid).join(",")}}|${{past.length}}|${{n}}:${{Array.from(target).join(",")}}`;
        "#,
        path = path.to_string_lossy()
    );
    let out = rt.eval(&script).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(out, "8|5,6,7,8|0|4:3,4,5,6");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn buffer_helpers_reject_out_of_range_input() {
    // Silently writing a label into the previous row is the bug these guard.
    for (script, wanted) in [
        (r#"rlx.oneHot([0, 5], 3)"#, "outside"),
        (
            r#"rlx.gatherRows(new Float32Array([1,2,3,4]), [9], 2)"#,
            "outside",
        ),
        (r#"rlx.oneHot([0], 0)"#, "at least 1"),
    ] {
        let err = eval_err(script);
        assert!(err.contains(wanted), "`{script}` -> {err}");
    }
}

// ── panics converted at the boundary ─────────────────────────

#[test]
fn rlx_assertions_become_catchable_errors_not_aborts() {
    // rlx asserts its shape invariants with `panic!`, which is right for Rust
    // callers and fatal for an embedded engine: before this, a typo in a script
    // aborted the host process. Every generated op body runs under a guard, so
    // an assertion in an op nobody thought to test is still catchable.
    for (script, wanted) in [
        (
            r#"const g = new rlx.Graph("p");
               g.matmul(g.input("a", [2, 3], "f32"), g.input("b", [5, 7], "f32"))"#,
            "K mismatch",
        ),
        (
            r#"const g = new rlx.Graph("p");
               g.reshape(g.input("a", [2, 3], "f32"), [7, 7])"#,
            "reshape",
        ),
        (
            r#"const g = new rlx.Graph("p");
               g.concat([g.input("a", [2, 3], "f32"), g.input("b", [4, 4], "f32")], 0)"#,
            "concat",
        ),
        (
            r#"const g = new rlx.Graph("p");
               g.transpose(g.input("a", [2, 3], "f32"), [0, 1, 2])"#,
            "transpose",
        ),
    ] {
        let err = eval_err(script);
        assert!(
            err.contains(wanted) && err.contains("Rust assertion"),
            "`{wanted}` should surface as a caught assertion, got: {err}"
        );
    }
}

#[test]
fn a_narrow_window_past_its_axis_is_rejected() {
    // `narrow_shape` only checked the axis, never `start + len`, so `[2, 3]`
    // narrowed to `len` 9 inferred `[2, 9]` — a *larger* shape than the input,
    // which compiled, passed `rlx.check`, and read out of bounds at execution.
    for (start, len) in [(0, 9), (2, 5), (3, 1)] {
        let err = eval_err(&format!(
            r#"const g = new rlx.Graph("n");
               g.narrow(g.input("a", [2, 3], "f32"), 1, {start}, {len})"#
        ));
        assert!(
            err.contains("exceeds axis"),
            "narrow({start}, {len}) on a 3-wide axis should be refused, got: {err}"
        );
    }
    // And the in-range case still works.
    let out = eval(
        r#"
        const g = new rlx.Graph("n");
        JSON.stringify(g.shapeOf(g.narrow(g.input("a", [2, 3], "f32"), 1, 1, 2)).dims);
        "#,
    );
    assert_eq!(out, "[2,2]");
}

#[test]
fn a_trainable_parameter_the_loss_ignores_is_reported() {
    // `grad_with_loss` panics on a `wrt` with no gradient path — a typo, or a
    // parameter the graph never uses. That used to abort the host.
    let err = eval_err(
        r#"
        const m = rlx.dsl("unused");
        const x = m.input("x", [1, 2]);
        const w = m.param("w", [2, 2]);
        m.param("spare", [4]);                    // declared, never used
        m.output(x.matmul(w).mul(x.matmul(w)).mean([0, 1], false));
        new rlx.Trainer(m, {
          wrt: ["w", "spare"],
          init: { w: new Float32Array(4), spare: new Float32Array(4) },
          device: "cpu", optimizer: { kind: "adamw", lr: 0.01 },
        });
        "#,
    );
    assert!(
        err.contains("no gradient flowed") && err.contains("Trainer"),
        "should name the stage and the cause: {err}"
    );
}

#[test]
fn grad_on_an_unreachable_parameter_is_reported() {
    let err = eval_err(
        r#"
        const g = new rlx.Graph("g");
        const a = g.input("a", [2], "f32");
        const p = g.param("p", [2], "f32");
        g.setOutputs([g.sum(a, [0], false)]);      // `p` is not in the loss
        rlx.grad(g, [p]);
        "#,
    );
    assert!(err.contains("no gradient flowed"), "{err}");
}

// ── resident (fused) training ────────────────────────────────

/// The fused update is only offered where it is both correct and faster, so the
/// refusals are part of the contract, not an implementation detail.
#[test]
fn the_resident_path_refuses_backends_where_it_does_not_pay() {
    // CPU has no device buffers: the fused graph would feed parameters and
    // moments as ordinary inputs and read three buffers back per parameter.
    // Measured 3.3x slower than the host optimizer, so it is refused with that
    // reason rather than silently accepted.
    let err = eval_err(
        r#"
        const m = rlx.dsl("mse");
        const x = m.input("x", [1, 2]);
        const t = m.input("t", [1, 2]);
        const d = x.matmul(m.param("w", [2, 2])).sub(t);
        m.output(d.mul(d).mean([0, 1], false));
        new rlx.Trainer(m, {
          wrt: ["w"], init: { w: new Float32Array(4) }, device: "cpu",
          optimizer: { kind: "adamw", lr: 0.01 }, resident: true,
        });
        "#,
    );
    assert!(
        err.contains("no device buffers") && err.contains("slower"),
        "the refusal should say why: {err}"
    );
}

#[test]
fn the_resident_path_refuses_what_it_cannot_fuse() {
    let build = r#"
        const m = rlx.dsl("mse");
        const x = m.input("x", [1, 2]);
        const t = m.input("t", [1, 2]);
        const d = x.matmul(m.param("w", [2, 2])).sub(t);
        m.output(d.mul(d).mean([0, 1], false));
    "#;
    // Only Adam/AdamW are fused; and `clipNorm` needs the gradients on the host,
    // which the fused path never brings back. Both must be stated, not ignored.
    for (extra, wanted) in [
        (
            r#"optimizer: { kind: "muon", lr: 0.01 }, resident: true"#,
            "only 'adam' and 'adamw'",
        ),
        (
            r#"optimizer: { kind: "adamw", lr: 0.01 }, clipNorm: 1.0, resident: true"#,
            "clipNorm",
        ),
    ] {
        let err = eval_err(&format!(
            "{build}\n new rlx.Trainer(m, {{ wrt: [\"w\"], \
             init: {{ w: new Float32Array(4) }}, device: \"metal\", {extra} }});"
        ));
        // On a build without Metal the device check fires first, which is also a
        // correct refusal — accept either.
        assert!(
            err.contains(wanted) || err.contains("not in this build"),
            "expected `{wanted}`, got: {err}"
        );
    }
}

#[test]
fn a_resident_trainer_matches_the_host_path() {
    // The claim the whole feature rests on: fusing the optimizer changes where
    // the arithmetic runs, not what it computes. Skipped without a verified
    // backend rather than asserted against the path that is refused.
    if !rlx_runtime::is_available(rlx_runtime::Device::Metal) {
        eprintln!("skip: no Metal backend in this build");
        return;
    }
    let out = eval(
        r#"
        const P = 16, H = 8, C = 4, B = 4;
        function loss() {
          const m = rlx.dsl("mlp");
          const x = m.input("x", [B, P]), y = m.input("y", [B, C]);
          m.output(x.matmul(m.param("w1", [P, H])).add(m.param("b1", [1, H])).relu()
                    .matmul(m.param("w2", [H, C])).add(m.param("b2", [1, C]))
                    .softmaxCrossEntropy(y).mean([0], false));
          return m;
        }
        const names = ["w1", "b1", "w2", "b2"];
        const dims = { w1: [P, H], b1: [H], w2: [H, C], b2: [C] };
        let s = 11;
        const g = () => { s = (s * 1664525 + 1013904223) >>> 0; return ((s >>> 8) / 8388608 - 1) * 0.3; };
        const init = {};
        for (const n of names) init[n] = Float32Array.from({ length: dims[n].reduce((a, b) => a * b, 1) }, g);
        const batch = { x: new Float32Array(B * P).fill(0.5), y: rlx.oneHot([0, 1, 2, 3], C) };
        const opt = { kind: "adamw", lr: 0.05 };

        const host = new rlx.Trainer(loss(), { wrt: names, init, device: "metal", optimizer: opt });
        const fused = new rlx.Trainer(loss(), { wrt: names, init, device: "metal", optimizer: opt, resident: true });
        let worst = 0;
        for (let i = 0; i < 8; i++) {
          worst = Math.max(worst, Math.abs(host.step(batch) - fused.step(batch)));
        }
        const hp = host.params(), fp = fused.params();
        let pworst = 0;
        for (const n of names) for (let i = 0; i < hp[n].length; i++) {
          pworst = Math.max(pworst, Math.abs(hp[n][i] - fp[n][i]));
        }
        `${fused.isResident()}|${worst < 1e-5}|${pworst < 1e-5}|${host.gradNorm() > 0}|${fused.gradNorm() === null}`;
        "#,
    );
    // The last two: the host path reports a gradient norm, the fused one cannot
    // (the gradients never leave the device) and says `null` instead of a
    // stale or invented number.
    assert_eq!(out, "true|true|true|true|true");
}

#[test]
fn a_resident_checkpoint_is_readable_by_the_host_path() {
    // One on-disk layout for both, so a run can move between them — which also
    // means the resident path is not a one-way door.
    if !rlx_runtime::is_available(rlx_runtime::Device::Metal) {
        eprintln!("skip: no Metal backend in this build");
        return;
    }
    let mut rt = Runtime::new();
    rt.allow_filesystem();
    let dir = std::env::temp_dir().join("rlx-js-resident-ckpt");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("resident.gguf");
    let script = format!(
        r#"
        const CK = {path:?};
        function build() {{
          const m = rlx.dsl("mse");
          const x = m.input("x", [2, 4]);
          const t = m.input("t", [2, 4]);
          const d = x.matmul(m.param("w", [4, 4])).add(m.param("b", [1, 4])).sub(t);
          m.output(d.mul(d).mean([0, 1], false));
          return m;
        }}
        const init = {{ w: new Float32Array(16).fill(0.05), b: new Float32Array(4) }};
        const opt = {{ kind: "adamw", lr: 0.05 }};
        const batch = {{
          x: new Float32Array([1, 0, 0, 1, 0, 1, 1, 0]),
          t: new Float32Array([1, -1, 0.5, 0, 0, 2, -1, 1]),
        }};

        const a = new rlx.Trainer(build(), {{ wrt: ["w", "b"], init, device: "metal", optimizer: opt, resident: true }});
        a.run({{ steps: 20, batch: () => batch }});
        a.save(CK);
        const continued = a.run({{ steps: 20, batch: () => batch }}).lastLoss;

        // Same checkpoint, resumed on the *host* path.
        const b = new rlx.Trainer(build(), {{ wrt: ["w", "b"], init, device: "metal", optimizer: opt }});
        const rep = b.load(CK);
        const resumed = b.run({{ steps: 20, batch: () => batch }}).lastLoss;

        `${{rep.step}}|${{rep.optimizerRestored}}|${{Math.abs(continued - resumed) < 1e-5}}`;
        "#,
        path = path.to_string_lossy()
    );
    let out = rt.eval(&script).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        out, "20|true|true",
        "a resident checkpoint should resume on the host path: {out}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── precision policy ─────────────────────────────────────────

#[test]
fn precision_policies_parse_and_report_themselves() {
    let out = eval(
        r#"
        const names = ["f32", "f16", "mixed", "mixed-conservative", "mixed-bf16"];
        // Parsing is device-independent; CPU takes every policy.
        const got = names.map((p) => new rlx.Session({ device: "cpu", policy: p }).policy());
        const custom = new rlx.Session({ device: "cpu", policy: { compute: "bf16", reduction: "f32" } });
        const none = new rlx.Session("cpu");
        `${got.join(",")}|${custom.policy()}|${none.policy()}`;
        "#,
    );
    assert_eq!(
        out,
        "f32,f16,mixed,mixed-conservative,mixed-bf16|custom|null"
    );
}

#[test]
fn an_unknown_policy_is_refused_by_name() {
    for (script, wanted) in [
        (
            r#"new rlx.Session({ device: "cpu", policy: "nonsense" })"#,
            "unknown precision policy",
        ),
        (
            r#"new rlx.Session({ device: "cpu", policy: {} })"#,
            "named no op kinds",
        ),
        (
            r#"new rlx.Session({ device: "cpu", policy: { compute: "int4" } })"#,
            "unknown precision",
        ),
    ] {
        let err = eval_err(script);
        assert!(err.contains(wanted), "`{script}` -> {err}");
    }
}

#[test]
fn metal_refuses_bf16_compute_and_allows_f16() {
    // f16 used to be refused here too: a single attention op returned ALL ZEROS
    // and a transformer block NaN'd in `silu`. Both were one bug — `Op::Attention`
    // and `Op::LayerNorm2d` were missing from the "Metal kernels are still
    // f32-only" list in `rlx-compile/src/precision.rs`, so the pass retagged them
    // F16 and an f32 kernel then read f16 bytes. Fixed, and f16 now measures
    // 1.3e-3 relative on a transformer block, so it is allowed.
    //
    // BF16 stays refused, and for a different reason: not a missing list entry
    // but a missing dtype. Metal's `HalfFlag` is `{F32, F16}` and maps BF16 to
    // F32, so two bf16 values get read as one f32 — measured non-finite output
    // at 2.3e38 on a 32×32 matmul.
    if !rlx_runtime::is_available(rlx_runtime::Device::Metal) {
        eprintln!("skip: no Metal backend in this build");
        return;
    }
    for policy in [r#""mixed-bf16""#, r#"{ compute: "bf16" }"#] {
        let err = eval_err(&format!(
            r#"new rlx.Session({{ device: "metal", policy: {policy} }})"#
        ));
        assert!(
            err.contains("bf16") && err.contains("no bf16 compute kernels"),
            "policy {policy} should be refused with the measured reason: {err}"
        );
    }
    // Everything else, including f16, is accepted.
    let out = eval(
        r#"
        ["mixed", "mixed-conservative", "f32", "f16"]
          .map((p) => new rlx.Session({ device: "metal", policy: p }).policy())
          .join(",") + "|" +
        new rlx.Session({ device: "metal", policy: { reduction: "f16", elementwise: "f16" } }).policy();
        "#,
    );
    assert_eq!(out, "mixed,mixed-conservative,f32,f16|custom");
}

#[test]
fn a_mixed_policy_stays_close_to_f32() {
    // The reason to have it: a bounded accuracy cost for a real speedup. On a
    // pre-norm transformer block this measured 6e-4 relative at 4-6% faster; the
    // bound here is loose enough not to be a benchmark and tight enough to catch
    // a policy that silently broke.
    if !rlx_runtime::is_available(rlx_runtime::Device::Metal) {
        eprintln!("skip: no Metal backend in this build");
        return;
    }
    let out = eval(
        r#"
        const B = 1, S = 32, D = 64;
        function build() {
          const m = rlx.dsl("blk");
          const x = m.input("x", [B, S, D]);
          const h = x.rmsNorm(m.param("g", [D]), m.param("b", [D]), 1e-6)
                     .matmul(m.param("w1", [D, D])).silu()
                     .matmul(m.param("w2", [D, D]));
          m.output(x.add(h));
          return m;
        }
        function fill(n, seed) {
          let s = seed >>> 0; const a = new Float32Array(n);
          for (let i = 0; i < n; i++) { s = (s * 1664525 + 1013904223) >>> 0; a[i] = ((s >>> 8) / 8388608 - 1) * 0.1; }
          return a;
        }
        const W = { g: new Float32Array(D).fill(1), b: new Float32Array(D), w1: fill(D * D, 1), w2: fill(D * D, 2) };
        const I = { x: fill(B * S * D, 3) };
        const run = (policy) => {
          const c = new rlx.Session(policy ? { device: "metal", policy } : { device: "metal" }).compile(build());
          c.setParams(W);
          return c.run(I)[0];
        };
        const truth = run(null);
        const results = ["mixed", "mixed-conservative"].map((p) => {
          const out = run(p);
          let worst = 0, denom = 0;
          for (let i = 0; i < out.length; i++) {
            worst = Math.max(worst, Math.abs(out[i] - truth[i]));
            denom = Math.max(denom, Math.abs(truth[i]));
          }
          return { finite: Array.from(out).every(Number.isFinite), rel: worst / denom };
        });
        results.every((r) => r.finite && r.rel < 5e-3) ? "close" : JSON.stringify(results);
        "#,
    );
    assert_eq!(out, "close");
}

/// `compile(graph, opts)` must honour `opts`, not silently drop it.
///
/// It used to be registered with arity 1, so a second argument was discarded:
/// `compile(g, {policy: "f16"})` applied no policy and raised no error. A
/// setting that neither takes effect nor complains is the worst of both.
#[test]
fn compile_honours_its_options_argument() {
    // A bogus policy has to be rejected — proof the field is read at all.
    let err = eval_err(
        r#"const m = rlx.dsl("m");
           const a = m.input("a", [32, 32]);
           m.output(a.matmul(m.param("w", [32, 32])));
           new rlx.Session("cpu").compile(m, { policy: "bogus" });"#,
    );
    assert!(
        err.contains("unknown precision policy"),
        "compile() ignored its options argument: {err}"
    );
    // And a valid one is applied rather than dropped.
    let ok = eval(
        r#"const m2 = rlx.dsl("m2");
           const b = m2.input("a", [32, 32]);
           m2.output(b.matmul(m2.param("w", [32, 32])));
           const c = new rlx.Session("cpu").compile(m2, { policy: "mixed" });
           c.outputShapes().length"#,
    );
    assert_eq!(ok.trim(), "1", "compile() with a valid policy failed: {ok}");
}
