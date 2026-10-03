// Train a real classifier from JavaScript: a 2→32→32→3 MLP on a three-arm
// spiral, with AdamW and fused softmax cross-entropy.
//
//   rlx-js examples/train_classifier.js
//   FEATURES=metal just js crates/bindings/rlx-js/examples/train_classifier.js
//
// `rlx.Trainer` differentiates and compiles the graph once, holds the weights,
// and returns the loss per step — the script only supplies batches.

const DEVICE = rlx.devices().includes("metal") ? "metal" : "cpu";
const N = 300;          // points per class
const CLASSES = 3;
const HIDDEN = 32;
const BATCH = 64;
const STEPS = 400;

// ── data: three interleaved spiral arms ──────────────────────
function spiral(seed) {
    let s = seed >>> 0;
    const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) >>> 8) / 16777216;

    const xs = [], ys = [];
    for (let c = 0; c < CLASSES; c++) {
        for (let i = 0; i < N; i++) {
            const r = (i / N) * 4;
            const t = (c * 2 * Math.PI) / CLASSES + (i / N) * 3 + rand() * 0.25;
            xs.push(r * Math.sin(t), r * Math.cos(t));
            ys.push(c);
        }
    }
    return { xs: Float32Array.from(xs), labels: Int32Array.from(ys), count: CLASSES * N };
}

const data = spiral(7);

// ── graph: loss = mean(softmaxCrossEntropy(mlp(x), onehot)) ──
function buildLoss(batch) {
    const g = new rlx.Graph("spiral-mlp");
    const x = g.input("x", [batch, 2], "f32");
    const y = g.input("y", [batch, CLASSES], "f32");     // one-hot targets

    const h1 = g.relu(g.add(g.matmul(x, g.param("w1", [2, HIDDEN], "f32")),
                            g.param("b1", [1, HIDDEN], "f32")));
    const h2 = g.relu(g.add(g.matmul(h1, g.param("w2", [HIDDEN, HIDDEN], "f32")),
                            g.param("b2", [1, HIDDEN], "f32")));
    const logits = g.add(g.matmul(h2, g.param("w3", [HIDDEN, CLASSES], "f32")),
                         g.param("b3", [1, CLASSES], "f32"));

    g.setOutputs([g.mean(g.softmaxCrossEntropy(logits, y), [0], false)]);
    return { g, logits };
}

// ── init: He-scaled, deterministic ───────────────────────────
function init(seed) {
    let s = seed >>> 0;
    const gauss = () => {
        // Box-Muller off a linear congruential stream, so runs reproduce.
        s = (s * 1664525 + 1013904223) >>> 0; const u = (s >>> 8) / 16777216 || 1e-7;
        s = (s * 1664525 + 1013904223) >>> 0; const v = (s >>> 8) / 16777216;
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    const w = (rows, cols) => {
        const a = new Float32Array(rows * cols);
        const scale = Math.sqrt(2 / rows);
        for (let i = 0; i < a.length; i++) a[i] = gauss() * scale;
        return a;
    };
    return {
        w1: w(2, HIDDEN), b1: new Float32Array(HIDDEN),
        w2: w(HIDDEN, HIDDEN), b2: new Float32Array(HIDDEN),
        w3: w(HIDDEN, CLASSES), b3: new Float32Array(CLASSES),
    };
}

const WRT = ["w1", "b1", "w2", "b2", "w3", "b3"];
const trainer = new rlx.Trainer(buildLoss(BATCH).g, {
    wrt: WRT,
    init: init(42),
    device: DEVICE,
    optimizer: { kind: "adamw", lr: 0.02, weightDecay: 1e-4 },
});

// ── batching ─────────────────────────────────────────────────
let cursor = 0;
const order = Int32Array.from({ length: data.count }, (_, i) => i);
function batch() {
    if (cursor + BATCH > data.count) {     // reshuffle each epoch
        for (let i = order.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [order[i], order[j]] = [order[j], order[i]];
        }
        cursor = 0;
    }
    const x = new Float32Array(BATCH * 2);
    const y = new Float32Array(BATCH * CLASSES);
    for (let b = 0; b < BATCH; b++) {
        const i = order[cursor + b];
        x[b * 2] = data.xs[i * 2];
        x[b * 2 + 1] = data.xs[i * 2 + 1];
        y[b * CLASSES + data.labels[i]] = 1;
    }
    cursor += BATCH;
    return { x, y };
}

// ── train ────────────────────────────────────────────────────
console.log(`${trainer} training ${WRT.length} tensors on ${trainer.device()}`);
const t0 = Date.now();
let first = null;
for (let step = 1; step <= STEPS; step++) {
    // Cosine decay, driven from JS — the optimizer's lr is settable per step.
    trainer.setLr(0.02 * 0.5 * (1 + Math.cos((Math.PI * step) / STEPS)));
    const loss = trainer.step(batch());
    if (first === null) first = loss;
    if (step % 100 === 0 || step === 1) {
        console.log(`  step ${String(step).padStart(4)}  loss ${loss.toFixed(4)}`);
    }
}
const ms = Date.now() - t0;

// ── accuracy: reuse the trained weights on an inference graph ─
const weights = trainer.params();
const infer = (() => {
    const g = new rlx.Graph("spiral-infer");
    const x = g.input("x", [data.count, 2], "f32");
    const h1 = g.relu(g.add(g.matmul(x, g.param("w1", [2, HIDDEN], "f32")),
                            g.param("b1", [1, HIDDEN], "f32")));
    const h2 = g.relu(g.add(g.matmul(h1, g.param("w2", [HIDDEN, HIDDEN], "f32")),
                            g.param("b2", [1, HIDDEN], "f32")));
    const logits = g.add(g.matmul(h2, g.param("w3", [HIDDEN, CLASSES], "f32")),
                         g.param("b3", [1, CLASSES], "f32"));
    g.setOutputs([g.argmax(logits, 1, false)]);
    const c = new rlx.Session(DEVICE).compile(g);
    c.setParams(weights);
    return c;
})();

const [pred] = infer.run({ x: data.xs });
let correct = 0;
for (let i = 0; i < data.count; i++) if (pred[i] === data.labels[i]) correct++;

console.log(`loss ${first.toFixed(4)} -> ${trainer.step(batch()).toFixed(4)} in ${ms}ms`);
console.log(`accuracy ${((100 * correct) / data.count).toFixed(1)}% on ${data.count} points`);
