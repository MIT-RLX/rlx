// LoRA from JavaScript: adapt a frozen MNIST model to a shifted task.
//
//   rlx-js examples/mnist.js --save /tmp/mnist-mlp.gguf     # make a base first
//   rlx-js examples/mnist_lora.js /tmp/mnist-mlp.gguf
//
// The story, in order:
//   1. a base MLP trained on normal MNIST, loaded from a checkpoint
//   2. evaluated on *inverted* MNIST (255 - pixel) — it collapses
//   3. LoRA-adapted to the inverted task with the base weights FROZEN, training
//      only the rank-8 factors: 8 416 parameters against the base's 101 632
//   4. merged back into a plain weight and checked to agree with the adapter
//
// Two different kinds of "frozen" appear here, and the difference matters:
//   * left out of `wrt`  -> no gradient is computed at all (the base weights)
//   * `trainer.freeze()` -> gradient computed, update skipped (reversible)

const CKPT = scriptArgs[0] || "/tmp/mnist-mlp.gguf";
const RANK = Number(scriptArgs.includes("--rank") ? scriptArgs[scriptArgs.indexOf("--rank") + 1] : 8);
const STEPS = Number(scriptArgs.includes("--steps") ? scriptArgs[scriptArgs.indexOf("--steps") + 1] : 600);
const DEVICE = rlx.devices().includes("metal") ? "metal" : "cpu";

const CLASSES = 10, HIDDEN = 128, BATCH = 64;
const MEAN = 0.1307, STD = 0.3081;
// alpha/rank, the usual LoRA scaling: keeps the adapter's contribution
// comparable as the rank changes.
const ALPHA = 16;
const SCALE = ALPHA / RANK;

function findData() {
  const home = rlx.env("HOME") || ".";
  for (const dir of [rlx.env("MNIST_DIR"), `${home}/.cache/torchvision-mnist/MNIST/raw`,
                     `${home}/.cache/rlx-datasets/mnist`, "data"].filter(Boolean)) {
    if ((rlx.fileSize(`${dir}/t10k-images-idx3-ubyte`) ?? 0) > 100) return dir;
  }
  return null;
}

function readIdx(path) {
  const bytes = rlx.readFile(path);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const rank = view.getUint32(0, false) & 0xff;
  const dims = [];
  for (let i = 0; i < rank; i++) dims.push(view.getUint32(4 + 4 * i, false));
  return { dims, data: bytes.subarray(4 + 4 * rank) };
}

/// `invert` flips the pixel scale, which is the distribution shift the adapter
/// has to absorb: the same digits, the opposite polarity.
function split(dir, images, labels, invert) {
  const im = readIdx(`${dir}/${images}`), lb = readIdx(`${dir}/${labels}`);
  const [n, h, w] = im.dims;
  const x = invert
    ? rlx.toFloat32(im.data, { scale: -1 / (255 * STD), bias: (1 - MEAN) / STD })
    : rlx.toFloat32(im.data, { scale: 1 / (255 * STD), bias: -MEAN / STD });
  return { n, pixels: h * w, x, y: lb.data };
}

// ── graphs ───────────────────────────────────────────────────

/// The plain model, used for the base and for the merged check.
function dense(batch, pixels, head) {
  const m = rlx.dsl("mnist-mlp");
  const x = m.input("x", [batch, pixels]);
  const y = m.input("y", [batch, CLASSES]);
  const logits = x
    .matmul(m.param("w1", [pixels, HIDDEN])).add(m.param("b1", [1, HIDDEN])).relu()
    .matmul(m.param("w2", [HIDDEN, CLASSES])).add(m.param("b2", [1, CLASSES]));
  m.output(head === "loss" ? logits.softmaxCrossEntropy(y).mean([0], false)
                           : logits.argmax(1, false));
  return m;
}

/// The same model with both dense layers wrapped in `Op::LoraMatMul`:
/// `out = x·W + scale·(x·A)·B`, one fused op rather than three.
function lora(batch, pixels, head) {
  const m = rlx.dsl("mnist-lora");
  const x = m.input("x", [batch, pixels]);
  const y = m.input("y", [batch, CLASSES]);

  const h = x.loraMatmul(
      m.param("w1", [pixels, HIDDEN]),
      m.param("a1", [pixels, RANK]),
      m.param("b1lora", [RANK, HIDDEN]),
      SCALE, [batch, HIDDEN], "f32",
    ).add(m.param("b1", [1, HIDDEN])).relu();

  const logits = h.loraMatmul(
      m.param("w2", [HIDDEN, CLASSES]),
      m.param("a2", [HIDDEN, RANK]),
      m.param("b2lora", [RANK, CLASSES]),
      SCALE, [batch, CLASSES], "f32",
    ).add(m.param("b2", [1, CLASSES]));

  m.output(head === "loss" ? logits.softmaxCrossEntropy(y).mean([0], false)
                           : logits.argmax(1, false));
  return m;
}

/// `w + scale * A·B`, evaluated as a one-shot graph so the fold is done by the
/// same kernels that trained it rather than by a JS loop.
function mergeLora(base, a, b, inDim, rank, outDim) {
  const m = rlx.dsl("merge");
  const w = m.input("w", [inDim, outDim]);
  const av = m.input("a", [inDim, rank]);
  const bv = m.input("b", [rank, outDim]);
  m.output(w.add(av.matmul(bv).mul(SCALE)));
  const c = new rlx.Session(DEVICE).compile(m);
  return c.run({ w: base, a, b })[0];
}

function accuracy(graph, params, data) {
  const c = new rlx.Session(DEVICE).compile(graph);
  c.setParams(params);
  const [pred] = c.run({ x: data.x, y: new Float32Array(data.n * CLASSES) });
  let ok = 0;
  for (let i = 0; i < data.n; i++) if (pred[i] === data.y[i]) ok++;
  return (100 * ok) / data.n;
}

function gauss(seedRef) {
  let s = seedRef.s;
  const u = ((s = (s * 1664525 + 1013904223) >>> 0) >>> 8) / 16777216 || 1e-7;
  const v = ((s = (s * 1664525 + 1013904223) >>> 0) >>> 8) / 16777216;
  seedRef.s = s;
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function main(dir) {
  // ── 1. the base model ──
  if (!rlx.fileExists(CKPT)) {
    console.error(`no base checkpoint at ${CKPT}`);
    console.error("make one with:  rlx-js examples/mnist.js --save " + CKPT);
    return;
  }
  const trainNormal = split(dir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", false);
  const testNormal = split(dir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", false);
  const trainInverted = split(dir, "train-images-idx3-ubyte", "train-labels-idx1-ubyte", true);
  const testInverted = split(dir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", true);
  const pixels = trainNormal.pixels;

  // A throwaway trainer is the loader: `Trainer.load` is what reads a
  // checkpoint, and the graph it was written from is the one that defines the
  // parameter names.
  const zeros = {
    w1: new Float32Array(pixels * HIDDEN), b1: new Float32Array(HIDDEN),
    w2: new Float32Array(HIDDEN * CLASSES), b2: new Float32Array(CLASSES),
  };
  const loader = new rlx.Trainer(dense(BATCH, pixels, "loss"), {
    wrt: ["w1", "b1", "w2", "b2"], init: zeros, device: DEVICE,
    optimizer: { kind: "adamw", lr: 1e-3 },
  });
  const report = loader.load(CKPT);
  const base = loader.params();
  const baseCount = Object.values(base).reduce((a, v) => a + v.length, 0);
  console.log(`base    ${CKPT} @ step ${report.step}, ${baseCount} parameters`);
  console.log(`        normal MNIST   ${accuracy(dense(testNormal.n, pixels, "argmax"), base, testNormal).toFixed(2)}%`);
  const before = accuracy(dense(testInverted.n, pixels, "argmax"), base, testInverted);
  console.log(`        inverted MNIST ${before.toFixed(2)}%   <- the shift it has never seen`);

  // ── 2. LoRA adapter ──
  // `B` starts at zero so the adapter is an exact no-op at step 0: the adapted
  // model begins identical to the base, and only improves from there. Starting
  // both factors random would perturb a working model before it learns anything.
  const seed = { s: 99 };
  const aScale = (fanIn) => Math.sqrt(1 / fanIn);
  const adapter = {
    a1: Float32Array.from({ length: pixels * RANK }, () => gauss(seed) * aScale(pixels)),
    b1lora: new Float32Array(RANK * HIDDEN),
    a2: Float32Array.from({ length: HIDDEN * RANK }, () => gauss(seed) * aScale(HIDDEN)),
    b2lora: new Float32Array(RANK * CLASSES),
  };
  const adapterCount = Object.values(adapter).reduce((a, v) => a + v.length, 0);

  // The base weights go in `init` but *not* in `wrt`: no gradient is computed
  // for them at all, which is the cheap kind of frozen.
  const trainer = new rlx.Trainer(lora(BATCH, pixels, "loss"), {
    wrt: Object.keys(adapter),
    init: { ...base, ...adapter },
    device: DEVICE,
    optimizer: { kind: "adamw", lr: 3e-3, weightDecay: 0 },
    clipNorm: 1.0,
  });
  console.log(`adapter rank ${RANK}, alpha ${ALPHA} -> ${adapterCount} trainable ` +
              `(${((100 * adapterCount) / baseCount).toFixed(1)}% of the base), ` +
              `frozen: ${Object.keys(base).join(", ")}`);

  // ── 3. adapt ──
  const order = new Int32Array(trainInverted.n);
  for (let i = 0; i < order.length; i++) order[i] = i;
  let cursor = order.length;
  const pick = new Int32Array(BATCH), labels = new Int32Array(BATCH);
  function batch() {
    if (cursor + BATCH > order.length) {
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const t = order[i]; order[i] = order[j]; order[j] = t;
      }
      cursor = 0;
    }
    for (let b = 0; b < BATCH; b++) { pick[b] = order[cursor + b]; labels[b] = trainInverted.y[pick[b]]; }
    cursor += BATCH;
    return { x: rlx.gatherRows(trainInverted.x, pick, pixels), y: rlx.oneHot(labels, CLASSES) };
  }

  const t0 = Date.now();
  const result = trainer.run({
    steps: STEPS,
    batch,
    lr: (step) => 3e-3 * 0.5 * (1 + Math.cos((Math.PI * step) / STEPS)),
    every: Math.max(1, Math.round(STEPS / 5)),
    onStep: (step, loss) => {
      console.log(`  step ${String(step + 1).padStart(4)}  loss ${loss.toFixed(4)}`);
      return true;
    },
  });
  const adapted = trainer.params();
  const after = accuracy(lora(testInverted.n, pixels, "argmax"), adapted, testInverted);
  console.log(`adapt   ${result.firstLoss.toFixed(4)} -> ${result.lastLoss.toFixed(4)} ` +
              `in ${Date.now() - t0}ms (${result.stepsPerSecond.toFixed(0)} steps/s)`);
  console.log(`        inverted MNIST ${before.toFixed(2)}% -> ${after.toFixed(2)}%`);

  // Base weights must be byte-identical: they were never in `wrt`.
  const baseUntouched = Object.keys(base).every((k) =>
    Array.from(base[k]).every((v, i) => v === adapted[k][i]));
  console.log(`        base weights untouched: ${baseUntouched}`);

  // ── 4. merge ──
  // Folding the adapter into the weights gives a plain model with no adapter op
  // and no extra parameters at inference. It must score the same.
  const merged = {
    w1: mergeLora(adapted.w1, adapted.a1, adapted.b1lora, pixels, RANK, HIDDEN),
    b1: adapted.b1,
    w2: mergeLora(adapted.w2, adapted.a2, adapted.b2lora, HIDDEN, RANK, CLASSES),
    b2: adapted.b2,
  };
  const mergedAcc = accuracy(dense(testInverted.n, pixels, "argmax"), merged, testInverted);
  console.log(`merged  dense model, no adapter op: ${mergedAcc.toFixed(2)}% ` +
              `(${Math.abs(mergedAcc - after) < 0.05 ? "agrees" : `DIVERGES by ${(mergedAcc - after).toFixed(2)}`})`);

  // ── 5. freeze / unfreeze, the reversible kind ──
  // The adapter's own factors can be frozen mid-run: the gradient is still
  // computed (`wrt` is baked into the compiled graph) but the update is skipped.
  trainer.freeze(["a1", "a2"]);
  const heldBefore = Array.from(trainer.params().a1.slice(0, 4));
  trainer.run({ steps: 20, batch });
  const heldAfter = Array.from(trainer.params().a1.slice(0, 4));
  const bMoved = trainer.params().b1lora.some((v, i) => v !== adapted.b1lora[i]);
  console.log(`freeze  ${JSON.stringify(trainer.frozen())} held: ${heldBefore.join() === heldAfter.join()}` +
              `, unfrozen factors still moving: ${bMoved}`);
}

const DIR = findData();
if (!DIR) {
  console.error("MNIST not found — see examples/mnist.js for the fetch command.");
} else {
  main(DIR);
}
