// MNIST, trained from JavaScript on real data.
//
//   rlx-js examples/mnist.js                          # 784-128-10 MLP
//   rlx-js examples/mnist.js --cnn                    # 1->8->16 conv net
//   rlx-js examples/mnist.js --save ckpt.gguf         # checkpoint at the end
//   rlx-js examples/mnist.js --resume ckpt.gguf --steps 1000
//   rlx-js examples/mnist.js --patience 3             # early stop on validation
//
// Data: the idx-ubyte files from the MNIST distribution. The script searches the
// same cache directories rlx-vision-bench uses; if nothing is found it prints
// the fetch command and exits.
//
// Everything outside the graph is JavaScript: idx parsing, the train/validation
// split, batching, the LR schedule, early stopping and the accuracy sweep. RLX
// owns the graph, the backward pass, the optimizer and the checkpoint.

// ── arguments ────────────────────────────────────────────────
// `scriptArgs` holds the arguments the script was given, with the script path
// already removed — the same in `-e`, `-` and file mode.
const argv = scriptArgs;
const flag = (name, fallback) => {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : fallback;
};
const CNN = argv.includes("--cnn");
const STEPS = Number(flag("--steps", CNN ? 1500 : 3000));
const BATCH = Number(flag("--batch", 64));
const LR = Number(flag("--lr", CNN ? 1.5e-3 : 3e-3));
const DEVICE = flag("--device", rlx.devices().includes("metal") ? "metal" : "cpu");
const SAVE = flag("--save", null);
const RESUME = flag("--resume", null);
// 0 disables early stopping. Otherwise: stop after this many validation checks
// with no improvement.
const PATIENCE = Number(flag("--patience", 0));
const VAL_EVERY = Number(flag("--val-every", 250));
const VAL_N = Number(flag("--val", 2000));

const CLASSES = 10;
const HIDDEN = 128;
const C1 = 8, C2 = 16;
// MNIST's published statistics, the same constants torchvision uses, so the
// accuracy here is comparable to other reports.
const MEAN = 0.1307, STD = 0.3081;

// ── locating the dataset ─────────────────────────────────────
const FILES = {
  trainImages: "train-images-idx3-ubyte",
  trainLabels: "train-labels-idx1-ubyte",
  testImages: "t10k-images-idx3-ubyte",
  testLabels: "t10k-labels-idx1-ubyte",
};

function findDataDir() {
  const home = rlx.env("HOME") || ".";
  const candidates = [
    flag("--data", null),
    rlx.env("MNIST_DIR"),
    `${home}/.cache/torchvision-mnist/MNIST/raw`,
    `${home}/.cache/rlx-datasets/mnist`,
    "data",
  ].filter(Boolean);
  for (const dir of candidates) {
    // All four files, and non-empty: a truncated download is the failure mode
    // worth catching here rather than three screens later.
    if (Object.values(FILES).every((f) => (rlx.fileSize(`${dir}/${f}`) ?? 0) > 100)) return dir;
  }
  return null;
}

// ── idx-ubyte parsing ────────────────────────────────────────
// Header: 4-byte big-endian magic where byte 2 is the element type (0x08 =
// unsigned byte) and byte 3 is the rank, so images are 0x0000_0803 and labels
// 0x0000_0801. Then `rank` big-endian u32 dimensions, then raw u8 data.
function readIdx(path) {
  const bytes = rlx.readFile(path);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const magic = view.getUint32(0, false);
  if ((magic & 0xffffff00) !== 0x00000800) {
    throw new Error(`${path}: bad idx magic 0x${magic.toString(16)}`);
  }
  const rank = magic & 0xff;
  const dims = [];
  for (let i = 0; i < rank; i++) dims.push(view.getUint32(4 + 4 * i, false));
  const offset = 4 + 4 * rank;
  const count = dims.reduce((a, b) => a * b, 1);
  if (bytes.byteLength - offset !== count) {
    throw new Error(`${path}: expected ${count} bytes of data, found ${bytes.byteLength - offset}`);
  }
  return { dims, data: bytes.subarray(offset) };
}

/// Widen and normalize in one native pass.
///
/// `rlx.toFloat32` folds `(pixel / 255 - mean) / std` into the u8->f32 widening:
/// 21 ms for 47 M pixels, against 930 ms for `Float32Array.prototype.set` (which
/// only widens) and ~8.7 s for an interpreted loop.
function loadSplit(imagesPath, labelsPath) {
  const images = readIdx(imagesPath);
  const labels = readIdx(labelsPath);
  const [n, h, w] = images.dims;
  if (labels.dims[0] !== n) throw new Error("image/label count mismatch");
  const x = rlx.toFloat32(images.data, { scale: 1 / (255 * STD), bias: -MEAN / STD });
  return { n, h, w, pixels: h * w, x, y: labels.data };
}

// ── graphs ───────────────────────────────────────────────────
/// 784 -> 128 -> 10, ReLU, fused softmax cross-entropy.
function mlp(batch, pixels, head) {
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

/// conv(1->8) -> conv(8->16) -> dense. NCHW throughout; `stride: 2` stands in
/// for max-pooling — the same 4x spatial reduction in one op.
function cnn(batch, h, w, head) {
  const m = rlx.dsl("mnist-cnn");
  const x = m.input("x", [batch, 1, h, w]);
  const y = m.input("y", [batch, CLASSES]);
  const c1 = x.conv2d(m.param("c1", [C1, 1, 3, 3]), { kernelSize: 3, padding: 1, stride: 2 }).relu();
  const c2 = c1.conv2d(m.param("c2", [C2, C1, 3, 3]), { kernelSize: 3, padding: 1, stride: 2 }).relu();
  const flat = C2 * Math.ceil(h / 4) * Math.ceil(w / 4);
  const logits = c2.reshape([batch, flat])
                   .matmul(m.param("fc", [flat, CLASSES])).add(m.param("fcb", [1, CLASSES]));
  m.output(head === "loss" ? logits.softmaxCrossEntropy(y).mean([0], false)
                           : logits.argmax(1, false));
  return m;
}

function flatSize(h, w) {
  return C2 * Math.ceil(h / 4) * Math.ceil(w / 4);
}

// ── deterministic He init ────────────────────────────────────
function initializer(seed) {
  let s = seed >>> 0;
  const uniform = () => ((s = (s * 1664525 + 1013904223) >>> 0) >>> 8) / 16777216;
  const gauss = () => {
    const u = uniform() || 1e-7;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * uniform());
  };
  // Fan-in depends on the layout, and getting it backwards is not visible in the
  // shapes: a dense weight is [in, out] so fan-in is `shape[0]`, while a conv
  // kernel is [out, in, kh, kw] so it is everything *but* `shape[0]`. Using one
  // rule for both made the dense init 2.5x too large and the first loss 21.4
  // instead of ~3.3.
  return (shape) => {
    const n = shape.reduce((a, b) => a * b, 1);
    const fanIn = shape.length === 2 ? shape[0] : shape.length === 4 ? n / shape[0] : n;
    const scale = Math.sqrt(2 / fanIn);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = gauss() * scale;
    return out;
  };
}

function main(DATA) {
  console.log(`data    ${DATA}`);
  let t = Date.now();
  const all = loadSplit(`${DATA}/${FILES.trainImages}`, `${DATA}/${FILES.trainLabels}`);
  const test = loadSplit(`${DATA}/${FILES.testImages}`, `${DATA}/${FILES.testLabels}`);
  console.log(`loaded  ${all.n} train + ${test.n} test ${all.h}x${all.w} in ${Date.now() - t}ms`);

  // Held-out validation carved off the *training* split, so the test split stays
  // untouched by any decision the run makes (early stopping included).
  const valN = Math.min(VAL_N, Math.floor(all.n / 10));
  const trainN = all.n - valN;
  const pixels = all.pixels;

  const init = initializer(1234);
  const weights = CNN
    ? {
        c1: init([C1, 1, 3, 3]), c2: init([C2, C1, 3, 3]),
        fc: init([flatSize(all.h, all.w), CLASSES]), fcb: new Float32Array(CLASSES),
      }
    : {
        w1: init([pixels, HIDDEN]), b1: new Float32Array(HIDDEN),
        w2: init([HIDDEN, CLASSES]), b2: new Float32Array(CLASSES),
      };

  const buildLoss = () => (CNN ? cnn(BATCH, all.h, all.w, "loss") : mlp(BATCH, pixels, "loss"));
  const trainer = new rlx.Trainer(buildLoss(), {
    wrt: Object.keys(weights),
    init: weights,
    device: DEVICE,
    optimizer: { kind: "adamw", lr: LR, weightDecay: 1e-4 },
    clipNorm: 5.0,
  });

  if (RESUME) {
    const report = trainer.load(RESUME);
    console.log(`resume  ${RESUME}: step ${report.step}, ${report.restored} tensors, ` +
                `optimizer state ${report.optimizerRestored ? "restored" : "NOT restored"}`);
    if (!report.optimizerRestored && report.hadOptimizerState) {
      console.error("        (checkpoint carried optimizer state this trainer refused)");
    }
  }
  console.log(`model   ${CNN ? "cnn" : "mlp"} on ${trainer.device()}, ` +
              `${trainer.trainable().length} tensors, ${STEPS} steps of ${BATCH}, ` +
              `${trainN} train / ${valN} val`);

  // ── batching: shuffle once per epoch, assemble natively ──
  const order = new Int32Array(trainN);
  for (let i = 0; i < trainN; i++) order[i] = i;
  let cursor = trainN;
  const pick = new Int32Array(BATCH);
  const labels = new Int32Array(BATCH);

  function batch() {
    if (cursor + BATCH > trainN) {
      for (let i = trainN - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const t = order[i]; order[i] = order[j]; order[j] = t;
      }
      cursor = 0;
    }
    for (let b = 0; b < BATCH; b++) {
      pick[b] = order[cursor + b];
      labels[b] = all.y[pick[b]];
    }
    cursor += BATCH;
    // Both of these are one native call per batch rather than one JS call per
    // row: `gatherRows` is 52x faster than `subarray` + `set` in a loop.
    return { x: rlx.gatherRows(all.x, pick, pixels), y: rlx.oneHot(labels, CLASSES) };
  }

  // ── validation ──
  const buildEval = (n) => (CNN ? cnn(n, all.h, all.w, "argmax") : mlp(n, pixels, "argmax"));
  const valGraph = new rlx.Session(DEVICE).compile(buildEval(valN));
  const valIdx = new Int32Array(valN);
  for (let i = 0; i < valN; i++) valIdx[i] = trainN + i;
  const valX = rlx.gatherRows(all.x, valIdx, pixels);
  const valY = new Float32Array(valN * CLASSES);

  function accuracy(compiled, x, truth, n, offset) {
    compiled.setParams(trainer.params());
    const [pred] = compiled.run({ x, y: new Float32Array(n * CLASSES) });
    let correct = 0;
    for (let i = 0; i < n; i++) if (pred[i] === truth[offset + i]) correct++;
    return { pct: (100 * correct) / n, pred };
  }

  // ── train, in stretches, with early stopping ──
  //
  // The schedule is keyed on the step *within this invocation*, not the
  // trainer's absolute counter. Using the absolute one after `--resume` walked
  // back into the cosine's next period and ended the window at full LR, which
  // cost 2.6 points of test accuracy (98.00% -> 95.42%) while the training loss
  // stayed healthy — a schedule bug that does not show up in the loss curve.
  const startStep = trainer.steps();
  const t1 = Date.now();
  let best = -1, bestStep = 0, stale = 0, firstLoss = null, stoppedEarly = false;
  let done = 0;
  while (done < STEPS) {
    const chunk = PATIENCE > 0 ? Math.min(VAL_EVERY, STEPS - done) : STEPS;
    const result = trainer.run({
      steps: chunk,
      batch,
      // Linear warmup then cosine decay, driven from JS.
      lr: (step) => {
        const local = step - startStep;
        const warm = Math.min(1, (local + 1) / 100);
        const decay = 0.5 * (1 + Math.cos((Math.PI * local) / STEPS));
        return LR * warm * Math.max(decay, 0.02);
      },
      every: Math.max(1, Math.round(chunk / 6)),
      onStep: (step, loss) => {
        console.log(`  step ${String(step + 1).padStart(5)}  loss ${loss.toFixed(4)}`);
        return true;
      },
    });
    if (firstLoss === null) firstLoss = result.firstLoss;
    done += result.steps;

    if (PATIENCE > 0) {
      const val = accuracy(valGraph, valX, all.y, valN, trainN).pct;
      const improved = val > best;
      console.log(`  val   ${val.toFixed(2)}%${improved ? "  (best)" : `  (${stale + 1}/${PATIENCE} stale)`}`);
      if (improved) {
        best = val; bestStep = done; stale = 0;
        if (SAVE) trainer.save(SAVE);        // keep the best, not the last
      } else if (++stale >= PATIENCE) {
        stoppedEarly = true;
        break;
      }
    }
    if (result.stopped) break;
  }
  const trainMs = Date.now() - t1;

  // ── test ──
  // With early stopping the best weights are the checkpoint's, not the
  // trainer's: it kept going past the peak by `PATIENCE` checks. Reporting the
  // last weights while having saved the best is how a run gets credited with a
  // number it did not produce.
  if (stoppedEarly && SAVE) {
    trainer.load(SAVE);
    console.log(`reload  best checkpoint from step ${trainer.steps()} for the test sweep`);
  }
  const testGraph = new rlx.Session(DEVICE).compile(buildEval(test.n));
  const { pct } = accuracy(testGraph, test.x, test.y, test.n, 0);
  const testPred = accuracy(testGraph, test.x, test.y, test.n, 0).pred;
  const missed = new Int32Array(CLASSES);
  for (let i = 0; i < test.n; i++) if (testPred[i] !== test.y[i]) missed[test.y[i]]++;

  console.log(`loss    ${firstLoss.toFixed(4)} -> ${trainer.evaluate(batch()).toFixed(4)}` +
              `  gradNorm ${trainer.gradNorm().toExponential(2)}`);
  console.log(`time    ${trainMs}ms for ${done} steps (${(1000 * done / trainMs).toFixed(0)} steps/s)`);
  if (PATIENCE > 0) {
    console.log(`val     best ${best.toFixed(2)}% at step ${bestStep}` +
                `${stoppedEarly ? ` — stopped early, ${PATIENCE} checks without improvement` : ""}`);
  }
  console.log(`test    ${pct.toFixed(2)}% on ${test.n} held-out digits`);
  const worst = [...missed.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3);
  console.log(`missed  ${worst.map(([d, n]) => `${d}:${n}`).join("  ")}`);

  if (SAVE && PATIENCE === 0) {
    const n = trainer.save(SAVE);
    console.log(`saved   ${n} tensors to ${SAVE} at step ${trainer.steps()}` +
                ` (${(rlx.fileSize(SAVE) / 1e6).toFixed(1)} MB)`);
  }
}

// Entry point last: `main` is hoisted, but the `const` bindings it closes over
// are not — calling it above their declarations is a TDZ error.
const DATA = findDataDir();
if (!DATA) {
  console.error("MNIST not found. Fetch it with:\n");
  console.error("  D=\"$HOME/.cache/torchvision-mnist/MNIST/raw\"; mkdir -p \"$D\" && cd \"$D\"");
  console.error("  for f in " + Object.values(FILES).join(" ") + "; do \\");
  console.error("    curl -sSfL \"https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz\" -o \"$f.gz\" && gunzip -f \"$f.gz\"; done\n");
  console.error("Or pass --data <dir> / set $MNIST_DIR.");
} else {
  main(DATA);
}
