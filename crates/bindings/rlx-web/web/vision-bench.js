// RLX vision benchmark — live charts + multi-backend training (CPU / WebGPU / WebGL).

import Rlx from "./rlx.js";

const $ = (id) => document.getElementById(id);
const COLORS = {
  cpu: "#2563eb",
  webgpu: "#059669",
  webgl: "#d97706",
  sps: "#7c3aed",
  grid: "#ecece8",
  axis: "#9a9a94",
  text: "#5a5a5a",
};

const maxDiff = (a, b) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

const yieldUi = () => new Promise((r) => setTimeout(r, 0));

/** Lightweight multi-series line chart (no external deps). */
class LineChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.series = new Map(); // name -> { color, ys: number[] }
  }

  clear() {
    this.series.clear();
    this.draw();
  }

  setSeries(name, color, ys) {
    this.series.set(name, { color, ys: ys.slice() });
    this.draw();
  }

  append(name, color, y) {
    const s = this.series.get(name) || { color, ys: [] };
    s.color = color;
    s.ys.push(y);
    this.series.set(name, s);
    this.draw();
  }

  draw() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const pad = { l: 44, r: 12, t: 10, b: 22 };
    ctx.clearRect(0, 0, w, h);

    const all = [];
    for (const s of this.series.values()) all.push(...s.ys);
    if (!all.length) {
      ctx.fillStyle = COLORS.text;
      ctx.font = "12px ui-monospace, monospace";
      ctx.fillText("waiting for steps…", pad.l, h / 2);
      return;
    }

    let ymin = Math.min(...all);
    let ymax = Math.max(...all);
    if (ymin === ymax) {
      ymin -= 1;
      ymax += 1;
    }
    const padY = (ymax - ymin) * 0.08;
    ymin -= padY;
    ymax += padY;
    const xmax = Math.max(1, ...[...this.series.values()].map((s) => s.ys.length - 1));

    const xOf = (i) => pad.l + (i / xmax) * (w - pad.l - pad.r);
    const yOf = (v) => pad.t + (1 - (v - ymin) / (ymax - ymin)) * (h - pad.t - pad.b);

    // grid
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const y = pad.t + ((h - pad.t - pad.b) * g) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(w - pad.r, y);
      ctx.stroke();
    }

    // axes labels
    ctx.fillStyle = COLORS.text;
    ctx.font = "11px ui-monospace, monospace";
    ctx.textAlign = "right";
    ctx.fillText(ymax.toFixed(3), pad.l - 6, pad.t + 10);
    ctx.fillText(ymin.toFixed(3), pad.l - 6, h - pad.b);
    ctx.textAlign = "left";
    ctx.fillText("0", pad.l, h - 6);
    ctx.textAlign = "right";
    ctx.fillText(String(xmax), w - pad.r, h - 6);

    for (const [, s] of this.series) {
      if (s.ys.length < 1) continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      s.ys.forEach((v, i) => {
        const x = xOf(i);
        const y = yOf(v);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      // last point
      const li = s.ys.length - 1;
      ctx.fillStyle = s.color;
      ctx.beginPath();
      ctx.arc(xOf(li), yOf(s.ys[li]), 3.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function setProgress(phase, fraction, meta) {
  $("progress-panel").hidden = false;
  $("phase").textContent = phase;
  $("bar").style.width = `${Math.max(0, Math.min(100, fraction * 100)).toFixed(1)}%`;
  $("progress-meta").textContent = meta || "—";
}

function appendLog(line) {
  const el = $("log");
  if (el.textContent === "—" || el.textContent === "") el.textContent = line;
  else el.textContent += "\n" + line;
  el.scrollTop = el.scrollHeight;
}

function row(backend, fwd, train, bench) {
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${backend}</td><td>${fwd}</td><td>${train}</td><td>${bench}</td>`;
  $("rows").appendChild(tr);
}

async function trainBackend(model, backend, x, label, params, steps, lr, charts, onTick) {
  let p = params;
  const losses = [];
  const spsHist = [];
  const t0 = performance.now();
  let lastLoss = NaN;
  const color = COLORS[backend] || COLORS.cpu;
  const reportEvery = steps > 80 ? 5 : steps > 40 ? 2 : 1;

  for (let i = 0; i < steps; i++) {
    let step;
    if (backend === "cpu") step = model.trainStepCpu(x, label, p, lr);
    else if (backend === "webgpu") step = await model.trainStepWebgpu(x, label, p, lr);
    else if (backend === "webgl") step = model.trainStepWebgl(x, label, p, lr);
    else throw new Error(`unknown backend ${backend}`);

    p = step.params;
    lastLoss = step.loss;
    losses.push(lastLoss);

    const done = i + 1;
    if (done === 1 || done === steps || done % reportEvery === 0) {
      const elapsed = (performance.now() - t0) / 1000;
      const sps = done / Math.max(elapsed, 1e-6);
      spsHist.push(sps);
      charts.loss.setSeries(backend, color, losses);
      charts.sps.setSeries("sps", COLORS.sps, spsHist);
      onTick({
        backend,
        step: done,
        steps,
        loss: lastLoss,
        initialLoss: losses[0],
        stepsPerSec: sps,
      });
      await yieldUi();
    }
  }

  const elapsed = (performance.now() - t0) / 1000;
  return {
    initialLoss: losses[0],
    finalLoss: lastLoss,
    stepsPerSec: steps / Math.max(elapsed, 1e-6),
    losses,
    params: p,
  };
}

function detectGpu(rlx) {
  const webgpu =
    typeof rlx.wasm.init_webgpu === "function" &&
    typeof rlx.wasm.VisionBench !== "undefined";
  // VisionBench methods appear after construction; check prototype / feature flags via mlp helpers.
  const hasGpuFwd = typeof rlx.wasm.mlp_forward_gpu === "function";
  const hasGlFwd = typeof rlx.wasm.mlp_forward_webgl === "function";
  return {
    webgpuBundle: hasGpuFwd || webgpu,
    webglBundle: hasGlFwd,
  };
}

async function main() {
  const rlx = await Rlx.init({ webgpu: true });
  $("status").textContent = "wasm ready";
  $("status").className = "ok";
  $("pref").textContent = rlx.preferredBackend();

  const gpu = detectGpu(rlx);
  const feats = [];
  if (gpu.webgpuBundle) feats.push(rlx.webgpuReady ? "webgpu✓" : "webgpu (no adapter)");
  else feats.push("webgpu✗");
  if (gpu.webglBundle) feats.push("webgl✓");
  else feats.push("webgl✗");
  feats.push("cpu✓");
  $("gpu-feat").textContent = feats.join(", ");

  const select = $("model");
  for (const slug of rlx.listVisionModels()) {
    const opt = document.createElement("option");
    opt.value = slug;
    opt.textContent = `${slug} — ${rlx.modelInfo(slug).title}`;
    select.appendChild(opt);
  }

  const lossChart = new LineChart($("loss-chart"));
  const spsChart = new LineChart($("sps-chart"));

  $("run").addEventListener("click", async () => {
    const btn = $("run");
    btn.disabled = true;
    $("rows").innerHTML = "";
    $("log").textContent = "—";
    lossChart.clear();
    spsChart.clear();

    const slug = select.value;
    const model = rlx.vision(slug);
    const steps = Number($("steps").value) || 40;
    const seed = 42;
    const lr = 0.01;
    const mode = $("train-backends").value;
    const charts = { loss: lossChart, sps: spsChart };

    const wantGpu = mode !== "cpu";
    const wantGl = mode === "all";

    try {
      appendLog(`${model.info.title}`);
      appendLog(
        `input=${model.info.inputLen} · params=${model.info.paramFlatLen} · classes=${model.info.numClasses}`,
      );

      setProgress("Initializing…", 0.02, `seed=${seed}`);
      await yieldUi();
      const params0 = model.initParams(seed);
      const { x, label } = model.syntheticBatch(seed);

      setProgress("CPU forward…", 0.06, "compile + run");
      await yieldUi();
      const tFwd = performance.now();
      const cpuLogits = model.forwardCpu(x, params0);
      const fwdMs = (performance.now() - tFwd).toFixed(1);
      appendLog(`CPU forward ${fwdMs} ms`);

      // --- CPU train ---
      setProgress(`CPU training (0/${steps})…`, 0.1, `lr=${lr}`);
      const cpuBench = await trainBackend(
        model,
        "cpu",
        x,
        label,
        params0,
        steps,
        lr,
        charts,
        ({ step, steps: n, loss, initialLoss, stepsPerSec }) => {
          setProgress(
            `CPU training (${step}/${n})…`,
            0.1 + 0.35 * (step / n),
            `loss=${loss.toFixed(6)} · Δ=${(loss - initialLoss).toFixed(6)} · ${stepsPerSec.toFixed(1)}/s`,
          );
          if (step === 1 || step === n || step % 10 === 0) {
            appendLog(`  [cpu] step ${step}/${n}  loss=${loss.toFixed(6)}  (${stepsPerSec.toFixed(1)}/s)`);
          }
        },
      );
      row(
        "cpu",
        `logits[0..2]=[${[...cpuLogits.slice(0, 3)].map((v) => v.toFixed(4)).join(", ")}]`,
        cpuBench.losses[0].toFixed(6),
        `${cpuBench.initialLoss.toFixed(4)} → ${cpuBench.finalLoss.toFixed(4)} @ ${cpuBench.stepsPerSec.toFixed(1)}/s`,
      );

      // --- WebGPU ---
      if (wantGpu && gpu.webgpuBundle && rlx.webgpuReady) {
        try {
          setProgress("WebGPU forward…", 0.5, "async");
          await yieldUi();
          const t0 = performance.now();
          const gpuLogits = await model.forwardWebgpu(x, params0);
          const ms = (performance.now() - t0).toFixed(1);
          appendLog(`WebGPU forward ${ms} ms · max|Δ|=${maxDiff(gpuLogits, cpuLogits).toExponential(2)}`);

          setProgress(`WebGPU training (0/${steps})…`, 0.55, "");
          const gpuBench = await trainBackend(
            model,
            "webgpu",
            x,
            label,
            params0,
            steps,
            lr,
            charts,
            ({ step, steps: n, loss, initialLoss, stepsPerSec }) => {
              setProgress(
                `WebGPU training (${step}/${n})…`,
                0.55 + 0.25 * (step / n),
                `loss=${loss.toFixed(6)} · Δ=${(loss - initialLoss).toFixed(6)} · ${stepsPerSec.toFixed(1)}/s`,
              );
              if (step === 1 || step === n || step % 10 === 0) {
                appendLog(
                  `  [webgpu] step ${step}/${n}  loss=${loss.toFixed(6)}  (${stepsPerSec.toFixed(1)}/s)`,
                );
              }
            },
          );
          row(
            "webgpu",
            `max|Δ|=${maxDiff(gpuLogits, cpuLogits).toExponential(2)} (${ms} ms)`,
            gpuBench.losses[0].toFixed(6),
            `${gpuBench.initialLoss.toFixed(4)} → ${gpuBench.finalLoss.toFixed(4)} @ ${gpuBench.stepsPerSec.toFixed(1)}/s`,
          );
        } catch (e) {
          row("webgpu", "—", String(e), "—");
          appendLog(`WebGPU: ${e}`);
        }
      } else if (wantGpu) {
        const why = !gpu.webgpuBundle
          ? "not built (rebuild with --webgpu / --all)"
          : "no adapter";
        row("webgpu", why, "—", "—");
        appendLog(`WebGPU: ${why}`);
      }

      // --- WebGL ---
      if (wantGl && gpu.webglBundle) {
        try {
          setProgress("WebGL forward…", 0.85, "");
          await yieldUi();
          const glLogits = model.forwardWebgl(x, params0);
          appendLog(`WebGL forward · max|Δ|=${maxDiff(glLogits, cpuLogits).toExponential(2)}`);

          let trainCell = "forward only";
          let benchCell = "—";
          if (model.info.webglTrain) {
            setProgress(`WebGL training (0/${steps})…`, 0.88, "");
            const glBench = await trainBackend(
              model,
              "webgl",
              x,
              label,
              params0,
              steps,
              lr,
              charts,
              ({ step, steps: n, loss, initialLoss, stepsPerSec }) => {
                setProgress(
                  `WebGL training (${step}/${n})…`,
                  0.88 + 0.1 * (step / n),
                  `loss=${loss.toFixed(6)} · Δ=${(loss - initialLoss).toFixed(6)} · ${stepsPerSec.toFixed(1)}/s`,
                );
              },
            );
            trainCell = glBench.losses[0].toFixed(6);
            benchCell = `${glBench.initialLoss.toFixed(4)} → ${glBench.finalLoss.toFixed(4)} @ ${glBench.stepsPerSec.toFixed(1)}/s`;
          }
          row(
            "webgl",
            `max|Δ|=${maxDiff(glLogits, cpuLogits).toExponential(2)}`,
            trainCell,
            benchCell,
          );
        } catch (e) {
          const msg = String(e);
          row("webgl", msg.includes("not enabled") ? "not built" : msg, "—", "—");
          appendLog(`WebGL: ${msg}`);
        }
      } else if (wantGl) {
        row("webgl", "not built (rebuild with --webgl / --all)", "—", "—");
        appendLog("WebGL: not built");
      }

      setProgress("Done", 1, "all selected backends finished");
      $("status").textContent = "bench complete";
      $("status").className = "ok";
    } catch (e) {
      $("status").textContent = "error: " + e;
      $("status").className = "err";
      setProgress("Failed", 0, String(e));
      appendLog(`ERROR: ${e}`);
      console.error(e);
    } finally {
      btn.disabled = false;
    }
  });
}

main().catch((e) => {
  $("status").textContent = "error: " + e;
  $("status").className = "err";
  console.error(e);
});
