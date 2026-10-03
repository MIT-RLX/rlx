// Gradient descent, driven entirely from JavaScript.
//
// `rlx.grad` rewrites the forward graph into one whose outputs are
// [loss, ...dWrt]; the update itself stays in JS, so the optimizer is
// whatever the script wants it to be.
//
//   rlx-js examples/train.js

const DEVICE = rlx.devices().includes("metal") ? "metal" : "cpu";

// Fit y = Wx against a fixed target: loss = mean((Wx - target)^2).
function buildLoss() {
  const g = new rlx.Graph("mse");
  const x = g.input("x", [1, 3], "f32");
  const target = g.input("target", [1, 2], "f32");
  const w = g.param("w", [3, 2], "f32");

  const diff = g.sub(g.matmul(x, w), target);
  g.setOutputs([g.mean(g.mul(diff, diff), [0, 1], false)]);
  return { g, w };
}

const { g, w } = buildLoss();
const backward = rlx.grad(g, [w]);           // [loss, dW]
const step = new rlx.Session(DEVICE).compile(backward);

const x = new Float32Array([1, 2, 3]);
const target = new Float32Array([1, -1]);
const seed = new Float32Array([1]);          // d(loss)/d(loss)

let weights = new Float32Array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]);
const lr = 0.02;

console.log(`training on ${DEVICE}`);
for (let i = 0; i <= 100; i++) {
  step.setParam("w", weights);
  const [loss, dW] = step.run({ x, target, d_output: seed });

  if (i % 25 === 0) console.log(`  step ${String(i).padStart(3)}  loss ${loss[0].toFixed(6)}`);

  for (let k = 0; k < weights.length; k++) weights[k] -= lr * dW[k];
}
console.log("W =", Array.from(weights).map((v) => v.toFixed(4)).join(", "));
