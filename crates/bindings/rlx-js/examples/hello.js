// A linear layer, end to end: build, compile, upload weights, run.
//   rlx-js examples/hello.js

console.log("backends:", rlx.devices().join(", "));

const g = new rlx.Graph("linear");
const x = g.input("x", [1, 4], "f32");
const w = g.param("w", [4, 2], "f32");
const b = g.param("b", [1, 2], "f32");
g.setOutputs([g.relu(g.add(g.matmul(x, w), b))]);

console.log("nodes:", g.nodeCount(), "picked:", rlx.fastestDeviceFor(g));

const compiled = new rlx.Session({ device: "cpu", precision: "f32" }).compile(g);
compiled.setParams({
  w: new Float32Array([1, 0, 0, 1, 1, 0, 0, 1]),
  b: new Float32Array([0, -10]),
});

const [y] = compiled.run({ x: new Float32Array([1, 2, 3, 4]) });
console.log("y =", Array.from(y));          // [4, 0] — the -10 bias is clamped by relu
console.log("shape:", compiled.outputShapes()[0]);
