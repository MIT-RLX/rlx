// Inspect a GGUF checkpoint from JavaScript.
//
//   rlx-js examples/gguf.js /path/to/model.gguf

const path = scriptArgs[0];
if (!path) {
  console.error("usage: rlx-js examples/gguf.js <model.gguf>");
} else {
  const f = rlx.openGguf(path);
  const meta = f.metadata();
  console.log(`${f} arch=${meta["general.architecture"]}`);

  const names = f.tensorNames();
  console.log(`${names.length} tensors; first five:`);
  for (const name of names.slice(0, 5)) {
    const { dims, dtype, elements } = f.info(name);
    console.log(`  ${name.padEnd(34)} [${dims}] ${dtype} (${elements})`);
  }

  // Dequantizing is per-tensor and on demand — the file itself is mmap'd.
  const first = f.tensor(names[0]);
  const mean = first.reduce((a, b) => a + b, 0) / first.length;
  console.log(`${names[0]}: ${first.length} f32, mean ${mean.toFixed(6)}`);
}
