"""Run the same graph on every available backend and print latencies.

Usage::

    maturin develop --release --features cpu,blas-accelerate,metal,mlx,gpu
    python pyrlx/examples/cross_backend_parity.py
"""

import time
import numpy as np

import pyrlx as rlx


def build_graph():
    g = rlx.Graph("mm_bias_gelu")
    x   = g.input("x", [128, 768], "f32")
    w   = g.param("w", [768, 768], "f32")
    b   = g.param("b", [768],      "f32")
    out = g.gelu(g.add(g.matmul(x, w), b))
    g.set_outputs([out])
    return g


def run(device: str, x: np.ndarray, w: np.ndarray, b: np.ndarray, iters: int = 50):
    sess     = rlx.Session(device=device)
    compiled = sess.compile(build_graph())
    compiled.set_param("w", w)
    compiled.set_param("b", b)
    inp = {"x": x}

    [out] = compiled.run(inp)         # warm-up + correctness
    t0 = time.perf_counter()
    for _ in range(iters):
        compiled.run(inp)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / iters
    return out, elapsed_ms


def main():
    devs = rlx.available_devices()
    print(f"available backends: {devs}")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((128, 768)).astype(np.float32)
    # Cast last so the divide stays in f32 (otherwise numpy promotes to f64).
    w = (rng.standard_normal((768, 768)) / np.sqrt(768)).astype(np.float32)
    b = np.zeros(768, dtype=np.float32)

    cpu_out, cpu_ms = run("cpu", x, w, b)
    print(f"  cpu   {cpu_ms:7.3f} ms/iter   (reference)")

    for dev in devs:
        if dev == "cpu":
            continue
        try:
            out, ms = run(dev, x, w, b)
        except Exception as e:                                  # noqa: BLE001
            print(f"  {dev:5} skipped: {e}")
            continue
        diff = np.max(np.abs(out - cpu_out))
        print(f"  {dev:5} {ms:7.3f} ms/iter   max|Δ vs cpu|={diff:.3e}")


if __name__ == "__main__":
    main()
