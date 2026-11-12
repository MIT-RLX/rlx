#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare rlx-vq e-means against faiss's NumpyEmeans (arXiv:2611.09501).

Needs numpy + faiss-cpu and a faiss checkout that contains contrib/e_means:

    FAISS_EMEANS_DIR=<faiss>/contrib/e_means python3 emeans_faiss_reference.py \
        [--n 50000 --d 32 --k 256 --nepochs 5 --seed 1234 --work <dir>]

Generates blob data, runs the faiss reference and the rlx `emeans_train`
example with the same seed, and reports per-epoch objectives, final MSE,
centroid differences and label agreement. Both use NumPy's MT19937 stream for
initialization, shuffles and revival, so they visit the same rows in the same
order; remaining differences come from f32 summation order in assignment.
"""

import argparse
import importlib
import os
import subprocess
import sys
import tempfile

import numpy as np


def load_faiss_emeans():
    path = os.environ.get("FAISS_EMEANS_DIR")
    if not path:
        sys.exit("set FAISS_EMEANS_DIR to <faiss>/contrib/e_means")
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    return importlib.import_module(os.path.basename(os.path.abspath(path)))


def blobs(n, d, nblobs, seed):
    rng = np.random.RandomState(seed)
    centres = rng.uniform(-10, 10, (nblobs, d)).astype("float32")
    which = (rng.random_sample(n) ** 2 * nblobs).astype(int) % nblobs
    spread = (0.3 + (np.arange(nblobs) % 5) * 0.2).astype("float32")
    noise = rng.standard_normal((n, d)).astype("float32") * spread[which, None]
    return np.ascontiguousarray(centres[which] + noise, dtype="float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--nepochs", type=float, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--work", default=None)
    args = ap.parse_args()

    e_means = load_faiss_emeans()
    import faiss

    work = args.work or tempfile.mkdtemp(prefix="emeans_ref_")
    x = blobs(args.n, args.d, max(2, args.k * 2), 0)
    data = os.path.join(work, "x.f32")
    x.tofile(data)

    em = e_means.NumpyEmeans(args.d, args.k, nepochs=args.nepochs, seed=args.seed)
    ref_c, ref_labels = em.train(x)
    print("faiss: batch=%d alpha0=%.6f steps=%d law=%s" % (
        em.recipe["batch_size"], em.recipe["alpha0"], em.recipe["total_steps"],
        em.recipe["batch_law"]))
    for i, s in enumerate(em.iteration_stats):
        print("faiss epoch %d: obj=%.6e imbalance=%.4f nsplit=%d" % (
            i, s["obj"], s["imbalance_factor"], s["nsplit"]))
    ref_mse = faiss.knn(x, ref_c, 1)[0].mean()
    print("faiss final mse=%.6e splits=%d" % (ref_mse, em.n_splits))

    out = os.path.join(work, "rlx_c.f32")
    crate = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(
        ["cargo", "run", "--release", "-q", "-p", "rlx-vq", "--features", "emeans",
         "--example", "emeans_train", "--", data, str(args.d), str(args.k),
         str(args.nepochs), str(args.seed), out],
        check=True, cwd=crate)
    rlx_c = np.fromfile(out, dtype="float32").reshape(args.k, args.d)
    rlx_labels = np.fromfile(out + ".labels.u32", dtype="uint32")
    rlx_mse = faiss.knn(x, rlx_c, 1)[0].mean()

    print("rlx final mse (faiss-evaluated)=%.6e  rel diff=%.2e" % (
        rlx_mse, abs(rlx_mse - ref_mse) / ref_mse))
    print("centroid max |diff|=%.3e  last-epoch label agreement=%.5f" % (
        np.abs(rlx_c - ref_c).max(), (rlx_labels == ref_labels).mean()))


if __name__ == "__main__":
    main()
