# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Models: any provider's model, opened by path or id and run by name.

    RLX_MODEL_PLUGINS=…/librlx_model_plugin.dylib python models.py

A graph saved as an ``.rlxp`` package runs anywhere; a plugin's models (a
downstream collection's LLMs, or the reference plugin's ``demo:count``) load
through ``RLX_MODEL_PLUGINS`` or ``pyrlx.load_model_plugin``. The Swift, JS and
C examples of the same name do the same steps.
"""

import os
import tempfile

import numpy as np

import pyrlx


def main():
    g = pyrlx.Graph("affine")
    x = g.input("x", [1, 2])
    g.set_outputs([g.matmul(x, g.param("w", [2, 2]))])
    path = os.path.join(tempfile.gettempdir(), "rlx-example-python.rlxp")
    pyrlx.save_model(g, path, {"w": np.array([2, 0, 0, 3], np.float32)},
                     {"task": "graph", "family": "affine", "outputs": ["y"]})

    m = pyrlx.Model.open(path, device="cpu")
    info = m.info()
    print(f"model: {info['family']} ({info['task']}) on {info['device']}")
    (y,) = m.run({"x": np.ones((1, 2), np.float32)})
    print(f"model run: {y[0, 0]:.3f} {y[0, 1]:.3f}")
    os.remove(path)

    lm = pyrlx.Model.open("demo:count")
    lm_info = lm.info()
    print(f"plugin model: {lm_info['family']} ({lm_info['task']})")
    print("model text:", lm.generate_text("c", max_tokens=4))
    print("model ids:", " ".join(map(str, lm.generate([2], max_tokens=3))))

    # By task: a chat, speech out and back in, embeddings.
    print("chat:", lm.chat([{"role": "user", "content": "c"}], max_tokens=4))
    samples, rate = pyrlx.Model.open("demo:voice").synthesize("hello")
    print(f"voice: {len(samples)} samples at {rate} Hz")
    print("heard:", pyrlx.Model.open("demo:speech").transcribe(samples, rate))
    rows = pyrlx.Model.open("demo:embed").embed(["hello", "sky"])
    print("embed:", " ".join(str(int(v)) for v in rows.flatten()))


if __name__ == "__main__":
    main()
