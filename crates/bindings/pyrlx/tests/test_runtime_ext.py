# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Runtime-level surface: static ``check``, precision policies, compiled-graph
helpers, ``Optimizer`` / ``Trainer`` (feature ``training``), and the text layer
(feature ``text``)."""

import os
from pathlib import Path

import numpy as np
import pytest

import pyrlx as rlx


def mse_graph(name="mse"):
    g = rlx.Graph(name)
    x = g.input("x", [1, 2], "f32")
    t = g.input("t", [1, 2], "f32")
    w = g.param("w", [2, 2], "f32")
    d = g.sub(g.matmul(x, w), t)
    g.set_outputs([g.mean(g.mul(d, d), [0, 1], False)])
    return g


# ── check ───────────────────────────────────────────────────────


def test_check_reports_every_requested_backend():
    g = mse_graph()
    report = rlx.check(g, backends=["cpu", "cuda", "metal"])
    assert report["graph"] == "mse"
    assert report["nodes"] == len(g)
    assert [b["backend"] for b in report["backends"]] == ["cpu", "cuda", "metal"]
    cpu = report["backends"][0]
    assert cpu["legality"] is not None and cpu["legality"]["compile_ready"]
    assert report["errors"] == 0
    # The graph is borrowed, not consumed.
    rlx.Session("cpu").compile(g)


def test_check_rejects_an_unknown_backend_by_name():
    with pytest.raises(ValueError, match="unknown backend 'nope'"):
        rlx.check(mse_graph(), backends=["nope"])
    assert "cpu" in rlx.check_backends()


# ── policy + compiled helpers ───────────────────────────────────


def test_policies_parse_and_report_themselves():
    assert rlx.Session("cpu", policy="mixed").policy == "mixed"
    assert rlx.Session("cpu", policy={"compute": "f16"}).policy == "custom"
    assert rlx.Session("cpu").policy is None
    with pytest.raises(ValueError, match="unknown precision policy"):
        rlx.Session("cpu", policy="bogus")
    with pytest.raises(ValueError, match="unknown op kind"):
        rlx.Session("cpu", policy={"computer": "f16"})


def test_a_policy_compile_stays_close_to_f32():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 32)).astype(np.float32)
    w = rng.standard_normal((32, 32)).astype(np.float32)

    def build():
        g = rlx.Graph("mm")
        g.set_outputs([g.matmul(g.input("a", [32, 32], "f32"), g.param("w", [32, 32], "f32"))])
        return g

    outs = []
    for session, kw in [(rlx.Session("cpu"), {}), (rlx.Session("cpu"), {"policy": "mixed"})]:
        c = session.compile_with(build(), **kw)
        c.set_params({"w": w})
        outs.append(c.run({"a": a})[0])
    np.testing.assert_allclose(outs[1], outs[0], rtol=5e-3, atol=5e-3)


def test_compiled_graph_reports_output_shapes_and_dtypes():
    g = rlx.Graph("io")
    x = g.input("x", [2, 3], "f32")
    g.set_outputs([g.relu(x), g.argmax(x, 1)])
    c = rlx.Session("cpu").compile(g)
    assert c.output_shapes() == [[2, 3], [2]]
    assert c.output_dtypes()[0] == "f32"


def test_fusion_options_carry_the_conv_bias_act_toggle():
    fo = rlx.FusionOptions(disable_conv_bias_act_fusion=True)
    assert fo.disable_conv_bias_act_fusion
    assert not rlx.FusionOptions.native_fk().disable_conv_bias_act_fusion


# ── training ────────────────────────────────────────────────────

training = pytest.mark.skipif(not hasattr(rlx, "Trainer"), reason="built without `training`")


@training
def test_optimizer_moves_weights_downhill():
    opt = rlx.Optimizer("adam", lr=0.1)
    p = np.array([1.0, -1.0], np.float32)
    for _ in range(50):
        p = opt.step({"p": (p, 2 * p)})["p"]  # grad of p²
    assert np.all(np.abs(p) < 0.2), p


@training
def test_every_optimizer_kind_constructs_and_steps():
    kinds = ["sgd", "adam", "adamw", "lion", "muon", "radam", "nadamw", "lamb", "mars",
             "soap", "sophia", "qhadamw", "adafactor"]
    for kind in kinds:
        out = rlx.Optimizer(kind, lr=1e-2).step(
            {"w": (np.ones((4, 4), np.float32), np.ones((4, 4), np.float32))}
        )
        assert out["w"].shape == (4, 4), kind
    with pytest.raises(ValueError, match="unknown optimizer"):
        rlx.Optimizer("adamx")
    with pytest.raises(ValueError, match="unknown optimizer option"):
        rlx.Optimizer("adam", learning_rate=0.1)


@training
def test_trainer_learns_a_linear_map():
    t = rlx.Trainer(
        mse_graph(), wrt=["w"], init={"w": np.zeros(4, np.float32)},
        optimizer={"kind": "adamw", "lr": 0.1},
    )
    batch = {"x": np.array([1, 1], np.float32), "t": np.array([1, -1], np.float32)}
    first = t.step(batch)
    for _ in range(200):
        t.step(batch)
    assert first > 0.9
    assert t.evaluate(batch) < 1e-4
    assert t.steps() == 201
    assert t.params()["w"].shape == (2, 2)
    assert t.grad_norm() >= 0.0 and not t.is_resident()


@training
def test_trainer_run_schedules_and_stops_early():
    t = rlx.Trainer(mse_graph(), wrt=["w"], init={"w": np.zeros(4, np.float32)},
                    optimizer="sgd")
    batch = {"x": np.array([1, 1], np.float32), "t": np.array([1, -1], np.float32)}
    seen = []
    summary = t.run(
        10, lambda i: batch, lr=lambda i: 0.05,
        on_step=lambda i, loss: (seen.append(i), i < 3)[1],
    )
    assert summary["steps"] == 4 and summary["stopped"]
    assert seen == [0, 1, 2, 3]
    # Resumes where it left off.
    assert t.run(2, lambda i: batch)["steps"] == 2 and t.steps() == 6


@training
def test_trainer_freezes_and_checkpoints(tmp_path):
    def build():
        g = mse_graph()
        return g

    batch = {"x": np.array([1, 1], np.float32), "t": np.array([1, -1], np.float32)}
    a = rlx.Trainer(build(), wrt=["w"], init={"w": np.zeros(4, np.float32)},
                    optimizer={"kind": "adamw", "lr": 0.05})
    a.run(20, lambda i: batch)
    ck = str(tmp_path / "ck.gguf")
    assert a.save(ck) >= 1
    continued = a.run(20, lambda i: batch)["last_loss"]

    b = rlx.Trainer(build(), wrt=["w"], init={"w": np.zeros(4, np.float32)},
                    optimizer={"kind": "adamw", "lr": 0.05})
    rep = b.load(ck)
    assert rep["step"] == 20 and rep["optimizer_restored"]
    assert abs(b.run(20, lambda i: batch)["last_loss"] - continued) < 1e-6

    assert b.freeze() == 1 and b.frozen() == ["w"]
    before = b.params()["w"].copy()
    b.step(batch)
    np.testing.assert_array_equal(b.params()["w"], before)
    assert b.unfreeze(["w"]) == 0

    with pytest.raises(ValueError, match="written by 'adamw'"):
        rlx.Trainer(build(), wrt=["w"], init={"w": np.zeros(4, np.float32)},
                    optimizer="lion").load(ck)


@training
def test_trainer_names_its_params_when_wrt_is_wrong():
    with pytest.raises(ValueError, match=r"not a Param in this graph — it has \[w\]"):
        rlx.Trainer(mse_graph(), wrt=["v"], init={"w": np.zeros(4, np.float32)})


@training
def test_trainer_refuses_resident_on_cpu():
    with pytest.raises(ValueError, match="no device buffers"):
        rlx.Trainer(mse_graph(), wrt=["w"], init={"w": np.zeros(4, np.float32)},
                    resident=True)


# ── text ────────────────────────────────────────────────────────

text = pytest.mark.skipif(not hasattr(rlx, "Sampler"), reason="built without `text`")


@text
def test_argmax_and_one_shot_sampling_are_deterministic():
    logits = [0.1, 3.0, 0.2, 2.9]
    assert rlx.argmax(logits) == 1
    assert rlx.sample_next(logits) == 1  # no options = greedy
    a = rlx.sample_next(logits, [], temperature=1.0, seed=7)
    assert a == rlx.sample_next(logits, [], temperature=1.0, seed=7)
    with pytest.raises(ValueError, match="unknown sampling option"):
        rlx.sample_next(logits, [], temprature=1.0)


@text
def test_sampler_carries_history_and_rng():
    s = rlx.Sampler(temperature=1.0, seed=3)
    s.prime([5, 6])
    draws = {s.next([1.0, 1.0, 1.0, 1.0]) for _ in range(64)}
    assert len(draws) > 1, "a carried RNG should not repeat one token forever"
    assert s.history()[:2] == [5, 6] and len(s.history()) == 66
    s.reset(seed=3)
    assert s.history() == []


@text
def test_render_chat_from_an_inline_template():
    tpl = "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}" \
          "{% if add_generation_prompt %}<assistant>{% endif %}"
    out = rlx.render_chat(tpl, [{"role": "user", "content": "hi"}])
    assert out == "<user>hi<assistant>"
    with pytest.raises(ValueError, match="role 'bot'"):
        rlx.render_chat(tpl, [{"role": "bot", "content": "x"}])


def _tokenizer_fixture():
    rel = Path("weights/qwen3-0.6b/tokenizer.json")
    here = Path(__file__).resolve().parents[5] / "rlx-models" / rel
    candidates = [here]
    if os.environ.get("RLX_MODELS_DIR"):
        candidates.append(Path(os.environ["RLX_MODELS_DIR"]) / rel)
    return next((p for p in candidates if p.is_file()), None)


@text
def test_streaming_decode_matches_a_full_decode():
    path = _tokenizer_fixture()
    if path is None:
        pytest.skip("qwen3-0.6b/tokenizer.json not present")
    tok = rlx.load_tokenizer(str(path))
    text_in = "Hello, 世界 — 👨‍👩‍👧 done."
    ids = tok.encode(text_in, add_special=False)
    streamed = "".join(tok.decode_incremental(i) for i in ids) + tok.finish_stream()
    assert streamed == tok.decode(ids)
