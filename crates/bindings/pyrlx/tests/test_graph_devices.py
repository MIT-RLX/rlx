# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""GraphDevices / DevicePolicy Python bindings."""

from __future__ import annotations

import json

import numpy as np
import pytest

import pyrlx as rlx


def _identity_graph():
    g = rlx.Graph("identity")
    x = g.input("x", [4], "f32")
    g.set_outputs([x])
    return g


def _add_param_graph():
    g = rlx.Graph("add_param")
    x = g.input("x", [4], "f32")
    w = g.param("w", [4], "f32")
    y = g.add(x, w)
    g.set_outputs([y])
    return g


def test_backends_manifest_json():
    raw = rlx.backends_manifest()
    doc = json.loads(raw)
    assert "backends" in doc
    assert "cpu" in doc["backends"]


def test_parse_device_aliases():
    assert rlx.parse_device("CUDA") == "cuda"
    assert rlx.parse_device("wgpu") == "gpu"


def test_fastest_device_for_cpu():
    g = _identity_graph()
    assert rlx.fastest_device_for(g) == "cpu"


def test_device_report_cpu():
    g = _identity_graph()
    rows = rlx.device_report(g, policy=rlx.DevicePolicy.only(["cpu"]))
    assert any(r.label == "cpu" and r.recommended for r in rows)


def test_graph_devices_cpu_run():
    g = _identity_graph()
    runner = rlx.GraphDevices(g)
    assert "cpu" in runner.devices()
    out = runner.run("cpu", {"x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)})
    assert len(out) == 1
    np.testing.assert_allclose(out[0], [1.0, 2.0, 3.0, 4.0])


def test_graph_devices_run_chain_cpu():
    g = _identity_graph()
    runner = rlx.GraphDevices(g, policy=rlx.DevicePolicy.only(["cpu"]))
    device, outs = runner.run_chain(
        {"x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)}
    )
    assert device == "cpu"
    assert len(outs) == 1


def test_graph_devices_run_try_cpu():
    g = _identity_graph()
    runner = rlx.GraphDevices(g, policy=rlx.DevicePolicy.only(["cpu"]))
    device, outs = runner.run_try(
        ["cpu"],
        {"x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)},
    )
    assert device == "cpu"
    assert len(outs) == 1


def test_graph_devices_set_param_typed():
    g = _add_param_graph()
    runner = rlx.GraphDevices(g, policy=rlx.DevicePolicy.only(["cpu"]))
    w = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    runner.set_param_typed("w", w.tobytes(), "f32")
    out = runner.run("cpu", {"x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)})
    np.testing.assert_allclose(out[0], [2.0, 2.0, 3.0, 5.0])


def test_flexible_session_from_env():
    g = _identity_graph()
    session = rlx.FlexibleSession.from_env()
    compiled = session.compile_resolved(g, device="cpu")
    assert compiled.device == "cpu"


def test_flexible_session_compile_resolved_cpu():
    g = _identity_graph()
    session = rlx.FlexibleSession(rlx.DevicePolicy.only(["cpu"]))
    compiled = session.compile_resolved(g, device="cpu")
    assert compiled.device == "cpu"


def test_device_router_cpu_run():
    g = _identity_graph()
    router = rlx.DeviceRouter(g, policy=rlx.DevicePolicy.only(["cpu"]))
    assert "cpu" in router.devices()
    device, out = router.run({"x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)})
    assert device == "cpu"
    assert len(out) == 1
    np.testing.assert_allclose(out[0], [1.0, 2.0, 3.0, 4.0])


def test_device_router_set_param_typed():
    g = _add_param_graph()
    router = rlx.DeviceRouter(g, policy=rlx.DevicePolicy.only(["cpu"]))
    w = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    router.set_param_typed("w", w.tobytes(), "f32")
    device, out = router.run({"x": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)})
    assert device == "cpu"
    np.testing.assert_allclose(out[0], [2.0, 2.0, 3.0, 5.0])
