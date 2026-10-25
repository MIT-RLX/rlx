# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Distributed nodes from Python: a two-rank mesh on threads (every blocking
call releases the GIL), collectives, and a shipped stage — plus the other rows
the runtime API table added for parity."""

import threading

import numpy as np
import pytest

import pyrlx


def _double():
    g = pyrlx.Graph("double")
    x = g.input("x", [2])
    g.set_outputs([g.add(x, x)])
    return g


def test_a_two_rank_mesh_reduces_and_serves_a_stage():
    peers = pyrlx.loopback_peers(2)
    assert len(peers) == 2
    report = {}

    def worker():
        g = pyrlx.Group.connect(1, 2, {"peers": peers})
        g.all_reduce([3.0], "max")
        g.broadcast(0, [0.0, 0.0])
        report.update(g.serve_worker({"steps": 1}))

    t = threading.Thread(target=worker)
    t.start()
    g = pyrlx.Group.connect(0, 2, {"peers": peers})
    assert (g.rank(), g.world(), g.is_leader()) == (0, 2, True)
    assert g.all_reduce([5.0], "max").tolist() == [5.0]
    assert g.broadcast(0, [7.0, 8.0]).tolist() == [7.0, 8.0]
    caps = g.collect_caps()
    assert [c["rank"] for c in caps] == [1]
    g.ship_stage(1, _double(), {"device": "cpu"})
    g.send_activation(1, [1.0, 2.0])
    assert g.recv_activation(1).tolist() == [2.0, 4.0]
    t.join(timeout=30)
    assert report["activations"] == 1
    with pytest.raises(Exception, match="median"):
        g.all_reduce([1.0], "median")


def test_connect_checks_rank_and_options():
    with pytest.raises(Exception, match="not in a world"):
        pyrlx.Group.connect(2, 1)
    with pytest.raises(Exception, match="unknown node option"):
        pyrlx.Group.connect(0, 1, {"peer": []})
    g = pyrlx.Group.connect(0, 1)
    assert g.all_gather([1.0, 2.0]).tolist() == [1.0, 2.0]


def test_runner_and_router_rows_added_for_parity(tmp_path):
    w = np.eye(2, dtype=np.float32)
    x = {"x": np.array([[3, 4]], dtype=np.float32)}

    def mk():
        g = pyrlx.Graph("lin")
        a = g.input("x", [1, 2])
        p = g.param("w", [2, 2])
        g.set_outputs([g.matmul(a, p)])
        return g

    r = pyrlx.GraphDevices(mk(), pyrlx.DevicePolicy.only(["cpu"]))
    assert r.set_params({"w": w}) == 1
    assert r.resolve() == "cpu"
    assert r.run_on("cpu", x)[0].ravel().tolist() == [3.0, 4.0]
    router = pyrlx.DeviceRouter(mk(), pyrlx.DevicePolicy.only(["cpu"]))
    assert router.set_params({"w": w}) == 1
    assert router.run_on("cpu", x)[0].ravel().tolist() == [3.0, 4.0]
    with pytest.raises(Exception, match="unknown weight format"):
        pyrlx.load_weights("nope", str(tmp_path / "missing"))
    with pytest.raises(Exception):
        pyrlx.to_rlxp("gguf", str(tmp_path / "missing.gguf"), str(tmp_path / "never.rlxp"))
