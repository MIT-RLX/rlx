# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#
# Higher-order autodiff: nth_order_grad, directional_nth_grad, hvp (CPU).

from __future__ import annotations

import numpy as np

import pyrlx as rlx


def test_nth_order_x_cubed_third_derivative():
    g = rlx.Graph("x3")
    x = g.input("x", [], "f32")
    x2 = g.binary("mul", x, x)
    x3 = g.binary("mul", x2, x)
    g.set_outputs([x3])

    hg = rlx.nth_order_grad(g, "x", 3)
    c = rlx.Session(device="cpu").compile(hg)
    [out] = c.run({"x": np.array(1.5, dtype=np.float32)})
    assert abs(float(out.flat[0]) - 6.0) < 1e-5


def test_directional_nth_sum_squares_hessian_vector():
    n = 4
    g = rlx.Graph("sum_sq")
    x = g.input("x", [n], "f32")
    xx = g.binary("mul", x, x)
    f = g.reduce(xx, "sum", [0], False)
    g.set_outputs([f])

    hg = rlx.directional_nth_grad(g, "x", 2)
    c = rlx.Session(device="cpu").compile(hg)
    x_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    v = np.array([0.5, -0.25, 1.0, -1.5], dtype=np.float32)
    [out] = c.run({"x": x_data, "dir_0": v, "dir_1": v})
    want = 2.0 * float(np.dot(v, v))
    assert abs(float(out.flat[0]) - want) < 1e-5
