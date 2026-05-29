# RLX — pyrlx FFT parity tests.

import struct

import pytest

import pyrlx as rlx


def _compile(g: rlx.Graph):
    return rlx.Session(device="cpu").compile(g)


def test_fft_round_trip_forward_norm():
    g = rlx.Graph("fft")
    x = g.input("x", [8], "f32")
    y = g.fft_norm(x, inverse=False, norm="forward")
    z = g.fft_norm(y, inverse=True, norm="forward")
    g.set_outputs([z])
    exe = _compile(g)

    signal = [1.0, 0.5, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0]
    x_bytes = b"".join(struct.pack("<f", v) for v in signal)
    raw, _ = exe.run_typed({"x": (x_bytes, "f32")})[0]
    got = struct.unpack("<8f", raw)
    for a, b in zip(got, signal):
        assert abs(a - b) < 1e-4


def test_rfft_irfft_and_fftfreq():
    g = rlx.Graph("rfft")
    x = g.input("x", [4], "f32")
    re, im = g.rfft(x, norm="forward")
    y = g.irfft(re, im, 4, norm="forward")
    freq = g.rfftfreq(4)
    g.set_outputs([y, freq])
    exe = _compile(g)

    signal = [1.0, 2.0, 3.0, 0.5]
    x_bytes = b"".join(struct.pack("<f", v) for v in signal)
    outs = exe.run_typed({"x": (x_bytes, "f32")})
    y_raw, _ = outs[0]
    got = struct.unpack("<4f", y_raw)
    for a, b in zip(got, signal):
        assert abs(a - b) < 1e-3
    freq_raw, _ = outs[1]
    rf = struct.unpack("<3d", freq_raw)
    assert abs(rf[0]) < 1e-12
    assert abs(rf[1] - 0.25) < 1e-12
