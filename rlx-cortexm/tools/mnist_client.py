#!/usr/bin/env python3
"""Talk to a flashed rlx-cortexm-firmware dongle over USB CDC.

Sends N MNIST test images (using the same quantized blob the bulk
validation test consumes), reads back the predictions, prints
per-image latency and accuracy.

Usage:
    python3 tools/mnist_client.py --port /dev/cu.usbmodemXXXX --n 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import serial   # pip install pyserial


HERE = Path(__file__).resolve().parent
DEFAULT_BLOB = HERE.parent / "tests" / "data" / "test_set.bin"
INPUT_LEN = 28 * 28 * 1


def find_default_port() -> str | None:
    """Cross-platform USB CDC port auto-detect via pyserial.

    Prefers a port whose VID matches the rlx-cortexm-firmware
    advertisement (VID 0x16c0, PID 0x27dd — the V-USB / pid.codes test
    pair). Falls back to any USB CDC port we can see.
    """
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    if not ports:
        return None
    for p in ports:
        if (p.vid, p.pid) == (0x16c0, 0x27dd):
            return p.device
    # Filter to USB CDC-ish ports — exclude built-in Bluetooth etc.
    likely = [p for p in ports if (p.vid is not None)]
    return likely[0].device if likely else ports[0].device


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None,
                    help="Serial port (default: first /dev/cu.usbmodem*)")
    ap.add_argument("--blob", default=str(DEFAULT_BLOB),
                    help=f"Path to test_set.bin (default: {DEFAULT_BLOB})")
    ap.add_argument("--n", type=int, default=50,
                    help="Number of images to send")
    ap.add_argument("--baud", type=int, default=115200,
                    help="Baud rate (CDC ignores this but pyserial wants it)")
    args = ap.parse_args()

    port = args.port or find_default_port()
    if port is None:
        print("error: no /dev/cu.usbmodem* found and --port not given",
              file=sys.stderr)
        return 1

    blob = Path(args.blob).read_bytes()
    rec_len = INPUT_LEN + 1
    if len(blob) % rec_len != 0:
        print(f"error: blob size {len(blob)} not a multiple of {rec_len}",
              file=sys.stderr)
        return 1
    total = len(blob) // rec_len
    n = min(args.n, total)

    print(f"port: {port}")
    print(f"blob: {args.blob}  ({total} images, using {n})")

    with serial.Serial(port, args.baud, timeout=2.0) as s:
        s.reset_input_buffer()
        s.reset_output_buffer()

        correct = 0
        t_start = time.perf_counter()
        for i in range(n):
            rec = blob[i * rec_len:(i + 1) * rec_len]
            img, label = rec[:INPUT_LEN], rec[INPUT_LEN]

            t0 = time.perf_counter_ns()
            s.write(img)
            s.flush()
            resp = s.read(1)
            t1 = time.perf_counter_ns()

            if not resp:
                print(f"  [{i:3d}] timeout", file=sys.stderr)
                continue
            pred = resp[0]
            ok = pred == label
            correct += ok
            print(f"  [{i:3d}] label={label} pred={pred} "
                  f"{'OK' if ok else 'MISS'}  {(t1 - t0) / 1e6:.2f} ms")

        dt = time.perf_counter() - t_start
        print(f"\nresult: {correct}/{n} = {correct / n:.4f} "
              f"in {dt:.2f}s ({n / dt:.1f} img/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
