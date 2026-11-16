#!/usr/bin/env python3
# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Render the README's k-means / e-means / o-means charts as static SVG.

The numbers are the measurements quoted in the README, from
`tests/omeans.rs` (`bench_kmeans_emeans_omeans`, `bench_large_k`). Rerun
those benchmarks and update RUNS / SPEEDUP to refresh the charts:

    python3 scripts/readme_charts.py      # writes docs/*.svg

No dependencies. Each SVG follows the viewer's light/dark preference.
"""

import math
import os

# Host runs, n = 100k, d = 32, same random rows: (seconds, training MSE).
RUNS = {
    1024: {
        "Lloyd-25": [(5.70, 7.7478, "")],
        "e-means": [(0.78, 4.1831, "1"), (1.21, 3.9736, "2"), (3.56, 5.4566, "5")],
        "o-means": [(0.52, 4.0131, "1"), (2.29, 3.9736, "2"), (4.19, 5.4566, "5")],
    },
    4096: {
        "Lloyd-25": [(30.27, 6.8297, "")],
        "e-means": [(1.76, 4.2086, "1"), (2.89, 3.8965, "2"), (6.55, 3.8652, "5")],
        "o-means": [(1.40, 3.9748, "1"), (2.81, 3.8996, "2"), (5.89, 3.8632, "5")],
    },
}

# Device o-means over dense device e-means, n = 200k, d = 96, k = 16384, 1 epoch.
SPEEDUP = [
    ("CUDA (RTX 3080 Ti)", 2.29),
    ("CPU (msi)", 2.20),
    ("ROCm (780M)", 2.03),
    ("Vulkan (RTX 3080 Ti)", 1.31),
    ("wgpu (RTX 3080 Ti)", 1.04),
    ("Metal (Apple M-series)", 0.87),
]

# Reference palette: categorical slots 1-3 (validated all-pairs, both modes),
# chart chrome and ink. Light first, dark under prefers-color-scheme.
STYLE = """
<style>
  .surface { fill: #fcfcfb; }
  .ink { fill: #0b0b0b; }
  .ink2 { fill: #52514e; }
  .muted { fill: #898781; }
  .grid { stroke: #e1e0d9; stroke-width: 1; }
  .axis { stroke: #c3c2b7; stroke-width: 1; }
  .ref { stroke: #52514e; stroke-width: 1.5; stroke-dasharray: 4 3; }
  .s1 { fill: #2a78d6; }
  .s2 { fill: #eb6834; }
  .s3 { fill: #1baf7a; }
  .ring { stroke: #fcfcfb; stroke-width: 2; }
  text { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .ink { fill: #f0efec; }
    .ink2 { fill: #c3c2b7; }
    .grid { stroke: #2c2c2a; }
    .axis { stroke: #383835; }
    .ref { stroke: #c3c2b7; }
    .s1 { fill: #3987e5; }
    .s2 { fill: #d95926; }
    .s3 { fill: #199e70; }
    .ring { stroke: #1a1a19; }
  }
</style>
"""

SERIES = [("Lloyd-25", "s1", "circle"), ("e-means", "s2", "square"), ("o-means", "s3", "triangle")]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;")


def marker(shape, cls, x, y, r=5.5):
    if shape == "circle":
        return f'<circle class="{cls} ring" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'
    if shape == "square":
        return (f'<rect class="{cls} ring" x="{x - r:.1f}" y="{y - r:.1f}" '
                f'width="{2 * r}" height="{2 * r}" rx="1.5"/>')
    h = r * 1.15
    pts = f"{x:.1f},{y - h:.1f} {x + h:.1f},{y + h * 0.8:.1f} {x - h:.1f},{y + h * 0.8:.1f}"
    return f'<polygon class="{cls} ring" points="{pts}"/>'


def error_vs_time(path):
    w, h = 800, 390
    pw, ph = 300, 220
    left0, top = 70, 110
    gap = 90
    xmin, xmax = 0.3, 50.0  # seconds, log scale
    ymax = 9.0
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
           f'width="{w}" height="{h}" role="img" aria-labelledby="t d">', STYLE,
           '<title id="t">Training MSE against wall time</title>',
           '<desc id="d">Host runs on 100,000 rows of dimension 32 from the same random '
           'rows. At k = 1024, Lloyd-25 reaches MSE 7.75 in 5.7 s; e-means and o-means '
           'reach 3.97 to 5.46 in 0.5 to 4.2 s. At k = 4096, Lloyd-25 reaches 6.83 in '
           '30.3 s; e-means and o-means reach 3.86 to 4.21 in 1.4 to 6.6 s.</desc>',
           f'<rect class="surface" width="{w}" height="{h}" rx="8"/>',
           '<text class="ink" x="20" y="30" font-size="16" font-weight="600">'
           'Training MSE against wall time (lower left is better)</text>',
           '<text class="ink2" x="20" y="50" font-size="12">Host, n = 100k, d = 32, '
           'same random starting rows; labels give the epochs</text>']
    # Legend: one row, marker + name in text ink.
    lx = 20
    for name, cls, shape in SERIES:
        out.append(marker(shape, cls, lx + 6, 66, 5))
        out.append(f'<text class="ink2" x="{lx + 16}" y="70" font-size="12">{name}</text>')
        lx += 100

    def sx(t, x0):
        return x0 + pw * (math.log10(t) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin))

    def sy(v):
        return top + ph * (1 - v / ymax)

    for i, k in enumerate(RUNS):
        x0 = left0 + i * (pw + gap)
        out.append(f'<text class="ink" x="{x0}" y="{top - 10}" font-size="13" '
                   f'font-weight="600">k = {k}</text>')
        for v in range(0, 10, 3):
            y = sy(v)
            out.append(f'<line class="grid" x1="{x0}" x2="{x0 + pw}" y1="{y:.1f}" y2="{y:.1f}"/>')
            out.append(f'<text class="muted" x="{x0 - 8}" y="{y + 4:.1f}" font-size="11" '
                       f'text-anchor="end">{v}</text>')
        for t in (0.5, 1, 2, 5, 10, 20, 50):
            x = sx(t, x0)
            out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + ph}"/>')
            out.append(f'<text class="muted" x="{x:.1f}" y="{top + ph + 16}" font-size="11" '
                       f'text-anchor="middle">{t:g}</text>')
        out.append(f'<line class="axis" x1="{x0}" x2="{x0 + pw}" y1="{top + ph}" y2="{top + ph}"/>')
        out.append(f'<text class="muted" x="{x0 + pw / 2}" y="{top + ph + 34}" font-size="11" '
                   f'text-anchor="middle">seconds (log scale)</text>')
        if i == 0:
            out.append(f'<text class="muted" x="{x0 - 44}" y="{top + ph / 2}" font-size="11" '
                       f'text-anchor="middle" transform="rotate(-90 {x0 - 44} {top + ph / 2})">'
                       'training MSE</text>')
        for name, cls, shape in SERIES:
            for t, v, epochs in RUNS[k][name]:
                x, y = sx(t, x0), sy(v)
                out.append(f'<g><title>{esc(name)}{" T=" + epochs if epochs else ""}: '
                           f'MSE {v:.4f} in {t:.2f} s</title>{marker(shape, cls, x, y)}</g>')
                label = f"T={epochs}" if epochs else name
                # e-means labels above, o-means below, so a shared point stays legible.
                dy = -10 if name == "e-means" else 18
                if name == "Lloyd-25":
                    dy = -10
                out.append(f'<text class="ink2" x="{x:.1f}" y="{y + dy:.1f}" font-size="10" '
                           f'text-anchor="middle">{label}</text>')
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def speedup(path):
    w = 760
    bar, step = 20, 34
    left, top = 190, 78
    pw = 470
    xmax = 2.5
    h = top + step * len(SPEEDUP) + 46
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
           f'width="{w}" height="{h}" role="img" aria-labelledby="t d">', STYLE,
           '<title id="t">Device o-means speed-up over dense device e-means</title>',
           '<desc id="d">n = 200,000, d = 96, k = 16384, one epoch. '
           + ", ".join(f"{n} {v:.2f}x" for n, v in SPEEDUP) + ".</desc>",
           f'<rect class="surface" width="{w}" height="{h}" rx="8"/>',
           '<text class="ink" x="20" y="30" font-size="16" font-weight="600">'
           'o-means speed-up over dense e-means, same device</text>',
           '<text class="ink2" x="20" y="50" font-size="12">n = 200k, d = 96, k = 16,384, '
           'one epoch; right of the dashed line is faster</text>']

    def sx(v):
        return left + pw * v / xmax

    for v in (0, 0.5, 1, 1.5, 2, 2.5):
        x = sx(v)
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 8}" '
                   f'y2="{top + step * len(SPEEDUP) - 8}"/>')
        out.append(f'<text class="muted" x="{x:.1f}" y="{top + step * len(SPEEDUP) + 8}" '
                   f'font-size="11" text-anchor="middle">{v:g}×</text>')
    for i, (name, v) in enumerate(SPEEDUP):
        y = top + i * step
        x1 = sx(v)
        r = 4
        # Square at the baseline, 4px rounded data end.
        d = (f"M{left},{y} H{x1 - r:.1f} Q{x1:.1f},{y} {x1:.1f},{y + r} "
             f"V{y + bar - r} Q{x1:.1f},{y + bar} {x1 - r:.1f},{y + bar} H{left} Z")
        out.append(f'<g><title>{esc(name)}: {v:.2f}× </title>'
                   f'<path class="s1" d="{d}"/></g>')
        out.append(f'<text class="ink2" x="{left - 10}" y="{y + 14}" font-size="12" '
                   f'text-anchor="end">{esc(name)}</text>')
        # Past the reference line, so a bar ending near 1× keeps its label clear.
        lx = max(x1, sx(1.0) + 2) + 6 if abs(x1 - sx(1.0)) < 40 else x1 + 6
        out.append(f'<text class="ink" x="{lx:.1f}" y="{y + 14}" font-size="12">{v:.2f}×</text>')
    out.append(f'<line class="axis" x1="{left}" x2="{left}" y1="{top - 8}" '
               f'y2="{top + step * len(SPEEDUP) - 8}"/>')
    x = sx(1.0)
    out.append(f'<line class="ref" x1="{x:.1f}" x2="{x:.1f}" y1="{top - 14}" '
               f'y2="{top + step * len(SPEEDUP) - 4}"/>')
    out.append(f'<text class="ink2" x="{x + 4:.1f}" y="{top - 16}" font-size="11">'
               'same speed</text>')
    out.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    docs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
    os.makedirs(docs, exist_ok=True)
    error_vs_time(os.path.join(docs, "kmeans_emeans_omeans_error_vs_time.svg"))
    speedup(os.path.join(docs, "omeans_device_speedup.svg"))
