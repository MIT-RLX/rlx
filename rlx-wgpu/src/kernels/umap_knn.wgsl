// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Per-row k-NN on a pairwise distance matrix (UMAP). One thread per row;
// insertion-sort into k slots. Matches `rlx_cpu::umap_knn::knn_forward_packed`.

struct Params {
    n: u32,
    k: u32,
    pw_off: u32,
    out_off: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform>              params: Params;

const INF: f32 = 3.4028235e+38;

@compute @workgroup_size(64)
fn umap_knn(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let row = gid.x + gid.y * nwg.x * 64u;
    let n = params.n;
    let k = params.k;
    if (row >= n || k >= n) {
        return;
    }

    let pw_base = params.pw_off + row * n;
    let out_base = params.out_off + row * (2u * k);

    for (var s: u32 = 0u; s < k; s = s + 1u) {
        arena[out_base + s] = f32(n);
        arena[out_base + k + s] = INF;
    }

    var worst: f32 = INF;
    for (var col: u32 = 0u; col < n; col = col + 1u) {
        if (col == row) {
            continue;
        }
        let dist = arena[pw_base + col];
        if (dist >= worst) {
            continue;
        }
        if (dist < arena[out_base + k + k - 1u]) {
            var slot: u32 = k - 1u;
            loop {
                if (slot == 0u) {
                    break;
                }
                if (dist < arena[out_base + k + slot - 1u]) {
                    arena[out_base + k + slot] = arena[out_base + k + slot - 1u];
                    arena[out_base + slot] = arena[out_base + slot - 1u];
                    slot = slot - 1u;
                } else {
                    break;
                }
            }
            arena[out_base + k + slot] = dist;
            arena[out_base + slot] = f32(col);
            worst = arena[out_base + k + k - 1u];
        }
    }
}
