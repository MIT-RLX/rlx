#![cfg(test)]
#![allow(unsafe_op_in_unsafe_fn)]

use super::*;
use rlx_ir::*;

/// The im2col+BLAS forward conv must match the reference scalar loop
/// (up to float reassociation) across a range of shapes — strided,
/// dilated, padded, grouped, and multi-batch. Guards the default
/// `RLX_FAST_CONV` fast path (disable with `=0` for the scalar oracle).
#[test]
fn conv2d_im2col_matches_naive() {
    // (n, c_in, h, w, c_out, kh, kw, sh, sw, ph, pw, dh, dw, groups)
    let cases = [
        (1, 1, 28, 28, 8, 3, 3, 1, 1, 0, 0, 1, 1, 1), // TinyConv layer 1
        (4, 8, 13, 13, 16, 3, 3, 1, 1, 0, 0, 1, 1, 1), // TinyConv layer 2, batched
        (2, 3, 16, 16, 6, 3, 3, 2, 2, 1, 1, 1, 1, 1), // stride 2, pad 1
        (1, 4, 12, 12, 4, 3, 3, 1, 1, 2, 2, 2, 2, 1), // dilation 2
        (3, 8, 10, 10, 8, 3, 3, 1, 1, 1, 1, 1, 1, 2), // groups = 2
        (1, 2, 7, 7, 5, 1, 1, 1, 1, 0, 0, 1, 1, 1),   // 1x1
    ];
    for (idx, &(n, c_in, h, w, c_out, kh, kw, sh, sw, ph, pw, dh, dw, groups)) in
        cases.iter().enumerate()
    {
        let c_in_per_g = c_in / groups;
        let h_out = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
        let w_out = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
        // Deterministic pseudo-random inputs (no rng dep in tests).
        let mut s: u32 = 0x9e37_79b9 ^ (idx as u32 + 1);
        let mut rand = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) - 0.5
        };
        let inp: Vec<f32> = (0..n * c_in * h * w).map(|_| rand()).collect();
        let wt: Vec<f32> = (0..c_out * c_in_per_g * kh * kw).map(|_| rand()).collect();
        let mut out_ref = vec![0f32; n * c_out * h_out * w_out];
        let mut out_fast = vec![0f32; n * c_out * h_out * w_out];

        conv2d_forward_naive(
            &inp,
            &wt,
            &mut out_ref,
            n,
            c_in,
            h,
            w,
            c_out,
            h_out,
            w_out,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            groups,
        );
        conv2d_forward_im2col(
            &inp,
            &wt,
            &mut out_fast,
            n,
            c_in,
            h,
            w,
            c_out,
            h_out,
            w_out,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            dh,
            dw,
            groups,
        );

        let max_abs = out_ref
            .iter()
            .zip(&out_fast)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs < 1e-3,
            "case {idx}: im2col vs naive max abs diff {max_abs}"
        );
    }
}

/// Direct forward conv (stride-1, no-pad) must match the reference scalar
/// conv exactly-ish across channel/batch/group shapes.
#[test]
fn conv2d_direct_matches_naive() {
    // (n, c_in, h, w, c_out, kh, kw, groups)
    let cases = [
        (1, 1, 28, 28, 8, 3, 3, 1),  // TinyConv L1
        (4, 8, 13, 13, 16, 3, 3, 1), // TinyConv L2, batched
        (2, 6, 10, 10, 9, 3, 3, 3),  // groups=3
        (1, 4, 9, 9, 4, 5, 5, 1),    // 5×5 kernel
        (3, 2, 7, 7, 2, 1, 1, 1),    // 1×1
    ];
    for (idx, &(n, c_in, h, w, c_out, kh, kw, groups)) in cases.iter().enumerate() {
        let h_out = h - kh + 1;
        let w_out = w - kw + 1;
        let c_in_per_g = c_in / groups;
        let mut s: u32 = 0xfeed_1234 ^ (idx as u32 + 1);
        let mut rand = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) - 0.5
        };
        let inp: Vec<f32> = (0..n * c_in * h * w).map(|_| rand()).collect();
        let wt: Vec<f32> = (0..c_out * c_in_per_g * kh * kw).map(|_| rand()).collect();
        let mut r = vec![0f32; n * c_out * h_out * w_out];
        let mut d = vec![0f32; n * c_out * h_out * w_out];
        conv2d_forward_naive(
            &inp, &wt, &mut r, n, c_in, h, w, c_out, h_out, w_out, kh, kw, 1, 1, 0, 0, 1, 1, groups,
        );
        conv2d_forward_direct(
            &inp, &wt, &mut d, n, c_in, h, w, c_out, h_out, w_out, kh, kw, groups,
        );
        let mx = r
            .iter()
            .zip(&d)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(mx < 1e-4, "case {idx}: direct vs naive max abs diff {mx}");
    }
}

/// Winograd F(2,3) forward conv must match the reference scalar conv (up to
/// the transform's float reassociation) for 3×3 stride-1 valid convs,
/// including odd output dims (boundary tiles) and multiple channels/batches.
#[test]
fn conv2d_winograd_matches_naive() {
    // (n, c_in, h, w, c_out)  — all 3×3 stride1 valid; covers even (26) and
    // odd (11) output dims and the TinyConv channel shapes.
    let cases = [
        (1, 1, 28, 28, 8),  // TinyConv layer 1: out 26×26 (even)
        (4, 8, 13, 13, 16), // TinyConv layer 2: out 11×11 (odd) — boundary tiles
        (2, 3, 9, 9, 5),    // out 7×7 (odd)
        (1, 4, 8, 8, 4),    // out 6×6 (even)
    ];
    for (idx, &(n, c_in, h, w, c_out)) in cases.iter().enumerate() {
        let h_out = h - 2;
        let w_out = w - 2;
        let mut s: u32 = 0x1234_5678 ^ (idx as u32 + 1);
        let mut rand = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) - 0.5
        };
        let inp: Vec<f32> = (0..n * c_in * h * w).map(|_| rand()).collect();
        let wt: Vec<f32> = (0..c_out * c_in * 9).map(|_| rand()).collect();
        let mut out_ref = vec![0f32; n * c_out * h_out * w_out];
        let mut out_win = vec![0f32; n * c_out * h_out * w_out];
        conv2d_forward_naive(
            &inp,
            &wt,
            &mut out_ref,
            n,
            c_in,
            h,
            w,
            c_out,
            h_out,
            w_out,
            3,
            3,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
        );
        conv2d_forward_winograd(&inp, &wt, &mut out_win, n, c_in, h, w, c_out, h_out, w_out);
        let max_abs = out_ref
            .iter()
            .zip(&out_win)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs < 1e-3,
            "case {idx}: winograd vs naive max abs diff {max_abs}"
        );
    }
}

/// Plan #45: when a Narrow's only consumer is a Rope, the thunk
/// fusion pass collapses them — the Narrow becomes Nop, and the
/// Rope reads from the parent buffer with its row stride. This
/// test runs the unfused path (batch*seq > FusedAttnBlock
/// threshold) and asserts the rewrite happened.
#[test]
fn narrow_rope_fuses_in_unfused_path() {
    let f = DType::F32;
    let mut g = Graph::new("nr_fuse");
    // Force batch*seq > 64 so FusedAttnBlock doesn't pre-empt us.
    let qkv = g.input("qkv", Shape::new(&[16, 8, 192], f)); // 16*8=128 > 64
    let cos = g.input("cos", Shape::new(&[16], f));
    let sin = g.input("sin", Shape::new(&[16], f));
    // Last-axis narrow: Q = qkv[..., 0..64]
    let q = g.narrow_(qkv, 2, 0, 64);
    let q_rope = g.rope(q, cos, sin, 16);
    g.set_outputs(vec![q_rope]);

    let plan = rlx_opt::memory::plan_memory(&g);
    let arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);

    let mut narrow_count = 0;
    let mut rope_with_stride: Option<u32> = None;
    for t in &sched.thunks {
        match t {
            Thunk::Narrow { .. } => narrow_count += 1,
            Thunk::Rope { src_row_stride, .. } => rope_with_stride = Some(*src_row_stride),
            _ => {}
        }
    }
    // After fusion the Narrow is gone; only the Rope remains, and
    // it now walks with the parent QKV's row stride (3 * 64 = 192).
    assert_eq!(
        narrow_count, 0,
        "Narrow→Rope fusion should leave zero Narrow thunks; saw {narrow_count}"
    );
    assert_eq!(
        rope_with_stride,
        Some(192),
        "Rope's src_row_stride should be 192 (parent qkv axis), saw {rope_with_stride:?}"
    );
}

/// Plan #15: SSM selective scan matches a naive Python-style
/// Python-style sequential reference.
#[test]
fn ssm_selective_scan_matches_reference() {
    use rlx_ir::Philox4x32;
    let bch = 1usize;
    let s = 4usize;
    let h = 3usize;
    let n = 2usize;

    let mut rng = Philox4x32::new(13);
    let mut x = vec![0f32; bch * s * h];
    rng.fill_normal(&mut x);
    let mut delta = vec![0f32; bch * s * h];
    // Keep Δ small so exp(Δ·A) doesn't blow up.
    for v in delta.iter_mut() {
        *v = (rng.next_f32() - 0.5) * 0.1;
    }
    let mut a = vec![0f32; h * n];
    for v in a.iter_mut() {
        *v = -(rng.next_f32() * 0.5 + 0.1);
    } // negative for stability
    let mut b = vec![0f32; bch * s * n];
    rng.fill_normal(&mut b);
    let mut c = vec![0f32; bch * s * n];
    rng.fill_normal(&mut c);

    // Reference scan.
    let mut expected = vec![0f32; bch * s * h];
    for bi in 0..bch {
        let mut state = vec![0f32; h * n];
        for si in 0..s {
            for ci in 0..h {
                let d = delta[bi * s * h + si * h + ci];
                let xv = x[bi * s * h + si * h + ci];
                let mut acc = 0f32;
                for ni in 0..n {
                    let da = (d * a[ci * n + ni]).exp();
                    state[ci * n + ni] =
                        da * state[ci * n + ni] + d * b[bi * s * n + si * n + ni] * xv;
                    acc += c[bi * s * n + si * n + ni] * state[ci * n + ni];
                }
                expected[bi * s * h + si * h + ci] = acc;
            }
        }
    }

    // RLX path.
    let f = DType::F32;
    let mut g = Graph::new("ssm");
    let xn = g.input("x", Shape::new(&[bch, s, h], f));
    let dn = g.input("delta", Shape::new(&[bch, s, h], f));
    let an = g.param("a", Shape::new(&[h, n], f));
    let bn = g.param("b", Shape::new(&[bch, s, n], f));
    let cn = g.param("c", Shape::new(&[bch, s, n], f));
    let yn = g.selective_scan(xn, dn, an, bn, cn, n, Shape::new(&[bch, s, h], f));
    g.set_outputs(vec![yn]);

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);

    let xn_off = arena.byte_offset(xn);
    let dn_off = arena.byte_offset(dn);
    let an_off = arena.byte_offset(an);
    let bn_off = arena.byte_offset(bn);
    let cn_off = arena.byte_offset(cn);
    let yn_off = arena.byte_offset(yn);
    let buf = arena.raw_buf_mut();
    unsafe {
        let copy = |dst: *mut f32, data: &[f32]| {
            for (i, &v) in data.iter().enumerate() {
                *dst.add(i) = v;
            }
        };
        copy(buf.as_mut_ptr().add(xn_off) as *mut f32, &x);
        copy(buf.as_mut_ptr().add(dn_off) as *mut f32, &delta);
        copy(buf.as_mut_ptr().add(an_off) as *mut f32, &a);
        copy(buf.as_mut_ptr().add(bn_off) as *mut f32, &b);
        copy(buf.as_mut_ptr().add(cn_off) as *mut f32, &c);
    }
    execute_thunks(&sched, arena.raw_buf_mut());

    let actual: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(yn_off) as *const f32;
        (0..bch * s * h).map(|i| *p.add(i)).collect()
    };

    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!(
            (e - a).abs() < 1e-3,
            "mismatch at {i}: expected {e}, got {a}"
        );
    }
}

/// Plan #26: 1×1 conv lowers to per-batch sgemm and matches the
/// scalar 7-loop reference.
#[test]
fn conv_1x1_fast_path_matches_scalar() {
    use rlx_ir::Philox4x32;
    // [N=2, C_in=4, H=3, W=3]
    let n = 2usize;
    let c_in = 4usize;
    let h = 3usize;
    let w = 3usize;
    let c_out = 5usize;
    let mut rng = Philox4x32::new(31);
    let mut x = vec![0f32; n * c_in * h * w];
    rng.fill_normal(&mut x);
    let mut weight = vec![0f32; c_out * c_in];
    rng.fill_normal(&mut weight);

    // Reference: scalar 1×1 conv = per-batch matmul
    // out[ni, co, hi, wi] = sum_ci weight[co, ci] * x[ni, ci, hi, wi]
    let mut expected = vec![0f32; n * c_out * h * w];
    for ni in 0..n {
        for co in 0..c_out {
            for hi in 0..h {
                for wi in 0..w {
                    let mut acc = 0f32;
                    for ci in 0..c_in {
                        acc += weight[co * c_in + ci] * x[((ni * c_in) + ci) * h * w + hi * w + wi];
                    }
                    expected[((ni * c_out) + co) * h * w + hi * w + wi] = acc;
                }
            }
        }
    }

    // RLX path: build a graph with Op::Conv (kernel=[1,1], stride=[1,1], etc).
    let f = DType::F32;
    let mut g = Graph::new("conv1x1");
    let xn = g.input("x", Shape::new(&[n, c_in, h, w], f));
    let wn = g.param("w", Shape::new(&[c_out, c_in, 1, 1], f));
    // Manually add Op::Conv since there's no `g.conv()` helper.
    let cn = g.add_node(
        rlx_ir::Op::Conv {
            kernel_size: vec![1, 1],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![xn, wn],
        Shape::new(&[n, c_out, h, w], f),
    );
    g.set_outputs(vec![cn]);

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);

    // Verify the fast path was selected.
    let saw_fast = sched
        .thunks
        .iter()
        .any(|t| matches!(t, Thunk::Conv2D1x1 { .. }));
    let saw_slow = sched
        .thunks
        .iter()
        .any(|t| matches!(t, Thunk::Conv2D { .. }));
    assert!(saw_fast, "1×1 conv should emit Conv2D1x1");
    assert!(!saw_slow, "1×1 conv must not fall through to scalar Conv2D");

    let xn_off = arena.byte_offset(xn);
    let wn_off = arena.byte_offset(wn);
    let cn_off = arena.byte_offset(cn);
    let buf = arena.raw_buf_mut();
    unsafe {
        let xp = buf.as_mut_ptr().add(xn_off) as *mut f32;
        for (i, &v) in x.iter().enumerate() {
            *xp.add(i) = v;
        }
        let wp = buf.as_mut_ptr().add(wn_off) as *mut f32;
        for (i, &v) in weight.iter().enumerate() {
            *wp.add(i) = v;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());

    let actual: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(cn_off) as *const f32;
        (0..(n * c_out * h * w)).map(|i| *p.add(i)).collect()
    };

    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!(
            (e - a).abs() < 1e-3,
            "mismatch at {i}: expected {e}, got {a}"
        );
    }
}

/// Plan #5: fused dequant matmul matches the dequant-then-matmul
/// reference (i.e. `(scale * (q - z)) @ x` materialized).
#[test]
fn dequant_matmul_int8_sym_matches_reference() {
    use rlx_ir::Philox4x32;
    use rlx_ir::quant::QuantScheme;

    let m = 3usize;
    let k = 8usize;
    let n = 4usize;
    let block_size = 4usize; // 2 blocks per column
    let blocks_per_col = k / block_size;

    // Random inputs: x f32, w_q i8, scales f32. Symmetric → no zp.
    let mut rng = Philox4x32::new(99);
    let mut x = vec![0f32; m * k];
    rng.fill_normal(&mut x);
    let w_q: Vec<i8> = (0..(k * n))
        .map(|i| ((i as i32 * 13 + 7) % 127 - 63) as i8)
        .collect();
    let scales: Vec<f32> = (0..(blocks_per_col * n))
        .map(|i| 0.01 + 0.001 * i as f32)
        .collect();

    // Reference: build f32 weights from (q * scale) per block.
    let mut w_f32 = vec![0f32; k * n];
    for p in 0..k {
        let block = p / block_size;
        for j in 0..n {
            let s = scales[block * n + j];
            w_f32[p * n + j] = w_q[p * n + j] as f32 * s;
        }
    }
    let mut expected = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += x[i * k + p] * w_f32[p * n + j];
            }
            expected[i * n + j] = acc;
        }
    }

    // RLX path.
    let f = DType::F32;
    let mut g = Graph::new("dq");
    let xn = g.input("x", Shape::new(&[m, k], f));
    let wn = g.param("w", Shape::new(&[k, n], DType::I8));
    let sn = g.param("scale", Shape::new(&[blocks_per_col, n], f));
    let zn = g.param("zp", Shape::new(&[blocks_per_col, n], f)); // unused (sym)
    let dq = g.dequant_matmul(
        xn,
        wn,
        sn,
        zn,
        QuantScheme::Int8Block {
            block_size: block_size as u32,
        },
        Shape::new(&[m, n], f),
    );
    g.set_outputs(vec![dq]);

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);

    let xn_off = arena.byte_offset(xn);
    let wn_off = arena.byte_offset(wn);
    let sn_off = arena.byte_offset(sn);
    let zn_off = arena.byte_offset(zn);
    let dq_off = arena.byte_offset(dq);
    let buf = arena.raw_buf_mut();
    unsafe {
        // Seed f32 inputs.
        let xp = buf.as_mut_ptr().add(xn_off) as *mut f32;
        for (i, &v) in x.iter().enumerate() {
            *xp.add(i) = v;
        }
        let sp = buf.as_mut_ptr().add(sn_off) as *mut f32;
        for (i, &v) in scales.iter().enumerate() {
            *sp.add(i) = v;
        }
        let zp = buf.as_mut_ptr().add(zn_off) as *mut f32;
        for i in 0..(blocks_per_col * n) {
            *zp.add(i) = 0.0;
        }
        // Seed i8 weights byte-by-byte.
        let wp = buf.as_mut_ptr().add(wn_off) as *mut i8;
        for (i, &v) in w_q.iter().enumerate() {
            *wp.add(i) = v;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());

    let actual: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(dq_off) as *const f32;
        (0..m * n).map(|i| *p.add(i)).collect()
    };

    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!(
            (e - a).abs() < 1e-3,
            "mismatch at {i}: expected {e}, got {a}"
        );
    }
}

/// Plan #9: LoRA matmul matches the unfused 3-matmul reference.
#[test]
fn lora_matmul_matches_unfused_reference() {
    use rlx_ir::Philox4x32;

    let m = 4usize;
    let k = 8usize;
    let n = 6usize;
    let r = 2usize;
    let scale = 0.5f32;

    // Random inputs (deterministic via Philox).
    let mut rng = Philox4x32::new(42);
    let mut x = vec![0f32; m * k];
    rng.fill_normal(&mut x);
    let mut w = vec![0f32; k * n];
    rng.fill_normal(&mut w);
    let mut a = vec![0f32; k * r];
    rng.fill_normal(&mut a);
    let mut b = vec![0f32; r * n];
    rng.fill_normal(&mut b);

    // Reference: out = x·W + scale * x·A·B. Naive triple-loop.
    let naive = |a_buf: &[f32], b_buf: &[f32], rows: usize, inner: usize, cols: usize| {
        let mut o = vec![0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                let mut acc = 0f32;
                for p in 0..inner {
                    acc += a_buf[i * inner + p] * b_buf[p * cols + j];
                }
                o[i * cols + j] = acc;
            }
        }
        o
    };
    let xw = naive(&x, &w, m, k, n);
    let xa = naive(&x, &a, m, k, r);
    let xab = naive(&xa, &b, m, r, n);
    let mut expected = xw;
    for i in 0..(m * n) {
        expected[i] += scale * xab[i];
    }

    // RLX path: build a graph with one LoraMatMul.
    let f = DType::F32;
    let mut g = Graph::new("lora");
    let xn = g.input("x", Shape::new(&[m, k], f));
    let wn = g.param("w", Shape::new(&[k, n], f));
    let an = g.param("a", Shape::new(&[k, r], f));
    let bn = g.param("b", Shape::new(&[r, n], f));
    let lm = g.lora_matmul(xn, wn, an, bn, scale, Shape::new(&[m, n], f));
    g.set_outputs(vec![lm]);

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);

    let xn_off = arena.byte_offset(xn);
    let wn_off = arena.byte_offset(wn);
    let an_off = arena.byte_offset(an);
    let bn_off = arena.byte_offset(bn);
    let lm_off = arena.byte_offset(lm);
    let buf = arena.raw_buf_mut();
    unsafe {
        let copy = |dst: *mut f32, data: &[f32]| {
            for (i, &v) in data.iter().enumerate() {
                *dst.add(i) = v;
            }
        };
        copy(buf.as_mut_ptr().add(xn_off) as *mut f32, &x);
        copy(buf.as_mut_ptr().add(wn_off) as *mut f32, &w);
        copy(buf.as_mut_ptr().add(an_off) as *mut f32, &a);
        copy(buf.as_mut_ptr().add(bn_off) as *mut f32, &b);
    }
    execute_thunks(&sched, arena.raw_buf_mut());

    let actual: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(lm_off) as *const f32;
        (0..m * n).map(|i| *p.add(i)).collect()
    };

    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!(
            (e - a).abs() < 1e-3,
            "mismatch at {i}: expected {e}, got {a}"
        );
    }
}

/// Plan #42: fused sampling kernel determinism + greedy fallback.
#[test]
fn sample_temperature_zero_is_argmax() {
    // Very low temperature → distribution collapses on argmax.
    // Same seed → same output bit-for-bit.
    let f = DType::F32;
    let mut g = Graph::new("samp");
    let logits = g.input("logits", Shape::new(&[1, 8], f));
    let s = g.sample(logits, 0, 1.0, 1e-3, 42, Shape::new(&[1], f));
    g.set_outputs(vec![s]);
    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);

    let logits_off = arena.byte_offset(logits);
    let s_off = arena.byte_offset(s);
    let buf = arena.raw_buf_mut();
    unsafe {
        let p = buf.as_mut_ptr().add(logits_off) as *mut f32;
        // argmax = index 5 (value 9.0).
        let inputs = [0.1f32, 0.2, 0.3, 0.4, 0.5, 9.0, 0.7, 0.8];
        for (i, &v) in inputs.iter().enumerate() {
            *p.add(i) = v;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());

    let token = unsafe {
        let p = arena.raw_buf().as_ptr().add(s_off) as *const f32;
        *p as usize
    };
    assert_eq!(token, 5, "low-temp sampling should pick the argmax");
}

#[test]
fn sample_top_k_one_is_deterministic() {
    // top_k=1 forces only the argmax to have nonzero probability.
    let f = DType::F32;
    let mut g = Graph::new("samp_k1");
    let logits = g.input("logits", Shape::new(&[1, 4], f));
    let s = g.sample(logits, 1, 1.0, 1.0, 7, Shape::new(&[1], f));
    g.set_outputs(vec![s]);
    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);

    let logits_off = arena.byte_offset(logits);
    let s_off = arena.byte_offset(s);
    let buf = arena.raw_buf_mut();
    unsafe {
        let p = buf.as_mut_ptr().add(logits_off) as *mut f32;
        let inputs = [0.1f32, 5.0, 0.3, 0.4]; // argmax = 1
        for (i, &v) in inputs.iter().enumerate() {
            *p.add(i) = v;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let token = unsafe {
        let p = arena.raw_buf().as_ptr().add(s_off) as *const f32;
        *p as usize
    };
    assert_eq!(token, 1);
}

/// Plan #44: cumsum primitive parity vs. naive scan.
#[test]
fn cumsum_inclusive_matches_naive() {
    let f = DType::F32;
    let mut g = Graph::new("cumsum");
    let x = g.input("x", Shape::new(&[2, 4], f));
    let cs = g.cumsum(x, -1, false, Shape::new(&[2, 4], f));
    g.set_outputs(vec![cs]);
    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);

    // Cache offsets up-front so we can drop the immutable borrow.
    let x_off = arena.byte_offset(x);
    let out_off = arena.byte_offset(cs);
    let buf = arena.raw_buf_mut();
    unsafe {
        let p = buf.as_mut_ptr().add(x_off) as *mut f32;
        let inputs = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        for (i, &v) in inputs.iter().enumerate() {
            *p.add(i) = v;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());

    let out: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(out_off) as *const f32;
        (0..8).map(|i| *p.add(i)).collect()
    };
    assert_eq!(out, vec![1.0, 3.0, 6.0, 10.0, 10.0, 30.0, 60.0, 100.0]);
}

/// Plan #46 deep: Narrow×3 → Attention fusion. The three QKV
/// narrows that BERT/Nomic emit on the unfused (batch*seq > 64)
/// path collapse into a single strided-Attention thunk.
#[test]
fn narrow_attention_fuses_in_unfused_path() {
    let f = DType::F32;
    let mut g = Graph::new("nattn_fuse");
    // batch*seq = 8*16 = 128 > 64 so FusedAttnBlock skips.
    let qkv = g.input("qkv", Shape::new(&[8, 16, 192], f)); // 3*64 = 192
    let mask = g.input("mask", Shape::new(&[8, 16], f));
    let q = g.narrow_(qkv, 2, 0, 64);
    let k = g.narrow_(qkv, 2, 64, 64);
    let v = g.narrow_(qkv, 2, 128, 64);
    let attn = g.attention(q, k, v, mask, 4, 16, Shape::new(&[8, 16, 64], f));
    g.set_outputs(vec![attn]);

    let plan = rlx_opt::memory::plan_memory(&g);
    let arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);

    let mut narrow_count = 0;
    let mut attn_strides: Option<(u32, u32, u32)> = None;
    for t in &sched.thunks {
        match t {
            Thunk::Narrow { .. } => narrow_count += 1,
            Thunk::Attention {
                q_row_stride,
                k_row_stride,
                v_row_stride,
                ..
            } => attn_strides = Some((*q_row_stride, *k_row_stride, *v_row_stride)),
            _ => {}
        }
    }
    // After fusion the 3 narrows are gone; Attention now walks the
    // QKV with parent row stride = 192 (3 × 64) on all three inputs.
    assert_eq!(
        narrow_count, 0,
        "Narrow×3→Attention fusion should eliminate all 3 narrows; saw {narrow_count}"
    );
    assert_eq!(
        attn_strides,
        Some((192, 192, 192)),
        "Attention should walk Q/K/V with parent row stride 192"
    );
}

/// Regression: when the QKV→Narrow×3→RoPE×2→Attention→OutProj chain
/// collapses into a single `FusedAttnBlock` (small batch·seq), the fused
/// kernel must honor a **causal** mask. It previously applied only the
/// per-key padding mask and dropped `mask_kind`, so a later token leaked
/// into earlier positions (decoder attention attended to the future).
#[test]
fn fused_attn_block_respects_causal_mask() {
    let f = DType::F32;
    let (s, d, nh, dh) = (5usize, 8usize, 2usize, 4usize);
    let half = dh / 2;

    let mut g = Graph::new("fused_causal");
    let hidden = g.input("hidden", Shape::new(&[s, d], f));
    let wqkv = g.input("wqkv", Shape::new(&[d, 3 * d], f));
    let wo = g.input("wo", Shape::new(&[d, d], f));
    let cos = g.input("cos", Shape::new(&[s, half], f));
    let sin = g.input("sin", Shape::new(&[s, half], f));
    let qkv = g.matmul(hidden, wqkv, Shape::new(&[s, 3 * d], f));
    let q = g.narrow_(qkv, 1, 0, d);
    let k = g.narrow_(qkv, 1, d, d);
    let v = g.narrow_(qkv, 1, 2 * d, d);
    let q3 = g.reshape(q, vec![1, s as i64, d as i64], Shape::new(&[1, s, d], f));
    let k3 = g.reshape(k, vec![1, s as i64, d as i64], Shape::new(&[1, s, d], f));
    let v3 = g.reshape(v, vec![1, s as i64, d as i64], Shape::new(&[1, s, d], f));
    let qr = g.rope(q3, cos, sin, dh);
    let kr = g.rope(k3, cos, sin, dh);
    let attn = g.attention_kind(
        qr,
        kr,
        v3,
        nh,
        dh,
        rlx_ir::op::MaskKind::Causal,
        Shape::new(&[1, s, d], f),
    );
    let a2 = g.reshape(attn, vec![s as i64, d as i64], Shape::new(&[s, d], f));
    let out = g.matmul(a2, wo, Shape::new(&[s, d], f));
    g.set_outputs(vec![out]);

    // The fusion must actually fire AND carry the causal kind.
    let plan = rlx_opt::memory::plan_memory(&g);
    let arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    assert!(
        sched.thunks.iter().any(|t| matches!(
            t,
            Thunk::FusedAttnBlock {
                mask_kind: rlx_ir::op::MaskKind::Causal,
                ..
            }
        )),
        "expected a FusedAttnBlock carrying MaskKind::Causal"
    );

    let wqkv_d: Vec<f32> = (0..d * 3 * d)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let wo_d: Vec<f32> = (0..d * d).map(|i| ((i % 5) as f32 - 2.0) * 0.05).collect();
    let mut cos_d = vec![0f32; s * half];
    let mut sin_d = vec![0f32; s * half];
    for p in 0..s {
        for i in 0..half {
            let fr = 1.0f32 / 10000f32.powf(2.0 * i as f32 / dh as f32);
            cos_d[p * half + i] = (p as f32 * fr).cos();
            sin_d[p * half + i] = (p as f32 * fr).sin();
        }
    }
    let base_h: Vec<f32> = (0..s * d).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
    let run = |hin: &[f32]| {
        run_graph(
            &g,
            &[
                (hidden, hin),
                (wqkv, &wqkv_d),
                (wo, &wo_d),
                (cos, &cos_d),
                (sin, &sin_d),
            ],
            out,
            s * d,
        )
    };
    let a = run(&base_h);
    // Perturb only the LAST position's hidden row.
    let mut changed = base_h.clone();
    for j in 0..d {
        changed[4 * d + j] += 1.0;
    }
    let b = run(&changed);
    for pos in 0..4 {
        for j in 0..d {
            let i = pos * d + j;
            assert!(
                (a[i] - b[i]).abs() < 1e-5,
                "causal leak at pos {pos}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }
    let last: f32 = (0..d).map(|j| (a[4 * d + j] - b[4 * d + j]).abs()).sum();
    assert!(last > 1e-4, "last position must react to its own token");
}

// ── Backward / training op parity tests ────────────────────
//
// Strategy: build a graph that contains exactly the backward op
// under test (plus its inputs as graph Inputs), execute, and
// compare against a hand-rolled scalar reference. For
// Conv2dBackwardInput we additionally check against the numerical
// gradient of the forward Conv2D — that's the gold-standard test
// that validates the math, not just consistency between two
// implementations of the same formula.

fn run_graph(g: &Graph, inputs: &[(NodeId, &[f32])], out_id: NodeId, out_len: usize) -> Vec<f32> {
    let plan = rlx_opt::memory::plan_memory(g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(g, &arena);
    for &(id, data) in inputs {
        let off = arena.byte_offset(id);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(off) as *mut f32;
            for (i, &v) in data.iter().enumerate() {
                *p.add(i) = v;
            }
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let off = arena.byte_offset(out_id);
    unsafe {
        let p = arena.raw_buf().as_ptr().add(off) as *const f32;
        (0..out_len).map(|i| *p.add(i)).collect()
    }
}

#[test]
fn relu_backward_matches_mask() {
    let f = DType::F32;
    let len = 7usize;
    let x: Vec<f32> = vec![-2.0, -0.1, 0.0, 0.1, 1.0, 3.0, -5.0];
    let dy: Vec<f32> = vec![0.5, 1.5, 2.5, -0.7, 4.0, -1.0, 9.0];

    let mut g = Graph::new("relu_bw");
    let xn = g.input("x", Shape::new(&[len], f));
    let dyn_ = g.input("dy", Shape::new(&[len], f));
    let dx = g.relu_backward(xn, dyn_);
    g.set_outputs(vec![dx]);

    let actual = run_graph(&g, &[(xn, &x), (dyn_, &dy)], dx, len);
    // Reference: gradient is dy where x>0 strictly, else 0.
    // (zero is not "positive" — the forward applied max(0, x), and at
    // x=0 the subgradient could be anything in [0, dy]; we pick 0.)
    let expected: Vec<f32> = x
        .iter()
        .zip(&dy)
        .map(|(&xi, &dyi)| if xi > 0.0 { dyi } else { 0.0 })
        .collect();
    for (a, e) in actual.iter().zip(&expected) {
        assert!((a - e).abs() < 1e-6, "relu_bw mismatch: {a} vs {e}");
    }
}

#[test]
fn maxpool2d_backward_routes_to_argmax() {
    let f = DType::F32;
    // [N=1, C=1, H=4, W=4] → 2x2 max-pool stride 2 → [1,1,2,2].
    let x: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    // Argmax of each 2x2 window:
    //   (0,0)→6 (idx 5), (0,1)→8 (idx 7),
    //   (1,0)→14(idx 13),(1,1)→16(idx 15).
    let dy: Vec<f32> = vec![0.5, 1.0, 2.0, 4.0];

    let mut g = Graph::new("maxpool_bw");
    let xn = g.input("x", Shape::new(&[1, 1, 4, 4], f));
    let dyn_ = g.input("dy", Shape::new(&[1, 1, 2, 2], f));
    let dx = g.maxpool2d_backward(xn, dyn_, vec![2, 2], vec![2, 2], vec![0, 0]);
    g.set_outputs(vec![dx]);

    let actual = run_graph(&g, &[(xn, &x), (dyn_, &dy)], dx, 16);
    let mut expected = vec![0f32; 16];
    expected[5] = 0.5;
    expected[7] = 1.0;
    expected[13] = 2.0;
    expected[15] = 4.0;
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!((a - e).abs() < 1e-6, "maxpool_bw[{i}] mismatch: {a} vs {e}");
    }
}

#[test]
fn conv2d_backward_input_matches_numerical_gradient() {
    use rlx_ir::Philox4x32;
    // Small enough to numerically differentiate exhaustively but
    // big enough to exercise stride/padding edge cases.
    let n = 1usize;
    let c_in = 2usize;
    let h = 4usize;
    let w = 4usize;
    let c_out = 3usize;
    let kh = 3usize;
    let kw = 3usize;
    let ph = 1usize;
    let pw = 1usize;
    let sh = 1usize;
    let sw = 1usize;
    // Output dims with padding=1, stride=1: same as input.
    let h_out = (h + 2 * ph - kh) / sh + 1;
    let w_out = (w + 2 * pw - kw) / sw + 1;
    assert_eq!(h_out, 4);
    assert_eq!(w_out, 4);

    let mut rng = Philox4x32::new(7);
    let mut x = vec![0f32; n * c_in * h * w];
    rng.fill_normal(&mut x);
    let mut wt = vec![0f32; c_out * c_in * kh * kw];
    rng.fill_normal(&mut wt);
    let mut dy = vec![0f32; n * c_out * h_out * w_out];
    rng.fill_normal(&mut dy);

    // Analytical: Conv2dBackwardInput on (dy, w).
    let f = DType::F32;
    let mut g = Graph::new("conv_bwi");
    let dy_in = g.input("dy", Shape::new(&[n, c_out, h_out, w_out], f));
    let w_in = g.input("w", Shape::new(&[c_out, c_in, kh, kw], f));
    let dx = g.conv2d_backward_input(
        dy_in,
        w_in,
        Shape::new(&[n, c_in, h, w], f),
        vec![kh, kw],
        vec![sh, sw],
        vec![ph, pw],
        vec![1, 1],
        1,
    );
    g.set_outputs(vec![dx]);
    let analytical = run_graph(&g, &[(dy_in, &dy), (w_in, &wt)], dx, n * c_in * h * w);

    // Numerical: for each x[i], finite-difference forward conv twice.
    // Forward: y[j] = sum over filter window of w * x ; dot(dy, y) is
    // the scalar we differentiate. Then dx[i] = ∂(dot(dy, y))/∂x[i].
    let forward = |x: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; n * c_out * h_out * w_out];
        for ni in 0..n {
            for co in 0..c_out {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let mut acc = 0f32;
                        for ci in 0..c_in {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let hi = ho * sh + ki;
                                    let wi = wo * sw + kj;
                                    if hi < ph || wi < pw {
                                        continue;
                                    }
                                    let hi = hi - ph;
                                    let wi = wi - pw;
                                    if hi >= h || wi >= w {
                                        continue;
                                    }
                                    let xv = x[((ni * c_in) + ci) * h * w + hi * w + wi];
                                    let wv = wt[((co * c_in) + ci) * kh * kw + ki * kw + kj];
                                    acc += xv * wv;
                                }
                            }
                        }
                        out[((ni * c_out) + co) * h_out * w_out + ho * w_out + wo] = acc;
                    }
                }
            }
        }
        out
    };
    let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(&u, &v)| u * v).sum() };
    let eps = 1e-3f32;
    let mut numerical = vec![0f32; x.len()];
    for i in 0..x.len() {
        let saved = x[i];
        x[i] = saved + eps;
        let plus = dot(&forward(&x), &dy);
        x[i] = saved - eps;
        let minus = dot(&forward(&x), &dy);
        x[i] = saved;
        numerical[i] = (plus - minus) / (2.0 * eps);
    }
    for (i, (a, n)) in analytical.iter().zip(&numerical).enumerate() {
        // f32 + eps=1e-3 numerical grad → ~1e-3 absolute is realistic.
        assert!(
            (a - n).abs() < 5e-3,
            "conv_bw_input[{i}]: analytical {a} vs numerical {n}"
        );
    }
}

#[test]
fn conv2d_backward_weight_matches_numerical_gradient() {
    use rlx_ir::Philox4x32;
    let n = 2usize;
    let c_in = 2usize;
    let h = 4usize;
    let w = 4usize;
    let c_out = 2usize;
    let kh = 3usize;
    let kw = 3usize;
    let ph = 0usize;
    let pw = 0usize;
    let sh = 1usize;
    let sw = 1usize;
    let h_out = (h + 2 * ph - kh) / sh + 1;
    let w_out = (w + 2 * pw - kw) / sw + 1;

    let mut rng = Philox4x32::new(11);
    let mut x = vec![0f32; n * c_in * h * w];
    rng.fill_normal(&mut x);
    let mut wt = vec![0f32; c_out * c_in * kh * kw];
    rng.fill_normal(&mut wt);
    let mut dy = vec![0f32; n * c_out * h_out * w_out];
    rng.fill_normal(&mut dy);

    let f = DType::F32;
    let mut g = Graph::new("conv_bww");
    let xn = g.input("x", Shape::new(&[n, c_in, h, w], f));
    let dyn_ = g.input("dy", Shape::new(&[n, c_out, h_out, w_out], f));
    let dwn = g.conv2d_backward_weight(
        xn,
        dyn_,
        Shape::new(&[c_out, c_in, kh, kw], f),
        vec![kh, kw],
        vec![sh, sw],
        vec![ph, pw],
        vec![1, 1],
        1,
    );
    g.set_outputs(vec![dwn]);
    let analytical = run_graph(&g, &[(xn, &x), (dyn_, &dy)], dwn, c_out * c_in * kh * kw);

    let forward = |wt: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; n * c_out * h_out * w_out];
        for ni in 0..n {
            for co in 0..c_out {
                for ho in 0..h_out {
                    for wo in 0..w_out {
                        let mut acc = 0f32;
                        for ci in 0..c_in {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let hi = ho + ki;
                                    let wi = wo + kj;
                                    let xv = x[((ni * c_in) + ci) * h * w + hi * w + wi];
                                    let wv = wt[((co * c_in) + ci) * kh * kw + ki * kw + kj];
                                    acc += xv * wv;
                                }
                            }
                        }
                        out[((ni * c_out) + co) * h_out * w_out + ho * w_out + wo] = acc;
                    }
                }
            }
        }
        out
    };
    let dot = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(&u, &v)| u * v).sum() };
    let eps = 1e-3f32;
    let mut numerical = vec![0f32; wt.len()];
    for i in 0..wt.len() {
        let saved = wt[i];
        wt[i] = saved + eps;
        let plus = dot(&forward(&wt), &dy);
        wt[i] = saved - eps;
        let minus = dot(&forward(&wt), &dy);
        wt[i] = saved;
        numerical[i] = (plus - minus) / (2.0 * eps);
    }
    for (i, (a, n)) in analytical.iter().zip(&numerical).enumerate() {
        assert!(
            (a - n).abs() < 5e-3,
            "conv_bw_weight[{i}]: analytical {a} vs numerical {n}"
        );
    }
}

#[test]
fn softmax_cross_entropy_matches_reference() {
    let f = DType::F32;
    let logits: Vec<f32> = vec![
        1.0, 2.0, 3.0, // row 0: max=3 (idx 2)
        -1.0, 0.0, 4.0, // row 1: max=4 (idx 2)
        5.0, 5.0, 5.0, // row 2: uniform
    ];
    let labels: Vec<f32> = vec![2.0, 0.0, 1.0];

    let mut g = Graph::new("sce");
    let lg = g.input("logits", Shape::new(&[3, 3], f));
    let lb = g.input("labels", Shape::new(&[3], f));
    let loss = g.softmax_cross_entropy_with_logits(lg, lb);
    g.set_outputs(vec![loss]);
    let actual = run_graph(&g, &[(lg, &logits), (lb, &labels)], loss, 3);

    // Reference per-row: -log(softmax(row)[label]).
    let mut expected = vec![0f32; 3];
    for ni in 0..3 {
        let row = &logits[ni * 3..(ni + 1) * 3];
        let m = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
        let sum: f32 = row.iter().map(|&v| (v - m).exp()).sum();
        let lse = m + sum.ln();
        let label_idx = labels[ni] as usize;
        expected[ni] = lse - row[label_idx];
    }
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!((a - e).abs() < 1e-5, "sce loss[{i}]: {a} vs {e}");
    }
}

#[test]
fn softmax_cross_entropy_backward_matches_numerical_gradient() {
    use rlx_ir::Philox4x32;
    let n = 4usize;
    let c = 5usize;
    let mut rng = Philox4x32::new(23);
    let mut logits = vec![0f32; n * c];
    rng.fill_normal(&mut logits);
    let labels: Vec<f32> = (0..n).map(|i| (i % c) as f32).collect();
    let mut d_loss = vec![0f32; n];
    rng.fill_normal(&mut d_loss);

    let f = DType::F32;
    let mut g = Graph::new("sce_bw");
    let lg = g.input("logits", Shape::new(&[n, c], f));
    let lb = g.input("labels", Shape::new(&[n], f));
    let dl = g.input("d_loss", Shape::new(&[n], f));
    let dlogits = g.softmax_cross_entropy_backward(lg, lb, dl);
    g.set_outputs(vec![dlogits]);
    let analytical = run_graph(
        &g,
        &[(lg, &logits), (lb, &labels), (dl, &d_loss)],
        dlogits,
        n * c,
    );

    // Numerical: differentiate dot(d_loss, sce_loss(logits)) w.r.t. each logit.
    let sce_loss = |logits: &[f32]| -> Vec<f32> {
        let mut out = vec![0f32; n];
        for ni in 0..n {
            let row = &logits[ni * c..(ni + 1) * c];
            let m = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            let sum: f32 = row.iter().map(|&v| (v - m).exp()).sum();
            out[ni] = (m + sum.ln()) - row[labels[ni] as usize];
        }
        out
    };
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(&u, &v)| u * v).sum::<f32>();
    let eps = 1e-3f32;
    let mut numerical = vec![0f32; logits.len()];
    for i in 0..logits.len() {
        let saved = logits[i];
        logits[i] = saved + eps;
        let plus = dot(&sce_loss(&logits), &d_loss);
        logits[i] = saved - eps;
        let minus = dot(&sce_loss(&logits), &d_loss);
        logits[i] = saved;
        numerical[i] = (plus - minus) / (2.0 * eps);
    }
    for (i, (a, num)) in analytical.iter().zip(&numerical).enumerate() {
        assert!(
            (a - num).abs() < 5e-3,
            "sce_bw[{i}]: analytical {a} vs numerical {num}"
        );
    }
}

// ── End-to-end autodiff parity tests ──────────────────────
//
// Build a forward graph, run `grad_with_loss` to produce a graph
// that emits [loss, gradients...], execute it through rlx-cpu,
// and compare each gradient to a finite-difference estimate
// produced by re-running the forward graph with each parameter
// entry perturbed. f32 + ε=1e-3 puts the tolerance floor around
// 5e-3 absolute error.

/// Initialize Op::Constant slots in the arena with their literal
/// data. Mirrors the loop in rlx_runtime::backend (which serves
/// the same role for production runs).
fn fill_constants_into_arena(graph: &Graph, arena: &mut crate::arena::Arena) {
    for node in graph.nodes() {
        if let Op::Constant { data } = &node.op
            && arena.has_buffer(node.id)
            && !data.is_empty()
        {
            let buf = arena.slice_mut(node.id);
            let n_floats = data.len() / 4;
            let n = buf.len().min(n_floats);
            for i in 0..n {
                let bytes = [
                    data[i * 4],
                    data[i * 4 + 1],
                    data[i * 4 + 2],
                    data[i * 4 + 3],
                ];
                buf[i] = f32::from_le_bytes(bytes);
            }
        }
    }
}

/// Compile + arena-prep helper for these tests. Returns the
/// schedule and a populated arena. `seed_inputs` writes f32 input
/// data into the arena slot for each (NodeId, &[f32]) pair.
fn prepare(
    graph: &Graph,
    seed_inputs: &[(NodeId, &[f32])],
) -> (ThunkSchedule, crate::arena::Arena) {
    let plan = rlx_opt::memory::plan_memory(graph);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(graph, &arena);
    fill_constants_into_arena(graph, &mut arena);
    for &(id, data) in seed_inputs {
        let off = arena.byte_offset(id);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(off) as *mut f32;
            for (i, &v) in data.iter().enumerate() {
                *p.add(i) = v;
            }
        }
    }
    (sched, arena)
}

fn read_arena(arena: &crate::arena::Arena, id: NodeId, len: usize) -> Vec<f32> {
    let off = arena.byte_offset(id);
    unsafe {
        let p = arena.raw_buf().as_ptr().add(off) as *const f32;
        (0..len).map(|i| *p.add(i)).collect()
    }
}

fn write_arena(arena: &mut crate::arena::Arena, id: NodeId, data: &[f32]) {
    let off = arena.byte_offset(id);
    let buf = arena.raw_buf_mut();
    unsafe {
        let p = buf.as_mut_ptr().add(off) as *mut f32;
        for (i, &v) in data.iter().enumerate() {
            *p.add(i) = v;
        }
    }
}

/// f64 sibling of `prepare`. Writes f64 input data into the arena.
fn prepare_f64(
    graph: &Graph,
    seed_inputs: &[(NodeId, &[f64])],
) -> (ThunkSchedule, crate::arena::Arena) {
    let plan = rlx_opt::memory::plan_memory(graph);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(graph, &arena);
    fill_constants_into_arena(graph, &mut arena);
    for &(id, data) in seed_inputs {
        let off = arena.byte_offset(id);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(off) as *mut f64;
            for (i, &v) in data.iter().enumerate() {
                *p.add(i) = v;
            }
        }
    }
    (sched, arena)
}

fn read_arena_f64(arena: &crate::arena::Arena, id: NodeId, len: usize) -> Vec<f64> {
    let off = arena.byte_offset(id);
    unsafe {
        let p = arena.raw_buf().as_ptr().add(off) as *const f64;
        (0..len).map(|i| *p.add(i)).collect()
    }
}

/// End-to-end f64 DenseSolve through the full compile + execute
/// path. Validates: IR shape inference, memory planner f64 sizing,
/// arena f64 accessors, Thunk::DenseSolveF64 lowering, executor
/// dispatch, Accelerate dgesv FFI.
///
/// System:
///   A = [[2, 1],
///        [1, 3]]   b = [5, 10]
///   ⇒  x = [1, 3]   (verified by hand)
#[test]
fn dense_solve_f64_end_to_end() {
    let mut g = Graph::new("solve_e2e");
    let a = g.input("A", Shape::new(&[2, 2], DType::F64));
    let b = g.input("b", Shape::new(&[2], DType::F64));
    let x = g.dense_solve(a, b, Shape::new(&[2], DType::F64));
    g.set_outputs(vec![x]);

    let a_data = [2.0, 1.0, 1.0, 3.0_f64];
    let b_data = [5.0, 10.0_f64];
    let (sched, mut arena) = prepare_f64(&g, &[(a, &a_data), (b, &b_data)]);
    execute_thunks(&sched, arena.raw_buf_mut());

    let got = read_arena_f64(&arena, x, 2);
    let want = [1.0, 3.0_f64];
    for i in 0..2 {
        assert!(
            (got[i] - want[i]).abs() < 1e-12,
            "x[{i}] = {} (expected {})",
            got[i],
            want[i]
        );
    }
}

/// Scaled-up f64 DenseSolve — tridiagonal Laplacian-shape (typical
/// MNA structure for a passive RC mesh in Circulax). Validates
/// that the solve scales beyond the trivial 2×2 and that the
/// row-major ↔ col-major dance in `dgesv` is correct for the
/// general case.
#[test]
fn dense_solve_f64_5x5_laplacian() {
    let n = 5usize;
    let mut g = Graph::new("solve_5x5");
    let a = g.input("A", Shape::new(&[n, n], DType::F64));
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let x = g.dense_solve(a, b, Shape::new(&[n], DType::F64));
    g.set_outputs(vec![x]);

    // 1-D Laplacian: 2 on diagonal, -1 on off-diagonals, 0 elsewhere.
    let mut a_data = vec![0.0_f64; n * n];
    for i in 0..n {
        a_data[i * n + i] = 2.0;
        if i > 0 {
            a_data[i * n + (i - 1)] = -1.0;
        }
        if i + 1 < n {
            a_data[i * n + (i + 1)] = -1.0;
        }
    }
    let b_data: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let (sched, mut arena) = prepare_f64(&g, &[(a, &a_data), (b, &b_data)]);
    execute_thunks(&sched, arena.raw_buf_mut());

    let got = read_arena_f64(&arena, x, n);
    // Verify A·x ≈ b by computing the residual.
    let mut residual = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            residual[i] += a_data[i * n + j] * got[j];
        }
    }
    for i in 0..n {
        assert!(
            (residual[i] - b_data[i]).abs() < 1e-10,
            "row {i}: residual {} vs b {}",
            residual[i],
            b_data[i]
        );
    }
}

/// Hello Resistor: end-to-end f64 gradient through a dense solve.
///
/// Forward:
///   A      : Param  [N, N]   f64
///   b      : Input  [N]      f64
///   x      = solve(A, b)            (DenseSolve)
///   loss   = sum(x)                 (Reduce::Sum)
///
/// Backward (via grad_with_loss):
///   ones [N] = expand(d_output, [N])      (Reduce::Sum VJP)
///   dx_int   = solve(Aᵀ, ones)             (DenseSolve VJP step 1)
///   dA       = -outer(dx_int, x)           (DenseSolve VJP step 2)
///   db       = dx_int                       (DenseSolve VJP step 3)
///
/// Closed form: with loss = sum(solve(A, b)) = ones·x and
/// implicit-function calculus, db = (Aᵀ)⁻¹·ones, dA = -db ⊗ x.
/// We verify this against the autodiff-emitted graph's output and
/// against a finite-difference baseline.
#[test]
fn hello_resistor_gradient_end_to_end() {
    use rlx_opt::autodiff::grad_with_loss;
    let n = 3usize;

    // ── Build forward graph ──
    let mut g = Graph::new("hello_resistor");
    let a = g.param("A", Shape::new(&[n, n], DType::F64));
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let x = g.dense_solve(a, b, Shape::new(&[n], DType::F64));
    let loss = g.reduce(
        x,
        ReduceOp::Sum,
        vec![0],
        false,
        Shape::new(&[1], DType::F64),
    );
    g.set_outputs(vec![loss]);

    // ── Run reverse-mode AD ──
    let bwd = grad_with_loss(&g, &[a, b]);
    assert_eq!(bwd.outputs.len(), 3, "expect [loss, dA, db]");

    // ── Locate the inputs the bwd graph still needs from us ──
    // grad_with_loss copies forward nodes into bwd, so A/b/d_output
    // appear under their original names. Find them by name.
    let find_by_name = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            let name = match &node.op {
                rlx_ir::Op::Input { name } => Some(name.as_str()),
                rlx_ir::Op::Param { name } => Some(name.as_str()),
                _ => None,
            };
            if name == Some(want) {
                return node.id;
            }
        }
        panic!("no node named {want:?} in bwd graph");
    };
    let a_bwd = find_by_name(&bwd, "A");
    let b_bwd = find_by_name(&bwd, "b");
    let d_out_bwd = find_by_name(&bwd, "d_output");

    // ── Test data ──
    // A = [[2,1,0],[1,3,1],[0,1,2]]   (SPD tridiagonal, well-conditioned)
    // b = [1,2,3]
    let a_data = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0_f64];
    let b_data = [1.0, 2.0, 3.0_f64];
    let d_output = [1.0_f64]; // ∂loss/∂loss

    // ── Compile + execute backward graph ──
    let (sched, mut arena) = prepare_f64(
        &bwd,
        &[(a_bwd, &a_data), (b_bwd, &b_data), (d_out_bwd, &d_output)],
    );
    execute_thunks(&sched, arena.raw_buf_mut());

    let loss_out = read_arena_f64(&arena, bwd.outputs[0], 1);
    let da_out = read_arena_f64(&arena, bwd.outputs[1], n * n);
    let db_out = read_arena_f64(&arena, bwd.outputs[2], n);

    // ── Closed-form reference ──
    // x = A⁻¹ b ; loss = sum(x).
    let x_ref = {
        let mut a = a_data;
        let mut b = b_data;
        let info = crate::blas::dgesv(&mut a, &mut b, n, 1);
        assert_eq!(info, 0);
        b
    };
    let loss_ref: f64 = x_ref.iter().sum();
    // db = (Aᵀ)⁻¹ · 1
    let db_ref = {
        let mut at = [0.0_f64; 9];
        for i in 0..n {
            for j in 0..n {
                at[i * n + j] = a_data[j * n + i];
            }
        }
        let mut ones = [1.0_f64; 3];
        let info = crate::blas::dgesv(&mut at, &mut ones, n, 1);
        assert_eq!(info, 0);
        ones
    };
    // dA = -outer(db, x) ; dA[i,j] = -db[i] * x[j]
    let mut da_ref = [0.0_f64; 9];
    for i in 0..n {
        for j in 0..n {
            da_ref[i * n + j] = -db_ref[i] * x_ref[j];
        }
    }

    // ── Assertions vs analytic answer ──
    assert!(
        (loss_out[0] - loss_ref).abs() < 1e-10,
        "loss: got {}, want {}",
        loss_out[0],
        loss_ref
    );
    for i in 0..n {
        assert!(
            (db_out[i] - db_ref[i]).abs() < 1e-10,
            "db[{i}]: got {}, want {}",
            db_out[i],
            db_ref[i]
        );
    }
    for i in 0..n * n {
        assert!(
            (da_out[i] - da_ref[i]).abs() < 1e-10,
            "dA[{i}]: got {}, want {}",
            da_out[i],
            da_ref[i]
        );
    }

    // ── Cross-check vs finite differences on db (a few entries) ──
    // ∂loss/∂b[k] ≈ (loss(b + h·e_k) - loss(b - h·e_k)) / (2h).
    let h = 1e-6_f64;
    for k in 0..n {
        let mut bp = b_data;
        bp[k] += h;
        let mut bm = b_data;
        bm[k] -= h;
        let lp = {
            let mut ac = a_data;
            let info = crate::blas::dgesv(&mut ac, &mut bp, n, 1);
            assert_eq!(info, 0);
            bp.iter().sum::<f64>()
        };
        let lm = {
            let mut ac = a_data;
            let info = crate::blas::dgesv(&mut ac, &mut bm, n, 1);
            assert_eq!(info, 0);
            bm.iter().sum::<f64>()
        };
        let fd = (lp - lm) / (2.0 * h);
        assert!(
            (db_out[k] - fd).abs() < 1e-7,
            "FD mismatch on db[{k}]: AD={} FD={}",
            db_out[k],
            fd
        );
    }
}

/// Smallest possible Op::Scan basic test: geometric growth.
/// init = [1, 1, 1] f64, body = (x → x + 0.1·x) = (x → 1.1·x),
/// length = 10. Final carry must equal init·(1.1)^10 ≈ 2.5937…
/// to f64 precision.
#[test]
fn scan_geometric_growth_f64() {
    let n = 3usize;
    let length = 10u32;

    // Body: (x) → x + 0.1·x. One Input, one output, same shape/dtype.
    let mut body = Graph::new("scan_body");
    let x = body.input("carry", Shape::new(&[n], DType::F64));
    let scale_bytes: Vec<u8> = (0..n).flat_map(|_| 0.1_f64.to_le_bytes()).collect();
    let scale = body.add_node(
        Op::Constant { data: scale_bytes },
        vec![],
        Shape::new(&[n], DType::F64),
    );
    let scaled = body.binary(BinaryOp::Mul, x, scale, Shape::new(&[n], DType::F64));
    let next = body.binary(BinaryOp::Add, x, scaled, Shape::new(&[n], DType::F64));
    body.set_outputs(vec![next]);

    // Outer graph: scan(init, body, length).
    let mut g = Graph::new("scan_outer");
    let init = g.input("init", Shape::new(&[n], DType::F64));
    let final_carry = g.scan(init, body, length);
    g.set_outputs(vec![final_carry]);

    let init_data = vec![1.0_f64; n];
    let (sched, mut arena) = prepare_f64(&g, &[(init, &init_data)]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let got = read_arena_f64(&arena, final_carry, n);
    let want: f64 = 1.1_f64.powi(length as i32);
    for i in 0..n {
        assert!(
            (got[i] - want).abs() < 1e-12,
            "got[{i}] = {} want {}",
            got[i],
            want
        );
    }
}

/// Per-step xs scan: cumulative-sum.
///   carry_0 = init
///   carry_{t+1} = carry_t + xs\[t\]
///   final = sum_{t<length} xs\[t\] + init
/// Body has 2 inputs (carry, x_t) in that NodeId order; one output
/// (next carry). Validates the per-step-input plumbing end-to-end.
#[test]
fn scan_with_xs_cumulative_sum() {
    let n = 3usize;
    let length = 4u32;

    let mut body = Graph::new("cumsum_body");
    // carry must come first in NodeId order — declare it first.
    let carry = body.input("carry", Shape::new(&[n], DType::F64));
    let x_t = body.input("x_t", Shape::new(&[n], DType::F64));
    let next = body.binary(BinaryOp::Add, carry, x_t, Shape::new(&[n], DType::F64));
    body.set_outputs(vec![next]);

    let mut g = Graph::new("cumsum_outer");
    let init = g.input("init", Shape::new(&[n], DType::F64));
    let xs = g.input("xs", Shape::new(&[length as usize, n], DType::F64));
    let final_carry = g.scan_with_xs(init, &[xs], body, length);
    g.set_outputs(vec![final_carry]);

    let init_data = vec![0.0_f64; n];
    let xs_data: Vec<f64> = (0..length as usize * n).map(|i| (i + 1) as f64).collect(); // 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    let (sched, mut arena) = prepare_f64(&g, &[(init, &init_data), (xs, &xs_data)]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let got = read_arena_f64(&arena, final_carry, n);

    // Reference: column-wise sum of xs rows + init. With our row-major
    // layout, column j of xs is xs_data[j], xs_data[n+j], xs_data[2n+j], ...
    // (per-step row at offset t*n contributes element j to slot j).
    let mut want = init_data.clone();
    for t in 0..length as usize {
        for j in 0..n {
            want[j] += xs_data[t * n + j];
        }
    }
    for i in 0..n {
        assert!(
            (got[i] - want[i]).abs() < 1e-12,
            "got[{i}] = {} want {}",
            got[i],
            want[i]
        );
    }
}

/// Per-step xs scan composing with DenseSolve — Circulax-shaped:
///   carry_{t+1} = solve(M, carry_t + xs\[t\])
/// Models a Backward-Euler step driven by a time-varying source.
#[test]
fn scan_with_xs_be_with_drive() {
    let n = 3usize;
    let length = 4u32;
    let dt = 0.1_f64;

    let mut m_data = vec![0.0_f64; n * n];
    for i in 0..n {
        m_data[i * n + i] = 1.0 + dt * 2.0;
        if i > 0 {
            m_data[i * n + (i - 1)] = -dt;
        }
        if i + 1 < n {
            m_data[i * n + (i + 1)] = -dt;
        }
    }
    let m_bytes: Vec<u8> = m_data.iter().flat_map(|x| x.to_le_bytes()).collect();

    let mut body = Graph::new("be_drive_body");
    let carry = body.input("carry", Shape::new(&[n], DType::F64));
    let drive = body.input("drive", Shape::new(&[n], DType::F64));
    let m = body.add_node(
        Op::Constant { data: m_bytes },
        vec![],
        Shape::new(&[n, n], DType::F64),
    );
    let driven = body.binary(BinaryOp::Add, carry, drive, Shape::new(&[n], DType::F64));
    let next = body.dense_solve(m, driven, Shape::new(&[n], DType::F64));
    body.set_outputs(vec![next]);

    let mut g = Graph::new("be_drive_outer");
    let init = g.input("init", Shape::new(&[n], DType::F64));
    let xs = g.input("xs", Shape::new(&[length as usize, n], DType::F64));
    let final_carry = g.scan_with_xs(init, &[xs], body, length);
    g.set_outputs(vec![final_carry]);

    let init_data = vec![0.0_f64; n];
    // Drive the system with a unit pulse on element 0 at t=0,
    // zeros after.
    let mut xs_data = vec![0.0_f64; length as usize * n];
    xs_data[0] = 1.0;

    let (sched, mut arena) = prepare_f64(&g, &[(init, &init_data), (xs, &xs_data)]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let got = read_arena_f64(&arena, final_carry, n);

    // Reference: per-step in pure Rust.
    let mut x = init_data.clone();
    for t in 0..length as usize {
        for j in 0..n {
            x[j] += xs_data[t * n + j];
        }
        let mut a_copy = m_data.clone();
        crate::blas::dgesv(&mut a_copy, &mut x, n, 1);
    }
    for i in 0..n {
        assert!(
            (got[i] - x[i]).abs() < 1e-12,
            "got[{i}] = {} ref {}",
            got[i],
            x[i]
        );
    }
}

/// Reverse-mode AD through Op::BatchedDenseSolve. Forward solves
/// `[B, N, N] · x = [B, N]`; loss = sum of all entries. Closed
/// form: dB = (Aᵀ)⁻¹·1, dA = -(Aᵀ)⁻¹·1 ⊗ x. Verified analytically
/// per batch (each slice matches what the unbatched DenseSolve VJP
/// would compute).
#[test]
fn batched_dense_solve_gradient_matches_per_batch_analytic() {
    use rlx_opt::autodiff::grad_with_loss;
    let n = 3usize;
    let batch = 4usize;

    let mut g = Graph::new("bds_grad");
    let a = g.param("A", Shape::new(&[batch, n, n], DType::F64));
    let b = g.input("b", Shape::new(&[batch, n], DType::F64));
    let x = g.batched_dense_solve(a, b, Shape::new(&[batch, n], DType::F64));
    let loss = g.reduce(
        x,
        ReduceOp::Sum,
        vec![0, 1],
        false,
        Shape::new(&[1], DType::F64),
    );
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[a, b]);

    let find = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            let name = match &node.op {
                Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                _ => None,
            };
            if name == Some(want) {
                return node.id;
            }
        }
        panic!("no node named {want}");
    };
    let a_id = find(&bwd, "A");
    let b_id = find(&bwd, "b");
    let d_out_id = find(&bwd, "d_output");

    let mut rng = rlx_ir::Philox4x32::new(0x57e1_u64);
    let mut a_data = vec![0.0_f64; batch * n * n];
    let mut b_data = vec![0.0_f64; batch * n];
    for bi in 0..batch {
        for i in 0..n {
            for j in 0..n {
                a_data[bi * n * n + i * n + j] = rng.next_f32() as f64 * 0.1;
            }
            a_data[bi * n * n + i * n + i] += 1.0 + n as f64;
        }
        for i in 0..n {
            b_data[bi * n + i] = rng.next_f32() as f64;
        }
    }
    let d_seed = [1.0_f64];

    let (sched, mut arena) = prepare_f64(
        &bwd,
        &[(a_id, &a_data), (b_id, &b_data), (d_out_id, &d_seed)],
    );
    execute_thunks(&sched, arena.raw_buf_mut());
    let da_out = read_arena_f64(&arena, bwd.outputs[1], batch * n * n);
    let db_out = read_arena_f64(&arena, bwd.outputs[2], batch * n);

    // Reference: per-batch analytic solve. dB_i = (A_iᵀ)⁻¹ · 1,
    // dA_i = -dB_i ⊗ x_i.
    for bi in 0..batch {
        let a_slice: Vec<f64> = a_data[bi * n * n..(bi + 1) * n * n].to_vec();
        let mut b_slice: Vec<f64> = b_data[bi * n..(bi + 1) * n].to_vec();
        let mut a_copy = a_slice.clone();
        crate::blas::dgesv(&mut a_copy, &mut b_slice, n, 1);
        let x_ref = b_slice.clone();
        // dB: solve(A^T, ones)
        let mut at = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                at[i * n + j] = a_slice[j * n + i];
            }
        }
        let mut ones = vec![1.0_f64; n];
        crate::blas::dgesv(&mut at, &mut ones, n, 1);
        let db_ref = ones;
        for i in 0..n {
            let got = db_out[bi * n + i];
            assert!(
                (got - db_ref[i]).abs() < 1e-10,
                "batch {bi}, db[{i}]: got {got} ref {}",
                db_ref[i]
            );
        }
        // dA: -outer(db, x)
        for i in 0..n {
            for j in 0..n {
                let got = da_out[bi * n * n + i * n + j];
                let want = -db_ref[i] * x_ref[j];
                assert!(
                    (got - want).abs() < 1e-10,
                    "batch {bi}, dA[{i},{j}]: got {got} ref {want}"
                );
            }
        }
    }
}

/// AD knob: gradient through `scan_checkpointed` automatically
/// uses the recompute backward path. Compares dinit from a plain
/// scan against the same forward written with `scan_checkpointed`,
/// both run through `grad_with_loss`. They must match to f64.
#[test]
fn scan_checkpointed_grad_matches_plain_scan_grad() {
    use rlx_opt::autodiff::grad_with_loss;
    let n = 2usize;
    let length = 6u32;

    let make_body = || {
        let mut body = Graph::new("ck_body");
        let carry = body.input("carry", Shape::new(&[n], DType::F64));
        let scale_bytes: Vec<u8> = (0..n).flat_map(|_| 1.05_f64.to_le_bytes()).collect();
        let scale = body.add_node(
            Op::Constant { data: scale_bytes },
            vec![],
            Shape::new(&[n], DType::F64),
        );
        let next = body.binary(BinaryOp::Mul, carry, scale, Shape::new(&[n], DType::F64));
        body.set_outputs(vec![next]);
        body
    };

    // Plain scan path.
    let mut g_plain = Graph::new("ck_plain");
    let init_p = g_plain.input("init", Shape::new(&[n], DType::F64));
    let final_p = g_plain.scan(init_p, make_body(), length);
    let loss_p = g_plain.reduce(
        final_p,
        ReduceOp::Sum,
        vec![0],
        false,
        Shape::new(&[1], DType::F64),
    );
    g_plain.set_outputs(vec![loss_p]);
    let bwd_p = grad_with_loss(&g_plain, &[init_p]);

    // Checkpointed scan path with K=2 (length=6).
    let mut g_ck = Graph::new("ck_ckpt");
    let init_c = g_ck.input("init", Shape::new(&[n], DType::F64));
    let final_c = g_ck.scan_checkpointed(init_c, make_body(), length, 2);
    let loss_c = g_ck.reduce(
        final_c,
        ReduceOp::Sum,
        vec![0],
        false,
        Shape::new(&[1], DType::F64),
    );
    g_ck.set_outputs(vec![loss_c]);
    let bwd_c = grad_with_loss(&g_ck, &[init_c]);

    let find = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            let name = match &node.op {
                Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                _ => None,
            };
            if name == Some(want) {
                return node.id;
            }
        }
        panic!("no {want}");
    };

    let init_data = vec![0.5_f64, -0.5];
    let d_seed = [1.0_f64];

    let (s_p, mut a_p) = prepare_f64(
        &bwd_p,
        &[
            (find(&bwd_p, "init"), &init_data),
            (find(&bwd_p, "d_output"), &d_seed),
        ],
    );
    execute_thunks(&s_p, a_p.raw_buf_mut());
    let dinit_p = read_arena_f64(&a_p, bwd_p.outputs[1], n);

    let (s_c, mut a_c) = prepare_f64(
        &bwd_c,
        &[
            (find(&bwd_c, "init"), &init_data),
            (find(&bwd_c, "d_output"), &d_seed),
        ],
    );
    execute_thunks(&s_c, a_c.raw_buf_mut());
    let dinit_c = read_arena_f64(&a_c, bwd_c.outputs[1], n);

    for i in 0..n {
        assert!(
            (dinit_p[i] - dinit_c[i]).abs() < 1e-12,
            "dinit[{i}]: plain={} checkpointed={}",
            dinit_p[i],
            dinit_c[i]
        );
    }
}

/// Recursive checkpointing end-to-end: build a ScanBackward
/// configured with K=2 checkpoints (for length=4), and compare
/// dinit against the same backward graph with full trajectory
/// (K=0). Forward computes a cumulative-sum-style scan; loss = sum.
/// Both paths must agree to f64 precision.
#[test]
fn recursive_checkpointing_matches_full_trajectory() {
    let n = 2usize;
    let length = 4u32;

    // Body: carry + ones (deterministic, no xs)
    let build_body = || -> Graph {
        let mut body = Graph::new("rc_body");
        let carry = body.input("carry", Shape::new(&[n], DType::F64));
        let ones_bytes: Vec<u8> = (0..n).flat_map(|_| 1.0_f64.to_le_bytes()).collect();
        let ones = body.add_node(
            Op::Constant { data: ones_bytes },
            vec![],
            Shape::new(&[n], DType::F64),
        );
        let next = body.binary(BinaryOp::Add, carry, ones, Shape::new(&[n], DType::F64));
        body.set_outputs(vec![next]);
        body
    };

    // body_vjp: same body + d_output, output dcarry. body_vjp is
    // used by ScanBackward to walk the chain rule per step.
    let body_vjp_for = || -> Graph {
        use rlx_opt::autodiff::grad;
        let body = build_body();
        // grad(body, [carry_id]) → graph with dcarry as the output.
        let carry_id = body
            .nodes()
            .iter()
            .find(|n| matches!(n.op, Op::Input { .. }))
            .map(|n| n.id)
            .unwrap();
        grad(&body, &[carry_id])
    };

    // ── Forward (All-strategy): scan with full trajectory ──
    let mut g_full = Graph::new("rc_outer_full");
    let init_full = g_full.input("init", Shape::new(&[n], DType::F64));
    let traj_full_id = g_full.scan_trajectory(init_full, build_body(), length);
    // Hand-build a ScanBackward node that reads the full trajectory.
    let upstream_full = g_full.input("upstream", Shape::new(&[length as usize, n], DType::F64));
    let dinit_full_id = g_full.scan_backward(
        init_full,
        traj_full_id,
        upstream_full,
        &[],
        body_vjp_for(),
        length,
        true,
        Shape::new(&[n], DType::F64),
    );
    g_full.set_outputs(vec![dinit_full_id]);

    // ── Forward (Recursive-2): scan saves only K=2 rows ──
    // Build the trajectory shape [K, *carry] = [2, 2].
    let k = 2u32;
    let mut g_rec = Graph::new("rc_outer_rec");
    let init_rec = g_rec.input("init", Shape::new(&[n], DType::F64));
    let traj_rec_id = g_rec.add_node(
        Op::Scan {
            body: Box::new(build_body()),
            length,
            save_trajectory: true,
            num_bcast: 0,
            num_xs: 0,
            num_checkpoints: k,
        },
        vec![init_rec],
        Shape::new(&[k as usize, n], DType::F64),
    );
    // Same upstream shape as the full version (the upstream is per
    // *forward step*, length rows — independent of K).
    let upstream_rec = g_rec.input("upstream", Shape::new(&[length as usize, n], DType::F64));
    let dinit_rec_id = g_rec.add_node(
        Op::ScanBackward {
            body_vjp: Box::new(body_vjp_for()),
            length,
            save_trajectory: true,
            num_xs: 0,
            num_checkpoints: k,
            forward_body: Some(Box::new(build_body())),
        },
        vec![init_rec, traj_rec_id, upstream_rec],
        Shape::new(&[n], DType::F64),
    );
    g_rec.set_outputs(vec![dinit_rec_id]);

    // ── Run both, same inputs ──
    let init_data = vec![0.5_f64, -0.5];
    let upstream_data: Vec<f64> = (0..length as usize * n).map(|i| (i as f64) * 0.1).collect();

    let find = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            if let Op::Input { name } = &node.op
                && name == want
            {
                return node.id;
            }
        }
        panic!("no input {want}");
    };

    let (s_full, mut a_full) = prepare_f64(
        &g_full,
        &[
            (find(&g_full, "init"), &init_data),
            (find(&g_full, "upstream"), &upstream_data),
        ],
    );
    execute_thunks(&s_full, a_full.raw_buf_mut());
    let dinit_full = read_arena_f64(&a_full, g_full.outputs[0], n);

    let (s_rec, mut a_rec) = prepare_f64(
        &g_rec,
        &[
            (find(&g_rec, "init"), &init_data),
            (find(&g_rec, "upstream"), &upstream_data),
        ],
    );
    execute_thunks(&s_rec, a_rec.raw_buf_mut());
    let dinit_rec = read_arena_f64(&a_rec, g_rec.outputs[0], n);

    for i in 0..n {
        assert!(
            (dinit_full[i] - dinit_rec[i]).abs() < 1e-12,
            "i={i}: full={} rec={}",
            dinit_full[i],
            dinit_rec[i]
        );
    }
}

/// vmap-of-grad: gradient through Scan, vmap'd over init.
/// Forward (per row):
///   carry_{t+1} = carry_t + ones    (body adds a constant)
///   loss = sum(carry_length) = sum(init) + length·n
/// Closed form: dloss/dinit_i = 1 for every i. vmap over init at
/// batch=3 → dinit_batched is all-ones [3, n]. Cross-checks
/// against per-row grad_with_loss runs. Validates the vmap rule
/// for Op::ScanBackward.
#[test]
fn vmap_of_grad_scan_matches_per_row_runs() {
    use rlx_opt::autodiff::grad_with_loss;
    use rlx_opt::vmap::vmap;
    let n = 2usize;
    let length = 3u32;
    let batch = 3usize;

    let mut body = Graph::new("scan_grad_body");
    let carry = body.input("carry", Shape::new(&[n], DType::F64));
    let ones_bytes: Vec<u8> = (0..n).flat_map(|_| 1.0_f64.to_le_bytes()).collect();
    let ones = body.add_node(
        Op::Constant { data: ones_bytes },
        vec![],
        Shape::new(&[n], DType::F64),
    );
    let next = body.binary(BinaryOp::Add, carry, ones, Shape::new(&[n], DType::F64));
    body.set_outputs(vec![next]);

    let mut g = Graph::new("scan_grad_outer");
    let init = g.input("init", Shape::new(&[n], DType::F64));
    let final_x = g.scan(init, body, length);
    let loss = g.reduce(
        final_x,
        ReduceOp::Sum,
        vec![0],
        false,
        Shape::new(&[1], DType::F64),
    );
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[init]);
    let bg = vmap(&bwd, &["init"], batch);

    let find = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            let name = match &node.op {
                Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                _ => None,
            };
            if name == Some(want) {
                return node.id;
            }
        }
        panic!("no node named {want}");
    };
    let init_b = find(&bg, "init");
    let d_out_b = find(&bg, "d_output");

    let init_data: Vec<f64> = (0..batch * n).map(|i| (i as f64) * 0.5).collect();
    let d_seed = [1.0_f64];

    let (sched, mut arena) = prepare_f64(&bg, &[(init_b, &init_data), (d_out_b, &d_seed)]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let dinit_b = read_arena_f64(&arena, bg.outputs[1], batch * n);

    for i in 0..batch * n {
        assert!(
            (dinit_b[i] - 1.0).abs() < 1e-12,
            "dinit[{i}] = {} (expected 1.0)",
            dinit_b[i]
        );
    }

    // Cross-check vs per-row grad_with_loss.
    for bi in 0..batch {
        let row = &init_data[bi * n..(bi + 1) * n];
        let mut g2 = Graph::new("per_row_grad");
        let init2 = g2.input("init", Shape::new(&[n], DType::F64));
        let mut body2 = Graph::new("per_row_body");
        let c2 = body2.input("carry", Shape::new(&[n], DType::F64));
        let ones2_bytes: Vec<u8> = (0..n).flat_map(|_| 1.0_f64.to_le_bytes()).collect();
        let ones2 = body2.add_node(
            Op::Constant { data: ones2_bytes },
            vec![],
            Shape::new(&[n], DType::F64),
        );
        let next2 = body2.binary(BinaryOp::Add, c2, ones2, Shape::new(&[n], DType::F64));
        body2.set_outputs(vec![next2]);
        let final2 = g2.scan(init2, body2, length);
        let loss2 = g2.reduce(
            final2,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F64),
        );
        g2.set_outputs(vec![loss2]);
        let bwd2 = grad_with_loss(&g2, &[init2]);
        let init2_id = find(&bwd2, "init");
        let d_out2_id = find(&bwd2, "d_output");
        let (s2, mut a2) = prepare_f64(&bwd2, &[(init2_id, row), (d_out2_id, &d_seed)]);
        execute_thunks(&s2, a2.raw_buf_mut());
        let row_dinit = read_arena_f64(&a2, bwd2.outputs[1], n);
        for j in 0..n {
            let got = dinit_b[bi * n + j];
            let want = row_dinit[j];
            assert!(
                (got - want).abs() < 1e-12,
                "row {bi}, j {j}: vmap'd={got} per-row={want}"
            );
        }
    }
}

/// vmap of Op::Scan: batched cumulative-sum. Forward
///   carry_{t+1} = carry_t + xs\[t\]
///   final = init + sum(xs)
/// vmap over both init and xs at batch=3. Each batch row should
/// equal the scalar run of the same body+xs subset.
#[test]
fn vmap_scan_cumulative_sum_matches_scalar_runs() {
    use rlx_opt::vmap::vmap;
    let n = 2usize;
    let length = 4u32;
    let batch = 3usize;

    // Body: (carry, x_t) → carry + x_t
    let mut body = Graph::new("scan_body_cumsum");
    let carry = body.input("carry", Shape::new(&[n], DType::F64));
    let x_t = body.input("x_t", Shape::new(&[n], DType::F64));
    let next = body.binary(BinaryOp::Add, carry, x_t, Shape::new(&[n], DType::F64));
    body.set_outputs(vec![next]);

    let mut g = Graph::new("scan_outer_cumsum");
    let init = g.input("init", Shape::new(&[n], DType::F64));
    let xs = g.input("xs", Shape::new(&[length as usize, n], DType::F64));
    let final_carry = g.scan_with_xs(init, &[xs], body, length);
    g.set_outputs(vec![final_carry]);

    // vmap over both init and xs.
    let bg = vmap(&g, &["init", "xs"], batch);

    // Test data — distinct per-batch rows.
    let init_data: Vec<f64> = (0..batch * n).map(|i| (i + 1) as f64).collect();
    // xs has shape [B, length, n] after vmap (the outer's xs is
    // [length, n]; vmap lifts it to [B, length, n]).
    let xs_data: Vec<f64> = (0..batch * length as usize * n)
        .map(|i| 0.1 * (i as f64))
        .collect();

    let find = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            if let Op::Input { name } = &node.op
                && name == want
            {
                return node.id;
            }
        }
        panic!("no input {want}");
    };
    let init_b = find(&bg, "init");
    let xs_b = find(&bg, "xs");
    let (sched, mut arena) = prepare_f64(&bg, &[(init_b, &init_data), (xs_b, &xs_data)]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let batched_out = read_arena_f64(&arena, bg.outputs[0], batch * n);

    // Reference: per-batch scalar Scan.
    for bi in 0..batch {
        let init_slice = &init_data[bi * n..(bi + 1) * n];
        let mut x = init_slice.to_vec();
        for t in 0..length as usize {
            for j in 0..n {
                x[j] += xs_data[bi * length as usize * n + t * n + j];
            }
        }

        for i in 0..n {
            let got = batched_out[bi * n + i];
            assert!(
                (got - x[i]).abs() < 1e-12,
                "row {bi}, i {i}: got {got} ref {}",
                x[i]
            );
        }
    }
}

/// vmap of dense solve — Circulax-shaped batched parameter sweep.
/// Forward: x = solve(A, b). vmap over both A (batched [B,N,N])
/// and b (batched [B,N]). Run on CPU and compare each batch row
/// against an independent scalar dgesv.
#[test]
fn vmap_dense_solve_matches_scalar_runs() {
    use rlx_opt::vmap::vmap;
    let n = 3usize;
    let batch = 4usize;

    let mut g = Graph::new("solve_forward");
    let a = g.input("A", Shape::new(&[n, n], DType::F64));
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let x = g.dense_solve(a, b, Shape::new(&[n], DType::F64));
    g.set_outputs(vec![x]);

    // vmap both A and b across the batch.
    let bg = vmap(&g, &["A", "b"], batch);

    // Independent A and b per batch row.
    let mut rng = rlx_ir::Philox4x32::new(0xb47c_u64);
    let mut a_data = vec![0.0_f64; batch * n * n];
    let mut b_data = vec![0.0_f64; batch * n];
    for bi in 0..batch {
        // Diagonally dominant A — guaranteed non-singular.
        for i in 0..n {
            for j in 0..n {
                a_data[bi * n * n + i * n + j] = rng.next_f32() as f64 * 0.1;
            }
            a_data[bi * n * n + i * n + i] += 1.0 + n as f64;
        }
        for i in 0..n {
            b_data[bi * n + i] = rng.next_f32() as f64;
        }
    }

    let find = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            if let Op::Input { name } = &node.op
                && name == want
            {
                return node.id;
            }
        }
        panic!("no input named {want}");
    };
    let ba = find(&bg, "A");
    let bb = find(&bg, "b");
    let (sched, mut arena) = prepare_f64(&bg, &[(ba, &a_data), (bb, &b_data)]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let batched_x = read_arena_f64(&arena, bg.outputs[0], batch * n);

    // Reference: per-batch dgesv.
    for bi in 0..batch {
        let mut a_slice: Vec<f64> = a_data[bi * n * n..(bi + 1) * n * n].to_vec();
        let mut b_slice: Vec<f64> = b_data[bi * n..(bi + 1) * n].to_vec();
        crate::blas::dgesv(&mut a_slice, &mut b_slice, n, 1);
        for i in 0..n {
            let got = batched_x[bi * n + i];
            let want = b_slice[i];
            assert!(
                (got - want).abs() < 1e-12,
                "row {bi}, i {i}: got {got} want {want}"
            );
        }
    }
}

/// vmap end-to-end: build a graph that computes y = MatMul(x, w) + b
/// and reduces to a per-element loss. vmap over x with batch=4.
/// Run the batched graph and compare each output row against an
/// independent scalar run of the original graph. Validates the
/// structural lift + the runtime path for batched MatMul +
/// batched Binary + batched Reduce.
#[test]
fn vmap_matmul_add_reduce_matches_scalar_runs() {
    use rlx_opt::vmap::vmap;
    let n = 3usize;
    let batch = 4usize;

    // Forward graph: y = MatMul(reshape(x, [1,n]), w) + b ; loss = sum(y).
    let mut g = Graph::new("vmap_e2e_forward");
    let x = g.input("x", Shape::new(&[n], DType::F64));
    let w = g.input("w", Shape::new(&[n, n], DType::F64));
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let x_row = g.add_node(
        Op::Reshape {
            new_shape: vec![1, n as i64],
        },
        vec![x],
        Shape::new(&[1, n], DType::F64),
    );
    let mm = g.matmul(x_row, w, Shape::new(&[1, n], DType::F64));
    let mm_flat = g.add_node(
        Op::Reshape {
            new_shape: vec![n as i64],
        },
        vec![mm],
        Shape::new(&[n], DType::F64),
    );
    let yv = g.binary(BinaryOp::Add, mm_flat, b, Shape::new(&[n], DType::F64));
    let loss = g.reduce(
        yv,
        ReduceOp::Sum,
        vec![0],
        false,
        Shape::new(&[1], DType::F64),
    );
    g.set_outputs(vec![loss]);

    // Build the vmap'd version (batch over x; w and b shared).
    let bg = vmap(&g, &["x"], batch);

    // Test data — distinct rows so we can verify the per-row dispatch.
    let mut rng = rlx_ir::Philox4x32::new(0xc1c0_u64);
    let n_w = n * n;
    let w_data: Vec<f64> = (0..n_w).map(|_| rng.next_f32() as f64).collect();
    let b_data: Vec<f64> = (0..n).map(|_| rng.next_f32() as f64).collect();
    let mut x_data_batched: Vec<f64> = Vec::with_capacity(batch * n);
    for _ in 0..batch * n {
        x_data_batched.push(rng.next_f32() as f64);
    }

    // Run the batched graph.
    let find = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            if let Op::Input { name } = &node.op
                && name == want
            {
                return node.id;
            }
        }
        panic!("no input named {want}");
    };
    let bx = find(&bg, "x");
    let bw = find(&bg, "w");
    let bb = find(&bg, "b");
    let (sched, mut arena) =
        prepare_f64(&bg, &[(bx, &x_data_batched), (bw, &w_data), (bb, &b_data)]);
    execute_thunks(&sched, arena.raw_buf_mut());
    // Reduce::Sum on shifted axis 1 with keep_dim=false → output [B, 1]
    // (it preserves the leading batch axis but reduces what was [n] to [].
    // Since the original output was [1] f64 and the reduce was over
    // axis 0, after vmap the leading-axis-shifted reduce keeps the
    // leading 1 from the original output's [1] shape.)
    let batched_out = read_arena_f64(&arena, bg.outputs[0], batch);

    // Reference: run the original (un-batched) graph once per batch row.
    for bi in 0..batch {
        let xs_slice = &x_data_batched[bi * n..(bi + 1) * n];
        let mut g2 = Graph::new("scalar_run");
        let x2 = g2.input("x", Shape::new(&[n], DType::F64));
        let w2 = g2.input("w", Shape::new(&[n, n], DType::F64));
        let b2 = g2.input("b", Shape::new(&[n], DType::F64));
        let xr = g2.add_node(
            Op::Reshape {
                new_shape: vec![1, n as i64],
            },
            vec![x2],
            Shape::new(&[1, n], DType::F64),
        );
        let m = g2.matmul(xr, w2, Shape::new(&[1, n], DType::F64));
        let mf = g2.add_node(
            Op::Reshape {
                new_shape: vec![n as i64],
            },
            vec![m],
            Shape::new(&[n], DType::F64),
        );
        let yv2 = g2.binary(BinaryOp::Add, mf, b2, Shape::new(&[n], DType::F64));
        let l2 = g2.reduce(
            yv2,
            ReduceOp::Sum,
            vec![0],
            false,
            Shape::new(&[1], DType::F64),
        );
        g2.set_outputs(vec![l2]);
        let (s2, mut a2) = prepare_f64(&g2, &[(x2, xs_slice), (w2, &w_data), (b2, &b_data)]);
        execute_thunks(&s2, a2.raw_buf_mut());
        let scalar_out = read_arena_f64(&a2, l2, 1);
        assert!(
            (batched_out[bi] - scalar_out[0]).abs() < 1e-12,
            "row {bi}: batched={} scalar={}",
            batched_out[bi],
            scalar_out[0]
        );
    }
}

/// Full gradient through scan-with-xs: dinit AND dxs both checked
/// against finite differences. Forward
///   carry_{t+1} = solve(M, carry_t + xs\[t\])
///   loss        = sum(carry_length)
/// Verifies that grad_with_loss returns gradients w.r.t. both
/// `init` and `xs` and that dxs matches per-element FD.
#[test]
fn scan_with_xs_dxs_matches_fd() {
    use rlx_opt::autodiff::grad_with_loss;
    let n = 3usize;
    let length = 3u32;
    let dt = 0.1_f64;

    let mut m_data = vec![0.0_f64; n * n];
    for i in 0..n {
        m_data[i * n + i] = 1.0 + dt * 2.0;
        if i > 0 {
            m_data[i * n + (i - 1)] = -dt;
        }
        if i + 1 < n {
            m_data[i * n + (i + 1)] = -dt;
        }
    }
    let m_bytes: Vec<u8> = m_data.iter().flat_map(|x| x.to_le_bytes()).collect();

    let mut body = Graph::new("be_dxs_body");
    let carry = body.input("carry", Shape::new(&[n], DType::F64));
    let drive = body.input("drive", Shape::new(&[n], DType::F64));
    let m = body.add_node(
        Op::Constant { data: m_bytes },
        vec![],
        Shape::new(&[n, n], DType::F64),
    );
    let driven = body.binary(BinaryOp::Add, carry, drive, Shape::new(&[n], DType::F64));
    let next = body.dense_solve(m, driven, Shape::new(&[n], DType::F64));
    body.set_outputs(vec![next]);

    let mut g = Graph::new("be_dxs_outer");
    let init = g.input("init", Shape::new(&[n], DType::F64));
    let xs = g.input("xs", Shape::new(&[length as usize, n], DType::F64));
    let final_carry = g.scan_with_xs(init, &[xs], body, length);
    let loss = g.reduce(
        final_carry,
        ReduceOp::Sum,
        vec![0],
        false,
        Shape::new(&[1], DType::F64),
    );
    g.set_outputs(vec![loss]);

    // wrt = [init, xs] — get both gradients back.
    let bwd = grad_with_loss(&g, &[init, xs]);
    assert_eq!(bwd.outputs.len(), 3, "[loss, dinit, dxs]");

    let find_by_name = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            let name = match &node.op {
                Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                _ => None,
            };
            if name == Some(want) {
                return node.id;
            }
        }
        panic!("no node named {want:?}");
    };
    let init_bwd = find_by_name(&bwd, "init");
    let xs_bwd = find_by_name(&bwd, "xs");
    let d_out_bwd = find_by_name(&bwd, "d_output");

    let init_data = vec![0.5_f64, 0.0, -0.5];
    let xs_data: Vec<f64> = (0..length as usize * n)
        .map(|i| 0.1_f64 * ((i as f64) - 4.0))
        .collect();
    let d_seed = [1.0_f64];

    let (sched, mut arena) = prepare_f64(
        &bwd,
        &[
            (init_bwd, &init_data),
            (xs_bwd, &xs_data),
            (d_out_bwd, &d_seed),
        ],
    );
    execute_thunks(&sched, arena.raw_buf_mut());
    let dinit = read_arena_f64(&arena, bwd.outputs[1], n);
    let dxs = read_arena_f64(&arena, bwd.outputs[2], length as usize * n);

    let h = 1e-6;
    let loss_at = |x0: &[f64], xs_in: &[f64]| -> f64 {
        let mut acc = x0.to_vec();
        for t in 0..length as usize {
            for j in 0..n {
                acc[j] += xs_in[t * n + j];
            }
            let mut a_copy = m_data.clone();
            crate::blas::dgesv(&mut a_copy, &mut acc, n, 1);
        }
        acc.iter().sum()
    };

    // FD on dinit (sanity).
    for i in 0..n {
        let mut ip = init_data.to_vec();
        ip[i] += h;
        let mut im = init_data.to_vec();
        im[i] -= h;
        let fd = (loss_at(&ip, &xs_data) - loss_at(&im, &xs_data)) / (2.0 * h);
        assert!(
            (dinit[i] - fd).abs() < 1e-7,
            "FD dinit[{i}]: AD={} FD={}",
            dinit[i],
            fd
        );
    }

    // FD on every dxs entry — full per-step gradient check.
    for t in 0..length as usize {
        for j in 0..n {
            let idx = t * n + j;
            let mut xp = xs_data.clone();
            xp[idx] += h;
            let mut xm = xs_data.clone();
            xm[idx] -= h;
            let fd = (loss_at(&init_data, &xp) - loss_at(&init_data, &xm)) / (2.0 * h);
            assert!(
                (dxs[idx] - fd).abs() < 1e-7,
                "FD dxs[t={t},j={j}]: AD={} FD={}",
                dxs[idx],
                fd
            );
        }
    }
}

/// Gradient through a scan with per-step xs (Circulax-shaped).
/// Forward:
///   carry_{t+1} = solve(M, carry_t + xs\[t\])
///   loss = sum(carry_length)
/// dxs is out of MVP (asserted in the VJP rule's body_vjp `wrt`),
/// but `dinit` flows correctly through the body's reverse Jacobian
/// even with xs in the chain. Verify dinit against finite differences.
#[test]
fn scan_with_xs_gradient_dinit_matches_fd() {
    use rlx_opt::autodiff::grad_with_loss;
    let n = 3usize;
    let length = 3u32;
    let dt = 0.1_f64;

    let mut m_data = vec![0.0_f64; n * n];
    for i in 0..n {
        m_data[i * n + i] = 1.0 + dt * 2.0;
        if i > 0 {
            m_data[i * n + (i - 1)] = -dt;
        }
        if i + 1 < n {
            m_data[i * n + (i + 1)] = -dt;
        }
    }
    let m_bytes: Vec<u8> = m_data.iter().flat_map(|x| x.to_le_bytes()).collect();

    let mut body = Graph::new("be_xs_grad_body");
    let carry = body.input("carry", Shape::new(&[n], DType::F64));
    let drive = body.input("drive", Shape::new(&[n], DType::F64));
    let m = body.add_node(
        Op::Constant { data: m_bytes },
        vec![],
        Shape::new(&[n, n], DType::F64),
    );
    let driven = body.binary(BinaryOp::Add, carry, drive, Shape::new(&[n], DType::F64));
    let next = body.dense_solve(m, driven, Shape::new(&[n], DType::F64));
    body.set_outputs(vec![next]);

    let mut g = Graph::new("be_xs_grad_outer");
    let init = g.input("init", Shape::new(&[n], DType::F64));
    let xs = g.input("xs", Shape::new(&[length as usize, n], DType::F64));
    let final_carry = g.scan_with_xs(init, &[xs], body, length);
    let loss = g.reduce(
        final_carry,
        ReduceOp::Sum,
        vec![0],
        false,
        Shape::new(&[1], DType::F64),
    );
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[init]);

    let find_by_name = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            let name = match &node.op {
                Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                _ => None,
            };
            if name == Some(want) {
                return node.id;
            }
        }
        panic!("no node named {want:?}");
    };
    let init_bwd = find_by_name(&bwd, "init");
    let xs_bwd = find_by_name(&bwd, "xs");
    let d_out_bwd = find_by_name(&bwd, "d_output");

    let init_data = vec![0.5_f64, 0.0, -0.5];
    // Drive: small per-step pulse, varying per element.
    let xs_data: Vec<f64> = (0..length as usize * n)
        .map(|i| 0.1_f64 * ((i as f64) - 4.0))
        .collect();
    let d_seed = [1.0_f64];

    let (sched, mut arena) = prepare_f64(
        &bwd,
        &[
            (init_bwd, &init_data),
            (xs_bwd, &xs_data),
            (d_out_bwd, &d_seed),
        ],
    );
    execute_thunks(&sched, arena.raw_buf_mut());
    let dinit = read_arena_f64(&arena, bwd.outputs[1], n);

    let h = 1e-6;
    let loss_at = |x0: &[f64]| -> f64 {
        let mut acc = x0.to_vec();
        for t in 0..length as usize {
            for j in 0..n {
                acc[j] += xs_data[t * n + j];
            }
            let mut a_copy = m_data.clone();
            crate::blas::dgesv(&mut a_copy, &mut acc, n, 1);
        }
        acc.iter().sum()
    };
    for i in 0..n {
        let mut ip = init_data.to_vec();
        ip[i] += h;
        let mut im = init_data.to_vec();
        im[i] -= h;
        let fd = (loss_at(&ip) - loss_at(&im)) / (2.0 * h);
        assert!(
            (dinit[i] - fd).abs() < 1e-7,
            "FD dinit[{i}]: AD={} FD={}",
            dinit[i],
            fd
        );
    }
}

/// Gradient through a geometric-growth scan: forward
///   x_{t+1} = 1.1 · x_t,    x_0 = init
///   final   = x_length     = init · 1.1^length
///   loss    = sum(final)
/// closed-form ∂loss/∂init\[i\] = 1.1^length for every i.
/// Validates the VJP path: AD pre-pass rewrites save_trajectory=false
/// to true, autodiff emits Op::ScanBackward, executor walks t back.
#[test]
fn scan_gradient_geometric_matches_closed_form() {
    use rlx_opt::autodiff::grad_with_loss;
    let n = 3usize;
    let length = 5u32;

    let mut body = Graph::new("scan_grad_body");
    let x = body.input("carry", Shape::new(&[n], DType::F64));
    let scale_bytes: Vec<u8> = (0..n).flat_map(|_| 1.1_f64.to_le_bytes()).collect();
    let scale = body.add_node(
        Op::Constant { data: scale_bytes },
        vec![],
        Shape::new(&[n], DType::F64),
    );
    let next = body.binary(BinaryOp::Mul, x, scale, Shape::new(&[n], DType::F64));
    body.set_outputs(vec![next]);

    let mut g = Graph::new("scan_grad_outer");
    let init = g.input("init", Shape::new(&[n], DType::F64));
    let final_x = g.scan(init, body, length);
    let loss = g.reduce(
        final_x,
        ReduceOp::Sum,
        vec![0],
        false,
        Shape::new(&[1], DType::F64),
    );
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[init]);
    assert_eq!(bwd.outputs.len(), 2);

    let find_by_name = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            let name = match &node.op {
                Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                _ => None,
            };
            if name == Some(want) {
                return node.id;
            }
        }
        panic!("no node named {want:?}");
    };
    let init_bwd = find_by_name(&bwd, "init");
    let d_out_bwd = find_by_name(&bwd, "d_output");

    let init_data = vec![1.0_f64; n];
    let d_seed = [1.0_f64];
    let (sched, mut arena) = prepare_f64(&bwd, &[(init_bwd, &init_data), (d_out_bwd, &d_seed)]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let dinit = read_arena_f64(&arena, bwd.outputs[1], n);

    let want = 1.1_f64.powi(length as i32);
    for i in 0..n {
        assert!(
            (dinit[i] - want).abs() < 1e-12,
            "dinit[{i}] = {} want {}",
            dinit[i],
            want
        );
    }

    // Finite-difference cross-check on init[0].
    let h = 1e-6;
    let loss_at = |x: &[f64]| -> f64 {
        let mut acc = x.to_vec();
        for _ in 0..length {
            for v in acc.iter_mut() {
                *v *= 1.1;
            }
        }
        acc.iter().sum()
    };
    let mut ip = init_data.clone();
    ip[0] += h;
    let mut im = init_data.clone();
    im[0] -= h;
    let fd = (loss_at(&ip) - loss_at(&im)) / (2.0 * h);
    assert!(
        (dinit[0] - fd).abs() < 1e-7,
        "FD dinit[0]: AD={} FD={}",
        dinit[0],
        fd
    );
}

/// Gradient through Backward Euler scan composing with DenseSolve.
/// Asserts dinit matches finite-difference per coordinate.
#[test]
fn scan_gradient_backward_euler_matches_fd() {
    use rlx_opt::autodiff::grad_with_loss;
    let n = 4usize;
    let length = 3u32;
    let dt = 0.05_f64;

    let mut m_data = vec![0.0_f64; n * n];
    for i in 0..n {
        m_data[i * n + i] = 1.0 + dt * 2.0;
        if i > 0 {
            m_data[i * n + (i - 1)] = -dt;
        }
        if i + 1 < n {
            m_data[i * n + (i + 1)] = -dt;
        }
    }
    let m_bytes: Vec<u8> = m_data.iter().flat_map(|x| x.to_le_bytes()).collect();

    let mut body = Graph::new("be_grad_body");
    let x = body.input("x", Shape::new(&[n], DType::F64));
    let m = body.add_node(
        Op::Constant { data: m_bytes },
        vec![],
        Shape::new(&[n, n], DType::F64),
    );
    let next = body.dense_solve(m, x, Shape::new(&[n], DType::F64));
    body.set_outputs(vec![next]);

    let mut g = Graph::new("be_grad_outer");
    let init = g.input("x0", Shape::new(&[n], DType::F64));
    let final_x = g.scan(init, body, length);
    let loss = g.reduce(
        final_x,
        ReduceOp::Sum,
        vec![0],
        false,
        Shape::new(&[1], DType::F64),
    );
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[init]);

    let find_by_name = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            let name = match &node.op {
                Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                _ => None,
            };
            if name == Some(want) {
                return node.id;
            }
        }
        panic!("no node named {want:?}");
    };
    let init_bwd = find_by_name(&bwd, "x0");
    let d_out_bwd = find_by_name(&bwd, "d_output");

    let init_data: [f64; 4] = [0.0, 1.0, 0.0, 0.0];
    let d_seed = [1.0_f64];
    let (sched, mut arena) = prepare_f64(&bwd, &[(init_bwd, &init_data), (d_out_bwd, &d_seed)]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let dinit = read_arena_f64(&arena, bwd.outputs[1], n);

    let h = 1e-6;
    let loss_at = |x0: &[f64]| -> f64 {
        let mut acc = x0.to_vec();
        for _ in 0..length {
            let mut a_copy = m_data.clone();
            crate::blas::dgesv(&mut a_copy, &mut acc, n, 1);
        }
        acc.iter().sum()
    };
    for i in 0..n {
        let mut ip = init_data.to_vec();
        ip[i] += h;
        let mut im = init_data.to_vec();
        im[i] -= h;
        let fd = (loss_at(&ip) - loss_at(&im)) / (2.0 * h);
        assert!(
            (dinit[i] - fd).abs() < 1e-7,
            "FD dinit[{i}]: AD={} FD={}",
            dinit[i],
            fd
        );
    }
}

/// Trajectory-mode scan: same Backward Euler body, but record the
/// carry at every step. Output is `[length, n]` — row `t` is the
/// state after step `t+1`. Validates the SaveAt-style waveform
/// recording end-to-end, including that the last row equals what
/// the no-trajectory variant would have returned.
#[test]
fn scan_trajectory_backward_euler_records_waveform() {
    let n = 4usize;
    let length = 5u32;
    let dt = 0.05_f64;

    let mut m_data = vec![0.0_f64; n * n];
    for i in 0..n {
        m_data[i * n + i] = 1.0 + dt * 2.0;
        if i > 0 {
            m_data[i * n + (i - 1)] = -dt;
        }
        if i + 1 < n {
            m_data[i * n + (i + 1)] = -dt;
        }
    }
    let m_bytes: Vec<u8> = m_data.iter().flat_map(|x| x.to_le_bytes()).collect();

    let mut body = Graph::new("be_traj_body");
    let x = body.input("x", Shape::new(&[n], DType::F64));
    let m = body.add_node(
        Op::Constant { data: m_bytes },
        vec![],
        Shape::new(&[n, n], DType::F64),
    );
    let next = body.dense_solve(m, x, Shape::new(&[n], DType::F64));
    body.set_outputs(vec![next]);

    let mut g = Graph::new("be_traj_outer");
    let init = g.input("x0", Shape::new(&[n], DType::F64));
    let traj = g.scan_trajectory(init, body, length);
    g.set_outputs(vec![traj]);

    let init_data: [f64; 4] = [0.0, 1.0, 0.0, 0.0];
    let (sched, mut arena) = prepare_f64(&g, &[(init, &init_data)]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let got = read_arena_f64(&arena, traj, length as usize * n);

    // Reference: each step's solve, recorded.
    let mut want = Vec::<f64>::with_capacity(length as usize * n);
    let mut x_ref = init_data.to_vec();
    for _ in 0..length {
        let mut a_copy = m_data.clone();
        crate::blas::dgesv(&mut a_copy, &mut x_ref, n, 1);
        want.extend_from_slice(&x_ref);
    }
    for i in 0..length as usize * n {
        assert!(
            (got[i] - want[i]).abs() < 1e-12,
            "got[{i}] = {} ref {}",
            got[i],
            want[i]
        );
    }

    // Sanity: trajectory rows are monotone-decreasing in mass
    // (Backward Euler diffuses; boundary leak removes mass).
    for t in 1..length as usize {
        let prev: f64 = got[(t - 1) * n..t * n].iter().sum();
        let curr: f64 = got[t * n..(t + 1) * n].iter().sum();
        assert!(
            curr <= prev + 1e-15,
            "mass should decay: row {} sum {prev}, row {t} sum {curr}",
            t - 1
        );
    }

    // Last row of the trajectory equals what a non-trajectory
    // scan returns — verify by running the same forward through
    // the simpler API and comparing.
    let mut body2 = Graph::new("be_final_body");
    let x2 = body2.input("x", Shape::new(&[n], DType::F64));
    let m_bytes2: Vec<u8> = m_data.iter().flat_map(|x| x.to_le_bytes()).collect();
    let m2 = body2.add_node(
        Op::Constant { data: m_bytes2 },
        vec![],
        Shape::new(&[n, n], DType::F64),
    );
    let next2 = body2.dense_solve(m2, x2, Shape::new(&[n], DType::F64));
    body2.set_outputs(vec![next2]);

    let mut g2 = Graph::new("be_final_outer");
    let init2 = g2.input("x0", Shape::new(&[n], DType::F64));
    let final_x = g2.scan(init2, body2, length);
    g2.set_outputs(vec![final_x]);
    let (sched2, mut arena2) = prepare_f64(&g2, &[(init2, &init_data)]);
    execute_thunks(&sched2, arena2.raw_buf_mut());
    let final_got = read_arena_f64(&arena2, final_x, n);

    let last_row = &got[(length as usize - 1) * n..length as usize * n];
    for i in 0..n {
        assert!(
            (last_row[i] - final_got[i]).abs() < 1e-15,
            "last trajectory row[{i}] = {} vs final-scan = {}",
            last_row[i],
            final_got[i]
        );
    }
}

/// Op::Scan composing with Op::DenseSolve — the Circulax-shaped
/// pattern for Backward Euler.
/// Body: x_{t+1} = solve(I + dt·A, x_t).
/// 1-D heat-equation Laplacian A; analytic ground truth from
/// composing the same per-step solve in Rust.
#[test]
fn scan_backward_euler_heat_f64() {
    let n = 4usize;
    let length = 5u32;
    let dt = 0.05_f64;

    // Construct M = I + dt · L  where L is the Laplacian (-1, 2, -1).
    // M is constant across iterations; embed it in the body via Op::Constant.
    let mut m_data = vec![0.0_f64; n * n];
    for i in 0..n {
        m_data[i * n + i] = 1.0 + dt * 2.0;
        if i > 0 {
            m_data[i * n + (i - 1)] = -dt;
        }
        if i + 1 < n {
            m_data[i * n + (i + 1)] = -dt;
        }
    }
    let m_bytes: Vec<u8> = m_data.iter().flat_map(|x| x.to_le_bytes()).collect();

    let mut body = Graph::new("be_body");
    let x = body.input("x", Shape::new(&[n], DType::F64));
    let m = body.add_node(
        Op::Constant { data: m_bytes },
        vec![],
        Shape::new(&[n, n], DType::F64),
    );
    let next = body.dense_solve(m, x, Shape::new(&[n], DType::F64));
    body.set_outputs(vec![next]);

    let mut g = Graph::new("be_outer");
    let init = g.input("x0", Shape::new(&[n], DType::F64));
    let final_x = g.scan(init, body, length);
    g.set_outputs(vec![final_x]);

    // Initial: a sharp pulse at index 1.
    let init_data: [f64; 4] = [0.0, 1.0, 0.0, 0.0];
    let (sched, mut arena) = prepare_f64(&g, &[(init, &init_data)]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let got = read_arena_f64(&arena, final_x, n);

    // Reference: apply the same M-solve `length` times in pure Rust.
    let mut ref_x = init_data.to_vec();
    for _ in 0..length {
        let mut a_copy = m_data.clone();
        crate::blas::dgesv(&mut a_copy, &mut ref_x, n, 1);
    }
    for i in 0..n {
        assert!(
            (got[i] - ref_x[i]).abs() < 1e-12,
            "got[{i}] = {} ref {}",
            got[i],
            ref_x[i]
        );
    }
    // Sanity: pulse should diffuse, mass should be conserved-ish
    // (Backward Euler is mass-conserving for this stencil with
    // zero-flux boundaries — but our boundaries leak, so check
    // that mass strictly decreases instead).
    let mass: f64 = got.iter().sum();
    assert!(mass > 0.0 && mass < 1.0, "diffusion mass: {mass}");
}

/// Multi-RHS forward DenseSolve: X = solve(A, B) with B [N, K]
/// stays correct end-to-end. Verifies the executor/lowering and
/// the LAPACK column-major dance both honour `nrhs > 1`.
#[test]
fn dense_solve_f64_multi_rhs_forward() {
    let n = 3usize;
    let k = 2usize;
    let mut g = Graph::new("solve_multi_rhs");
    let a = g.input("A", Shape::new(&[n, n], DType::F64));
    let b = g.input("B", Shape::new(&[n, k], DType::F64));
    let x = g.dense_solve(a, b, Shape::new(&[n, k], DType::F64));
    g.set_outputs(vec![x]);

    let a_data = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0_f64];
    let b_data = [1.0, 4.0, 2.0, -1.0, 3.0, 2.0_f64];
    let (sched, mut arena) = prepare_f64(&g, &[(a, &a_data), (b, &b_data)]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let x_got = read_arena_f64(&arena, x, n * k);
    for c in 0..k {
        for i in 0..n {
            let mut acc = 0.0_f64;
            for j in 0..n {
                acc += a_data[i * n + j] * x_got[j * k + c];
            }
            let want = b_data[i * k + c];
            assert!(
                (acc - want).abs() < 1e-10,
                "col {c} row {i}: got {acc} want {want}"
            );
        }
    }
}

/// Multi-RHS reverse-mode VJP: dB = (Aᵀ)⁻¹·1, dA = -dB · Xᵀ.
/// Verified analytically + finite differences on dB[0,0].
#[test]
fn dense_solve_f64_multi_rhs_gradient() {
    use rlx_opt::autodiff::grad_with_loss;
    let n = 3usize;
    let k = 2usize;
    let mut g = Graph::new("solve_mrhs_grad");
    let a = g.param("A", Shape::new(&[n, n], DType::F64));
    let b = g.input("B", Shape::new(&[n, k], DType::F64));
    let x = g.dense_solve(a, b, Shape::new(&[n, k], DType::F64));
    let loss = g.reduce(
        x,
        ReduceOp::Sum,
        vec![0, 1],
        false,
        Shape::new(&[1], DType::F64),
    );
    g.set_outputs(vec![loss]);

    let bwd = grad_with_loss(&g, &[a, b]);
    let find_by_name = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            let name = match &node.op {
                Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                _ => None,
            };
            if name == Some(want) {
                return node.id;
            }
        }
        panic!("no node named {want:?}");
    };
    let a_bwd = find_by_name(&bwd, "A");
    let b_bwd = find_by_name(&bwd, "B");
    let d_out = find_by_name(&bwd, "d_output");

    let a_data = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0_f64];
    let b_data = [1.0, 4.0, 2.0, -1.0, 3.0, 2.0_f64];
    let d_seed = [1.0_f64];

    let (sched, mut arena) = prepare_f64(
        &bwd,
        &[(a_bwd, &a_data), (b_bwd, &b_data), (d_out, &d_seed)],
    );
    execute_thunks(&sched, arena.raw_buf_mut());
    let da_got = read_arena_f64(&arena, bwd.outputs[1], n * n);
    let db_got = read_arena_f64(&arena, bwd.outputs[2], n * k);

    // Reference.
    let mut x_ref = b_data;
    {
        let mut a_copy = a_data;
        crate::blas::dgesv(&mut a_copy, &mut x_ref, n, k);
    }
    let mut at = [0.0_f64; 9];
    for i in 0..n {
        for j in 0..n {
            at[i * n + j] = a_data[j * n + i];
        }
    }
    let mut ones_nk = vec![1.0_f64; n * k];
    crate::blas::dgesv(&mut at, &mut ones_nk, n, k);
    let db_ref = ones_nk;
    let mut da_ref = [0.0_f64; 9];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for c in 0..k {
                acc += db_ref[i * k + c] * x_ref[j * k + c];
            }
            da_ref[i * n + j] = -acc;
        }
    }
    for i in 0..n * k {
        assert!(
            (db_got[i] - db_ref[i]).abs() < 1e-10,
            "dB[{i}]: got {} want {}",
            db_got[i],
            db_ref[i]
        );
    }
    for i in 0..n * n {
        assert!(
            (da_got[i] - da_ref[i]).abs() < 1e-10,
            "dA[{i}]: got {} want {}",
            da_got[i],
            da_ref[i]
        );
    }

    // FD on dB[0,0].
    let h = 1e-6;
    let mut bp = b_data;
    bp[0] += h;
    let mut bm = b_data;
    bm[0] -= h;
    let xp = {
        let mut a_copy = a_data;
        crate::blas::dgesv(&mut a_copy, &mut bp, n, k);
        bp
    };
    let xm = {
        let mut a_copy = a_data;
        crate::blas::dgesv(&mut a_copy, &mut bm, n, k);
        bm
    };
    let lp: f64 = xp.iter().sum();
    let lm: f64 = xm.iter().sum();
    let fd = (lp - lm) / (2.0 * h);
    assert!(
        (db_got[0] - fd).abs() < 1e-7,
        "FD dB[0,0]: AD={} FD={}",
        db_got[0],
        fd
    );
}

/// Multi-RHS forward-mode JVP w.r.t. B. Closed form: t_X = solve(A, t_B).
#[test]
fn dense_solve_f64_multi_rhs_jvp() {
    use rlx_opt::autodiff_fwd::jvp;
    let n = 3usize;
    let k = 2usize;
    let mut g = Graph::new("solve_mrhs_jvp");
    let a = g.input("A", Shape::new(&[n, n], DType::F64));
    let b = g.input("B", Shape::new(&[n, k], DType::F64));
    let x = g.dense_solve(a, b, Shape::new(&[n, k], DType::F64));
    g.set_outputs(vec![x]);

    let jg = jvp(&g, &[b]);
    let find_by_name = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            let name = match &node.op {
                Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                _ => None,
            };
            if name == Some(want) {
                return node.id;
            }
        }
        panic!("no node named {want:?}");
    };
    let a_id = find_by_name(&jg, "A");
    let b_id = find_by_name(&jg, "B");
    let tb_id = find_by_name(&jg, "tangent_B");

    let a_data = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0_f64];
    let b_data = [1.0, 4.0, 2.0, -1.0, 3.0, 2.0_f64];
    let tb_data = [0.5, 0.0, -0.25, 1.0, 1.0, -0.5_f64];

    let (sched, mut arena) =
        prepare_f64(&jg, &[(a_id, &a_data), (b_id, &b_data), (tb_id, &tb_data)]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let tangent_x = read_arena_f64(&arena, jg.outputs[1], n * k);

    let mut a_copy = a_data;
    let mut tb_copy = tb_data;
    crate::blas::dgesv(&mut a_copy, &mut tb_copy, n, k);
    for i in 0..n * k {
        assert!(
            (tangent_x[i] - tb_copy[i]).abs() < 1e-10,
            "t_X[{i}]: AD={} ref={}",
            tangent_x[i],
            tb_copy[i]
        );
    }

    let h = 1e-6;
    let mut bp = b_data;
    let mut bm = b_data;
    for i in 0..n * k {
        bp[i] += h * tb_data[i];
        bm[i] -= h * tb_data[i];
    }
    let xp = {
        let mut a_copy = a_data;
        crate::blas::dgesv(&mut a_copy, &mut bp, n, k);
        bp
    };
    let xm = {
        let mut a_copy = a_data;
        crate::blas::dgesv(&mut a_copy, &mut bm, n, k);
        bm
    };
    for i in 0..n * k {
        let fd = (xp[i] - xm[i]) / (2.0 * h);
        assert!(
            (tangent_x[i] - fd).abs() < 1e-7,
            "FD t_X[{i}]: AD={} FD={}",
            tangent_x[i],
            fd
        );
    }
}

/// Forward-mode JVP through DenseSolve, end-to-end at f64.
///
/// Build forward x = solve(A, b), call `jvp(forward, [b])`,
/// compile + run, and check the tangent output matches the
/// closed form `t_x = solve(A, t_b)` plus a finite-difference
/// cross-check `(solve(A, b + h·t_b) − solve(A, b − h·t_b)) / 2h`.
#[test]
fn jvp_dense_solve_b_runs_and_matches_fd() {
    use rlx_opt::autodiff_fwd::jvp;
    let n = 3usize;

    // Forward.
    let mut g = Graph::new("jvp_b_e2e");
    let a = g.input("A", Shape::new(&[n, n], DType::F64));
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let x = g.dense_solve(a, b, Shape::new(&[n], DType::F64));
    g.set_outputs(vec![x]);

    // JVP graph perturbing b only.
    let jg = jvp(&g, &[b]);
    // The JVP graph holds a fresh "tangent_b" Input on top of A and b.
    let find_by_name = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            let name = match &node.op {
                Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                _ => None,
            };
            if name == Some(want) {
                return node.id;
            }
        }
        panic!("no node named {want:?}");
    };
    let a_id = find_by_name(&jg, "A");
    let b_id = find_by_name(&jg, "b");
    let tb_id = find_by_name(&jg, "tangent_b");

    let a_data: [f64; 9] = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    let b_data: [f64; 3] = [1.0, 2.0, 3.0];
    // Pick an arbitrary perturbation direction.
    let tb_data: [f64; 3] = [0.5, -0.25, 1.0];

    let (sched, mut arena) =
        prepare_f64(&jg, &[(a_id, &a_data), (b_id, &b_data), (tb_id, &tb_data)]);
    execute_thunks(&sched, arena.raw_buf_mut());

    // Outputs: [primal_x, tangent_x].
    let primal_x = read_arena_f64(&arena, jg.outputs[0], n);
    let tangent_x = read_arena_f64(&arena, jg.outputs[1], n);

    // Closed form: t_x = solve(A, t_b).
    let t_x_ref = {
        let mut a = a_data;
        let mut tb = tb_data;
        let info = crate::blas::dgesv(&mut a, &mut tb, n, 1);
        assert_eq!(info, 0);
        tb
    };
    for i in 0..n {
        assert!(
            (tangent_x[i] - t_x_ref[i]).abs() < 1e-10,
            "t_x[{i}]: got {} want {}",
            tangent_x[i],
            t_x_ref[i]
        );
    }

    // FD: x(b + h·tb) − x(b − h·tb)) / 2h
    let h = 1e-6;
    let mut bp = b_data;
    let mut bm = b_data;
    for i in 0..n {
        bp[i] += h * tb_data[i];
        bm[i] -= h * tb_data[i];
    }
    let xp = {
        let mut a = a_data;
        let info = crate::blas::dgesv(&mut a, &mut bp, n, 1);
        assert_eq!(info, 0);
        bp
    };
    let xm = {
        let mut a = a_data;
        let info = crate::blas::dgesv(&mut a, &mut bm, n, 1);
        assert_eq!(info, 0);
        bm
    };
    let fd: Vec<f64> = (0..n).map(|i| (xp[i] - xm[i]) / (2.0 * h)).collect();
    for i in 0..n {
        assert!(
            (tangent_x[i] - fd[i]).abs() < 1e-7,
            "FD mismatch t_x[{i}]: AD={} FD={}",
            tangent_x[i],
            fd[i]
        );
    }
    // Sanity: primal output is the actual solve.
    let primal_ref = {
        let mut a = a_data;
        let mut b = b_data;
        crate::blas::dgesv(&mut a, &mut b, n, 1);
        b
    };
    for i in 0..n {
        assert!((primal_x[i] - primal_ref[i]).abs() < 1e-10);
    }
}

/// Forward-mode JVP through DenseSolve perturbing A. The tangent
/// path includes the −t_A·x correction term.
/// `t_x = −solve(A, t_A · x)` should match a finite-difference
/// directional derivative of `solve(A, b)` w.r.t. A in the
/// `t_A` direction.
#[test]
fn jvp_dense_solve_a_runs_and_matches_fd() {
    use rlx_opt::autodiff_fwd::jvp;
    let n = 3usize;

    let mut g = Graph::new("jvp_a_e2e");
    let a = g.input("A", Shape::new(&[n, n], DType::F64));
    let b = g.input("b", Shape::new(&[n], DType::F64));
    let x = g.dense_solve(a, b, Shape::new(&[n], DType::F64));
    g.set_outputs(vec![x]);

    let jg = jvp(&g, &[a]);
    let find_by_name = |graph: &Graph, want: &str| -> NodeId {
        for node in graph.nodes() {
            let name = match &node.op {
                Op::Input { name } | Op::Param { name } => Some(name.as_str()),
                _ => None,
            };
            if name == Some(want) {
                return node.id;
            }
        }
        panic!("no node named {want:?}");
    };
    let a_id = find_by_name(&jg, "A");
    let b_id = find_by_name(&jg, "b");
    let ta_id = find_by_name(&jg, "tangent_A");

    let a_data: [f64; 9] = [2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    let b_data: [f64; 3] = [1.0, 2.0, 3.0];
    // Asymmetric perturbation direction for A.
    let ta_data: [f64; 9] = [0.10, -0.05, 0.02, 0.03, 0.20, -0.04, -0.01, 0.07, 0.15];

    let (sched, mut arena) =
        prepare_f64(&jg, &[(a_id, &a_data), (b_id, &b_data), (ta_id, &ta_data)]);
    execute_thunks(&sched, arena.raw_buf_mut());

    let tangent_x = read_arena_f64(&arena, jg.outputs[1], n);

    // Closed form: x = solve(A, b); t_x = −solve(A, t_A · x).
    let x_ref = {
        let mut a = a_data;
        let mut b = b_data;
        crate::blas::dgesv(&mut a, &mut b, n, 1);
        b
    };
    let mut prod = [0.0_f64; 3];
    for i in 0..n {
        for j in 0..n {
            prod[i] += ta_data[i * n + j] * x_ref[j];
        }
    }
    let t_x_ref = {
        let mut a = a_data;
        let mut p = prod;
        crate::blas::dgesv(&mut a, &mut p, n, 1);
        [-p[0], -p[1], -p[2]]
    };
    for i in 0..n {
        assert!(
            (tangent_x[i] - t_x_ref[i]).abs() < 1e-10,
            "closed-form t_x[{i}]: AD={} ref={}",
            tangent_x[i],
            t_x_ref[i]
        );
    }

    // FD: solve(A + h·t_A, b) and solve(A − h·t_A, b).
    let h = 1e-6;
    let mut ap = a_data;
    let mut am = a_data;
    for i in 0..n * n {
        ap[i] += h * ta_data[i];
        am[i] -= h * ta_data[i];
    }
    let xp = {
        let mut a = ap;
        let mut b = b_data;
        crate::blas::dgesv(&mut a, &mut b, n, 1);
        b
    };
    let xm = {
        let mut a = am;
        let mut b = b_data;
        crate::blas::dgesv(&mut a, &mut b, n, 1);
        b
    };
    for i in 0..n {
        let fd = (xp[i] - xm[i]) / (2.0 * h);
        assert!(
            (tangent_x[i] - fd).abs() < 1e-7,
            "FD t_x[{i}]: AD={} FD={}",
            tangent_x[i],
            fd
        );
    }
}

/// Real INT8 conv2d parity. Same setup as QMatMul: pre-quantize
/// f32 inputs to i8, run `Op::QConv2d`, compare against an
/// in-test reference loop that does the same i32 accumulation
/// and requantize math. Symmetric quant (zp=0) to keep the math
/// head-to-head.
#[test]
fn q_conv2d_matches_reference() {
    use rlx_ir::Philox4x32;
    // Small NCHW shape — enough to exercise stride/padding edges.
    let n = 1usize;
    let c_in = 2usize;
    let h = 5usize;
    let w_in = 5usize;
    let c_out = 3usize;
    let kh = 3usize;
    let kw = 3usize;
    let ph = 1usize;
    let pw = 1usize;
    let sh = 1usize;
    let sw = 1usize;
    let h_out = (h + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let x_scale = 0.04f32;
    let w_scale = 0.02f32;
    let out_scale = 0.5f32;
    let mult = x_scale * w_scale / out_scale;

    let mut rng = Philox4x32::new(2099);
    let mut xf = vec![0f32; n * c_in * h * w_in];
    rng.fill_normal(&mut xf);
    let mut wf = vec![0f32; c_out * c_in * kh * kw];
    rng.fill_normal(&mut wf);
    let xq: Vec<i8> = xf
        .iter()
        .map(|&v| ((v / x_scale).round() as i32).clamp(-128, 127) as i8)
        .collect();
    let wq: Vec<i8> = wf
        .iter()
        .map(|&v| ((v / w_scale).round() as i32).clamp(-128, 127) as i8)
        .collect();
    let bias: Vec<i32> = vec![0i32; c_out];

    let mut g = Graph::new("qconv");
    let xn = g.input("x", Shape::new(&[n, c_in, h, w_in], DType::I8));
    let wn = g.input("w", Shape::new(&[c_out, c_in, kh, kw], DType::I8));
    let bn = g.input("b", Shape::new(&[c_out], DType::I32));
    let out = g.q_conv2d(
        xn,
        wn,
        bn,
        vec![kh, kw],
        vec![sh, sw],
        vec![ph, pw],
        vec![1, 1],
        1,
        0,
        0,
        0,
        mult,
        Shape::new(&[n, c_out, h_out, w_out], DType::I8),
    );
    g.set_outputs(vec![out]);

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    // Capture offsets before borrowing the buf mutably (avoids
    // overlap between &mut and the &arena.byte_offset reads).
    let xn_off = arena.byte_offset(xn);
    let wn_off = arena.byte_offset(wn);
    let bn_off = arena.byte_offset(bn);
    let out_off = arena.byte_offset(out);
    let buf = arena.raw_buf_mut();
    unsafe {
        let p = buf.as_mut_ptr().add(xn_off) as *mut i8;
        for (i, &v) in xq.iter().enumerate() {
            *p.add(i) = v;
        }
        let p = buf.as_mut_ptr().add(wn_off) as *mut i8;
        for (i, &v) in wq.iter().enumerate() {
            *p.add(i) = v;
        }
        let p = buf.as_mut_ptr().add(bn_off) as *mut i32;
        for (i, &v) in bias.iter().enumerate() {
            *p.add(i) = v;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let out_q: Vec<i8> = unsafe {
        let p = arena.raw_buf().as_ptr().add(out_off) as *const i8;
        (0..n * c_out * h_out * w_out).map(|i| *p.add(i)).collect()
    };

    // Reference: scalar loop in NCHW with the same requantize.
    let mut out_ref = vec![0i8; n * c_out * h_out * w_out];
    for ni in 0..n {
        for co in 0..c_out {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    let mut acc: i32 = 0;
                    for ci in 0..c_in {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let hi = ho * sh + ki;
                                let wi = wo * sw + kj;
                                if hi < ph || wi < pw {
                                    continue;
                                }
                                let hi = hi - ph;
                                let wi = wi - pw;
                                if hi >= h || wi >= w_in {
                                    continue;
                                }
                                let xv = xq[((ni * c_in) + ci) * h * w_in + hi * w_in + wi] as i32;
                                let wv = wq[((co * c_in) + ci) * kh * kw + ki * kw + kj] as i32;
                                acc += xv * wv;
                            }
                        }
                    }
                    let r = (acc as f32 * mult).round() as i32;
                    let r = r.clamp(-128, 127) as i8;
                    out_ref[((ni * c_out) + co) * h_out * w_out + ho * w_out + wo] = r;
                }
            }
        }
    }

    for (i, (a, r)) in out_q.iter().zip(&out_ref).enumerate() {
        assert_eq!(a, r, "q_conv2d[{i}]: kernel {a} vs reference {r}");
    }
}

/// Real INT8 matmul parity: compare `Op::QMatMul` against the
/// fake-quant reference `Dequantize → MatMul → Quantize` that
/// would produce the same output if we round-tripped through
/// f32. Both should agree element-for-element (or within ±1 i8
/// step, since rounding in the requantize uses different code
/// paths). Symmetric quantization (zp=0) for both paths to keep
/// the math head-to-head.
#[test]
fn q_matmul_matches_fake_quant_reference() {
    use rlx_ir::Philox4x32;
    let m = 3usize;
    let k = 8usize;
    let n = 5usize;
    let mut rng = Philox4x32::new(2031);

    // Pick scales and quantize random f32 inputs to i8.
    let x_scale = 0.05f32;
    let w_scale = 0.03f32;
    let out_scale = 0.4f32;
    let mult = x_scale * w_scale / out_scale;
    let mut xf = vec![0f32; m * k];
    rng.fill_normal(&mut xf);
    let mut wf = vec![0f32; k * n];
    rng.fill_normal(&mut wf);
    let xq: Vec<i8> = xf
        .iter()
        .map(|&v| ((v / x_scale).round() as i32).clamp(-128, 127) as i8)
        .collect();
    let wq: Vec<i8> = wf
        .iter()
        .map(|&v| ((v / w_scale).round() as i32).clamp(-128, 127) as i8)
        .collect();
    let bias: Vec<i32> = vec![0i32; n];

    // ── Direct INT8 path ──
    let _f = DType::F32;
    let mut g_q = Graph::new("qmm_direct");
    let xn = g_q.input("x", Shape::new(&[m, k], DType::I8));
    let wn = g_q.input("w", Shape::new(&[k, n], DType::I8));
    let bn = g_q.input("b", Shape::new(&[n], DType::I32));
    let out = g_q.q_matmul(xn, wn, bn, 0, 0, 0, mult, Shape::new(&[m, n], DType::I8));
    g_q.set_outputs(vec![out]);
    let plan = rlx_opt::memory::plan_memory(&g_q);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g_q, &arena);

    // Fill inputs.
    let xn_off = arena.byte_offset(xn);
    let wn_off = arena.byte_offset(wn);
    let bn_off = arena.byte_offset(bn);
    let out_off = arena.byte_offset(out);
    let buf = arena.raw_buf_mut();
    unsafe {
        let p = buf.as_mut_ptr().add(xn_off) as *mut i8;
        for (i, &v) in xq.iter().enumerate() {
            *p.add(i) = v;
        }
        let p = buf.as_mut_ptr().add(wn_off) as *mut i8;
        for (i, &v) in wq.iter().enumerate() {
            *p.add(i) = v;
        }
        let p = buf.as_mut_ptr().add(bn_off) as *mut i32;
        for (i, &v) in bias.iter().enumerate() {
            *p.add(i) = v;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let out_q: Vec<i8> = unsafe {
        let p = arena.raw_buf().as_ptr().add(out_off) as *const i8;
        (0..m * n).map(|i| *p.add(i)).collect()
    };

    // ── Fake-quant reference: scalar emulation in plain Rust ──
    // Same arithmetic the kernel does, but in a verifier loop:
    //   acc = Σ (x[m,k]) · (w[k,n]),  // zps are 0
    //   out[m,n] = saturate_i8(round(acc · mult) + 0)
    let mut out_ref = vec![0i8; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc: i32 = 0;
            for ki in 0..k {
                acc += (xq[mi * k + ki] as i32) * (wq[ki * n + ni] as i32);
            }
            let r = (acc as f32 * mult).round() as i32;
            out_ref[mi * n + ni] = r.clamp(-128, 127) as i8;
        }
    }

    for (i, (a, r)) in out_q.iter().zip(&out_ref).enumerate() {
        assert_eq!(a, r, "q_matmul[{i}]: kernel {a} vs reference {r}");
    }
}

/// Quantize/Dequantize round-trip — quantize an f32 tensor, then
/// dequantize back, and confirm the result tracks the input
/// within the per-element scale (the inevitable rounding error).
/// Also pins the kernel's saturation behavior at the i8 limits.
#[test]
fn quantize_dequantize_round_trip() {
    use rlx_ir::Philox4x32;
    let len = 64;
    let mut rng = Philox4x32::new(2027);
    let mut x = vec![0f32; len];
    rng.fill_normal(&mut x);
    // Stretch a couple values past the +/- saturation cliff so
    // the saturate_i8 path is exercised.
    x[0] = 999.0;
    x[1] = -999.0;

    let scale = 0.05f32;
    let zp = 3i32;

    let f = DType::F32;
    let mut g = Graph::new("qdq");
    let xn = g.input("x", Shape::new(&[len], f));
    let q = g.quantize(xn, scale, zp);
    let dq = g.dequantize(q, scale, zp);
    g.set_outputs(vec![dq]);

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    let xn_off = arena.byte_offset(xn);
    let dq_off = arena.byte_offset(dq);
    let buf = arena.raw_buf_mut();
    unsafe {
        let p = buf.as_mut_ptr().add(xn_off) as *mut f32;
        for (i, &v) in x.iter().enumerate() {
            *p.add(i) = v;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let out: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(dq_off) as *const f32;
        (0..len).map(|i| *p.add(i)).collect()
    };

    // Saturated values at i=0,1 should clamp to ±127's dequant
    // range (= (±127 - zp) · scale).
    let sat_pos = (127 - zp) as f32 * scale;
    let sat_neg = (-128 - zp) as f32 * scale;
    assert!((out[0] - sat_pos).abs() < 1e-6, "+sat: {}", out[0]);
    assert!((out[1] - sat_neg).abs() < 1e-6, "-sat: {}", out[1]);

    // Everything else should round-trip within `scale` (one quant
    // step = the worst-case rounding error).
    for i in 2..len {
        assert!(
            (out[i] - x[i]).abs() <= scale + 1e-5,
            "qdq[{i}]: {} → {}, scale={scale}",
            x[i],
            out[i]
        );
    }
}

/// Per-channel quantize / dequantize: independent scale and zp
/// per slice along an axis. Verifies (a) each channel uses its
/// own scale (not a shared one), (b) saturation still respects
/// the i8 range, (c) channel data layout decomposition is
/// correct (no cross-channel leakage).
#[test]
fn quantize_per_channel_round_trip() {
    let c = 4usize;
    let inner = 5usize;
    // Different magnitudes per channel — proves the per-channel
    // scale is actually being read for each row.
    let mags = [0.01f32, 0.5, 5.0, 50.0];
    let mut x = vec![0f32; c * inner];
    for ci in 0..c {
        for ii in 0..inner {
            // Sweep through values that span [-max_abs, +max_abs]
            // for each channel, plus one value past the cliff to
            // trigger saturation.
            x[ci * inner + ii] = match ii {
                0 => -mags[ci],
                1 => 0.0,
                2 => mags[ci],
                3 => mags[ci] * 1000.0,  // saturates +
                _ => -mags[ci] * 1000.0, // saturates -
            };
        }
    }
    let scales: Vec<f32> = mags.iter().map(|&m| m / 127.0).collect();
    let zps: Vec<i32> = vec![0, 0, 0, 0];

    let f = DType::F32;
    let mut g = Graph::new("qdq_pc");
    let xn = g.input("x", Shape::new(&[c, inner], f));
    let q = g.quantize_per_channel(xn, 0, scales.clone(), zps.clone());
    let dq = g.dequantize_per_channel(q, 0, scales.clone(), zps);
    g.set_outputs(vec![dq]);

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    let xn_off = arena.byte_offset(xn);
    let dq_off = arena.byte_offset(dq);
    let buf = arena.raw_buf_mut();
    unsafe {
        let p = buf.as_mut_ptr().add(xn_off) as *mut f32;
        for (i, &v) in x.iter().enumerate() {
            *p.add(i) = v;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let out: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(dq_off) as *const f32;
        (0..c * inner).map(|i| *p.add(i)).collect()
    };

    for ci in 0..c {
        // Within-range entries (positions 0, 1, 2) must round-trip
        // within one quant step of *that channel's* scale.
        for ii in 0..3 {
            let idx = ci * inner + ii;
            assert!(
                (out[idx] - x[idx]).abs() <= scales[ci] + 1e-5,
                "ch {ci} idx {ii}: {} vs {}",
                x[idx],
                out[idx]
            );
        }
        // Saturated positions clamp to ±127 · scale[ci].
        let sat_pos = 127.0 * scales[ci];
        let sat_neg = -128.0 * scales[ci];
        assert!(
            (out[ci * inner + 3] - sat_pos).abs() < 1e-5,
            "ch {ci} +sat: {}",
            out[ci * inner + 3]
        );
        assert!(
            (out[ci * inner + 4] - sat_neg).abs() < 1e-5,
            "ch {ci} -sat: {}",
            out[ci * inner + 4]
        );
    }
}

/// `Op::ActivationBackward` parity for every supported kind.
/// Builds a single-op graph `dx = activation_backward(x, dy)` and
/// compares each `dx[i]` to the central-difference `(act(x+ε) -
/// act(x-ε)) / (2ε) · dy\[i\]`. Sweeps the closed-form covered by
/// the kernel.
#[test]
fn activation_backward_matches_numerical_per_kind() {
    use rlx_ir::Philox4x32;
    use rlx_ir::op::Activation;
    let mut rng = Philox4x32::new(91);
    let len = 32;
    // x sampled away from kink/branch points: shifted positive
    // (exp/sqrt/log domain) for the unary-positive activations;
    // wide range otherwise. Two parallel tests would be cleaner
    // but this is concise enough.
    let mut x_pos = vec![0f32; len];
    rng.fill_normal(&mut x_pos);
    for v in x_pos.iter_mut() {
        *v = v.abs() + 0.5;
    }
    let mut x_any = vec![0f32; len];
    rng.fill_normal(&mut x_any);
    let mut dy = vec![0f32; len];
    rng.fill_normal(&mut dy);

    for &(kind, x_data, eps, tol) in &[
        (Activation::Sigmoid, &x_any[..], 1e-3, 5e-3),
        (Activation::Tanh, &x_any[..], 1e-3, 5e-3),
        (Activation::Silu, &x_any[..], 1e-3, 5e-3),
        (Activation::Gelu, &x_any[..], 1e-3, 5e-3),
        (Activation::GeluApprox, &x_any[..], 1e-3, 5e-3),
        (Activation::Exp, &x_any[..], 1e-4, 5e-3),
        (Activation::Log, &x_pos[..], 1e-4, 5e-3),
        (Activation::Sqrt, &x_pos[..], 1e-4, 5e-3),
        (Activation::Rsqrt, &x_pos[..], 1e-4, 5e-3),
        (Activation::Neg, &x_any[..], 1e-3, 5e-4),
    ] {
        let f = DType::F32;
        let mut g = Graph::new("act_bw");
        let xn = g.input("x", Shape::new(&[len], f));
        let dyn_ = g.input("dy", Shape::new(&[len], f));
        let dx = g.activation_backward(kind, xn, dyn_);
        g.set_outputs(vec![dx]);

        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);

        let xn_off = arena.byte_offset(xn);
        let dyn_off = arena.byte_offset(dyn_);
        let dx_off = arena.byte_offset(dx);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(xn_off) as *mut f32;
            for (i, &v) in x_data.iter().enumerate() {
                *p.add(i) = v;
            }
            let p = buf.as_mut_ptr().add(dyn_off) as *mut f32;
            for (i, &v) in dy.iter().enumerate() {
                *p.add(i) = v;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        let analytical: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(dx_off) as *const f32;
            (0..len).map(|i| *p.add(i)).collect()
        };

        // Apply the forward activation manually; finite-difference
        // each element.
        let act_apply = |kind: Activation, x: f32| -> f32 {
            match kind {
                Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                Activation::Tanh => x.tanh(),
                Activation::Silu => x / (1.0 + (-x).exp()),
                Activation::Gelu => {
                    // Match the kernel's exact erf form.
                    const INV_SQRT2: f32 = 0.707_106_77;
                    0.5 * x * (1.0 + erf_f32(x * INV_SQRT2))
                }
                Activation::GeluApprox => {
                    const C: f32 = 0.797_884_6;
                    const A: f32 = 0.044_715;
                    let inner = C * (x + A * x * x * x);
                    0.5 * x * (1.0 + inner.tanh())
                }
                Activation::Exp => x.exp(),
                Activation::Log => x.ln(),
                Activation::Sqrt => x.sqrt(),
                Activation::Rsqrt => 1.0 / x.sqrt(),
                Activation::Neg => -x,
                Activation::Relu => x.max(0.0),
                Activation::Abs => x.abs(),
                Activation::Round => x.round(),
                Activation::Sin => x.sin(),
                Activation::Cos => x.cos(),
                Activation::Tan => x.tan(),
                Activation::Atan => x.atan(),
            }
        };
        for i in 0..len {
            let xv = x_data[i];
            let plus = act_apply(kind, xv + eps);
            let minus = act_apply(kind, xv - eps);
            let num = (plus - minus) / (2.0 * eps) * dy[i];
            assert!(
                (analytical[i] - num).abs() < tol,
                "{kind:?}[{i}]: analytical {} vs numerical {num}",
                analytical[i]
            );
        }
    }
}

/// Batched 3-D MatMul VJP — the transformer-attention shape
/// `[B, M, K] @ [B, K, N] = [B, M, N]`. Both gradients flow through
/// `Op::Transpose` with a perm that swaps the last two dims.
#[test]
fn matmul_3d_gradient_matches_numerical() {
    use rlx_ir::Philox4x32;
    let batch = 2usize;
    let m = 3usize;
    let k = 4usize;
    let n = 5usize;
    let mut rng = Philox4x32::new(101);
    let mut a_data = vec![0f32; batch * m * k];
    rng.fill_normal(&mut a_data);
    let mut b_data = vec![0f32; batch * k * n];
    rng.fill_normal(&mut b_data);

    let f = DType::F32;
    let mut fwd = Graph::new("matmul_3d");
    let an = fwd.input("a", Shape::new(&[batch, m, k], f));
    let bp = fwd.param("b", Shape::new(&[batch, k, n], f));
    let mm = fwd.matmul(an, bp, Shape::new(&[batch, m, n], f));
    let loss = fwd.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0, 1, 2],
            keep_dim: false,
        },
        vec![mm],
        Shape::from_dims(&[], f),
    );
    fwd.set_outputs(vec![loss]);

    let bwd_graph = rlx_opt::autodiff::grad_with_loss(&fwd, &[bp]);
    let d_out = bwd_graph
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
        .map(|n| n.id)
        .unwrap();

    let plan = rlx_opt::memory::plan_memory(&bwd_graph);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&bwd_graph, &arena);
    for &(id, data) in &[(an, &a_data), (bp, &b_data), (d_out, &vec![1.0f32])] {
        let off = arena.byte_offset(id);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(off) as *mut f32;
            for (i, &v) in data.iter().enumerate() {
                *p.add(i) = v;
            }
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let gb_id = bwd_graph.outputs[1];
    let g_b: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(arena.byte_offset(gb_id)) as *const f32;
        (0..batch * k * n).map(|i| *p.add(i)).collect()
    };

    // Numerical gradient: differentiate sum(a @ b) w.r.t. each b entry.
    let forward_loss = |b_vals: &[f32]| -> f32 {
        let mut out = vec![0f32; batch * m * n];
        for bi in 0..batch {
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc = 0f32;
                    for ki in 0..k {
                        acc += a_data[bi * m * k + mi * k + ki] * b_vals[bi * k * n + ki * n + ni];
                    }
                    out[bi * m * n + mi * n + ni] = acc;
                }
            }
        }
        out.iter().sum()
    };
    let eps = 1e-3f32;
    let mut bp_p = b_data.clone();
    let mut g_b_num = vec![0f32; b_data.len()];
    for i in 0..b_data.len() {
        let s = bp_p[i];
        bp_p[i] = s + eps;
        let lp = forward_loss(&bp_p);
        bp_p[i] = s - eps;
        let lm = forward_loss(&bp_p);
        bp_p[i] = s;
        g_b_num[i] = (lp - lm) / (2.0 * eps);
    }
    for (i, (a, n)) in g_b.iter().zip(&g_b_num).enumerate() {
        assert!(
            (a - n).abs() < 5e-3,
            "matmul_3d g_b[{i}]: analytical {a} vs numerical {n}"
        );
    }
}

/// Composed `Op::Softmax` VJP — the gradient is built from
/// `mul + reduce_sum + expand + sub + mul`, no dedicated
/// SoftmaxBackward kernel. Verifies the closed-form
/// `dx = y · (g - Σ y·g)` matches the FD gradient over a small
/// 2-D logits tensor.
#[test]
fn softmax_gradient_matches_numerical() {
    use rlx_ir::Philox4x32;
    let n = 3usize;
    let c = 5usize;
    let mut rng = Philox4x32::new(57);
    let mut x_data = vec![0f32; n * c];
    rng.fill_normal(&mut x_data);

    let f = DType::F32;
    let mut fwd = Graph::new("softmax_only");
    let xn = fwd.input("x", Shape::new(&[n, c], f));
    let sm = fwd.add_node(Op::Softmax { axis: -1 }, vec![xn], Shape::new(&[n, c], f));
    // Loss = sum(softmax · target) for some random fixed target —
    // any linear loss will do; sum-of-all is the simplest and gives
    // a uniform gradient flow into the softmax.
    let loss = fwd.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0, 1],
            keep_dim: false,
        },
        vec![sm],
        Shape::from_dims(&[], f),
    );
    fwd.set_outputs(vec![loss]);

    // `wrt = [xn]` — autodiff exposes the gradient w.r.t. the
    // input so we can compare it directly. The forward NodeId for
    // `xn` doubles as its bwd-graph mirror.
    let bwd_graph = rlx_opt::autodiff::grad_with_loss(&fwd, &[xn]);
    let d_out = bwd_graph
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
        .map(|n| n.id)
        .unwrap();

    let plan = rlx_opt::memory::plan_memory(&bwd_graph);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&bwd_graph, &arena);
    for &(id, data) in &[(xn, &x_data), (d_out, &vec![1.0f32])] {
        let off = arena.byte_offset(id);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(off) as *mut f32;
            for (i, &v) in data.iter().enumerate() {
                *p.add(i) = v;
            }
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let g_x_id = bwd_graph.outputs[1];
    let g_x: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(arena.byte_offset(g_x_id)) as *const f32;
        (0..n * c).map(|i| *p.add(i)).collect()
    };

    // Loss derivative: softmax sums to 1 per row → d/dx_i sum(softmax) = 0
    // analytically. So expect g_x ≈ 0 within FD precision. (This
    // doubles as a strong sanity check for the composition.)
    let forward_loss = |x: &[f32]| -> f32 {
        let mut total = 0f32;
        for ni in 0..n {
            let row = &x[ni * c..(ni + 1) * c];
            let m = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            let denom: f32 = row.iter().map(|&v| (v - m).exp()).sum();
            for &v in row {
                total += (v - m).exp() / denom;
            }
        }
        total
    };
    let eps = 1e-3f32;
    let mut p = x_data.clone();
    for i in 0..x_data.len() {
        let s = p[i];
        p[i] = s + eps;
        let lp = forward_loss(&p);
        p[i] = s - eps;
        let lm = forward_loss(&p);
        p[i] = s;
        let num = (lp - lm) / (2.0 * eps);
        assert!(
            (g_x[i] - num).abs() < 5e-3,
            "softmax g_x[{i}]: analytical {} vs numerical {num}",
            g_x[i]
        );
    }
}

/// LayerNorm VJP — three gradients in one pass:
///   d_x via `LayerNormBackwardInput`,
///   d_gamma via `LayerNormBackwardGamma`,
///   d_beta = `unbroadcast(upstream)` to gamma's shape.
#[test]
fn layer_norm_gradient_matches_numerical() {
    use rlx_ir::Philox4x32;
    let rows = 3usize;
    let h = 6usize;
    let mut rng = Philox4x32::new(1009);
    let mut x_data = vec![0f32; rows * h];
    rng.fill_normal(&mut x_data);
    let mut g_data = vec![0f32; h];
    rng.fill_normal(&mut g_data);
    for v in g_data.iter_mut() {
        *v = v.abs() + 0.5;
    }
    let mut b_data = vec![0f32; h];
    rng.fill_normal(&mut b_data);
    let eps = 1e-5f32;

    let f = DType::F32;
    let mut fwd = Graph::new("ln_only");
    let xn = fwd.input("x", Shape::new(&[rows, h], f));
    let gp = fwd.param("gamma", Shape::new(&[h], f));
    let bp = fwd.param("beta", Shape::new(&[h], f));
    let ln = fwd.add_node(
        Op::LayerNorm { axis: -1, eps },
        vec![xn, gp, bp],
        Shape::new(&[rows, h], f),
    );
    let loss = fwd.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0, 1],
            keep_dim: false,
        },
        vec![ln],
        Shape::from_dims(&[], f),
    );
    fwd.set_outputs(vec![loss]);

    let bwd_graph = rlx_opt::autodiff::grad_with_loss(&fwd, &[xn, gp, bp]);
    let d_out = bwd_graph
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
        .map(|n| n.id)
        .unwrap();

    let plan = rlx_opt::memory::plan_memory(&bwd_graph);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&bwd_graph, &arena);
    for &(id, data) in &[
        (xn, &x_data),
        (gp, &g_data),
        (bp, &b_data),
        (d_out, &vec![1.0f32]),
    ] {
        let off = arena.byte_offset(id);
        let buf = arena.raw_buf_mut();
        unsafe {
            let p = buf.as_mut_ptr().add(off) as *mut f32;
            for (i, &v) in data.iter().enumerate() {
                *p.add(i) = v;
            }
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let read = |id: NodeId, n: usize| -> Vec<f32> {
        let off = arena.byte_offset(id);
        unsafe {
            let p = arena.raw_buf().as_ptr().add(off) as *const f32;
            (0..n).map(|i| *p.add(i)).collect()
        }
    };
    let dx_a = read(bwd_graph.outputs[1], rows * h);
    let dg_a = read(bwd_graph.outputs[2], h);
    let db_a = read(bwd_graph.outputs[3], h);

    let forward_loss = |x: &[f32], g: &[f32], b: &[f32]| -> f32 {
        let mut total = 0f32;
        for r in 0..rows {
            let row = &x[r * h..(r + 1) * h];
            let mean = row.iter().sum::<f32>() / h as f32;
            let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for d in 0..h {
                total += ((row[d] - mean) * inv_std) * g[d] + b[d];
            }
        }
        total
    };
    let h_eps = 1e-3f32;

    let mut x_p = x_data.clone();
    for i in 0..x_p.len() {
        let s = x_p[i];
        x_p[i] = s + h_eps;
        let lp = forward_loss(&x_p, &g_data, &b_data);
        x_p[i] = s - h_eps;
        let lm = forward_loss(&x_p, &g_data, &b_data);
        x_p[i] = s;
        let num = (lp - lm) / (2.0 * h_eps);
        assert!(
            (dx_a[i] - num).abs() < 5e-3,
            "ln dx[{i}]: analytical {} vs numerical {num}",
            dx_a[i]
        );
    }
    let mut g_p = g_data.clone();
    for i in 0..g_p.len() {
        let s = g_p[i];
        g_p[i] = s + h_eps;
        let lp = forward_loss(&x_data, &g_p, &b_data);
        g_p[i] = s - h_eps;
        let lm = forward_loss(&x_data, &g_p, &b_data);
        g_p[i] = s;
        let num = (lp - lm) / (2.0 * h_eps);
        assert!(
            (dg_a[i] - num).abs() < 5e-3,
            "ln dg[{i}]: analytical {} vs numerical {num}",
            dg_a[i]
        );
    }
    let mut b_p = b_data.clone();
    for i in 0..b_p.len() {
        let s = b_p[i];
        b_p[i] = s + h_eps;
        let lp = forward_loss(&x_data, &g_data, &b_p);
        b_p[i] = s - h_eps;
        let lm = forward_loss(&x_data, &g_data, &b_p);
        b_p[i] = s;
        let num = (lp - lm) / (2.0 * h_eps);
        assert!(
            (db_a[i] - num).abs() < 5e-3,
            "ln db[{i}]: analytical {} vs numerical {num}",
            db_a[i]
        );
    }
}

/// Single dense layer + softmax-cross-entropy + mean reduce —
/// the simplest non-trivial training graph. Validates MatMul,
/// broadcast Add, SCE, Reduce(Mean) VJPs and the grad_with_loss
/// plumbing all at once.
#[test]
fn dense_sce_mean_gradient_matches_numerical() {
    use rlx_ir::Philox4x32;
    let bs = 4usize;
    let k_in = 3usize;
    let c = 5usize;
    let mut rng = Philox4x32::new(7);
    let mut x = vec![0f32; bs * k_in];
    rng.fill_normal(&mut x);
    let mut w_init = vec![0f32; k_in * c];
    rng.fill_normal(&mut w_init);
    let mut b_init = vec![0f32; c];
    rng.fill_normal(&mut b_init);
    let labels: Vec<f32> = (0..bs).map(|i| (i % c) as f32).collect();

    // ── Forward graph: loss = mean(sce(x @ w + b, labels)) ──
    let f = DType::F32;
    let mut fwd = Graph::new("dense_sce");
    let xn = fwd.input("x", Shape::new(&[bs, k_in], f));
    let lb = fwd.input("labels", Shape::new(&[bs], f));
    let wp = fwd.param("w", Shape::new(&[k_in, c], f));
    let bp = fwd.param("b", Shape::new(&[c], f));
    let mm = fwd.matmul(xn, wp, Shape::new(&[bs, c], f));
    let logits = fwd.binary(BinaryOp::Add, mm, bp, Shape::new(&[bs, c], f));
    let loss_per = fwd.softmax_cross_entropy_with_logits(logits, lb);
    let loss = fwd.add_node(
        Op::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keep_dim: false,
        },
        vec![loss_per],
        // Reduce sum of [bs] with axes=[0] keep_dim=false → scalar [].
        Shape::from_dims(&[], f),
    );
    // Use Sum + manual /bs scalar mul — also exercises BinaryOp::Mul VJP path
    // less aggressively than Mean would, and gives us a closed-form
    // reference for the loss we expect.
    // For simplicity though, switch to Mean which the tests should also cover.
    // (Re-using `loss` with Sum here for now; the mean factor cancels in
    // the gradient comparison since both analytical and numerical use the
    // same forward.)
    fwd.set_outputs(vec![loss]);

    // ── Backward graph ──
    let bwd_graph = rlx_opt::autodiff::grad_with_loss(&fwd, &[wp, bp]);
    // Outputs: [loss, grad_w, grad_b]. NodeIds for x/labels/w/b/loss
    // in bwd_graph match their fwd ids (the mirror keeps order).
    let d_out = bwd_graph
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
        .map(|n| n.id)
        .expect("d_output input");

    let (sched, mut arena) = prepare(
        &bwd_graph,
        &[
            (xn, &x),
            (lb, &labels),
            (wp, &w_init),
            (bp, &b_init),
            (d_out, &[1.0]),
        ],
    );
    execute_thunks(&sched, arena.raw_buf_mut());

    let outs = &bwd_graph.outputs;
    let loss_id = outs[0];
    let gw_id = outs[1];
    let gb_id = outs[2];
    let loss_actual = read_arena(&arena, loss_id, 1)[0];
    let gw_actual = read_arena(&arena, gw_id, k_in * c);
    let gb_actual = read_arena(&arena, gb_id, c);

    // ── Forward-only graph for finite differences ──
    // Re-use the same `fwd` graph; set up its own arena and rerun
    // for each perturbed parameter.
    let plan = rlx_opt::memory::plan_memory(&fwd);
    let mut fwd_arena = crate::arena::Arena::from_plan(plan);
    let fwd_sched = compile_thunks(&fwd, &fwd_arena);
    write_arena(&mut fwd_arena, xn, &x);
    write_arena(&mut fwd_arena, lb, &labels);

    let run_loss = |arena: &mut crate::arena::Arena, w: &[f32], b: &[f32]| -> f32 {
        write_arena(arena, wp, w);
        write_arena(arena, bp, b);
        execute_thunks(&fwd_sched, arena.raw_buf_mut());
        read_arena(arena, loss, 1)[0]
    };

    // Sanity: the loss reported by the bwd graph matches the
    // forward-only graph on the unperturbed inputs.
    let loss_check = run_loss(&mut fwd_arena, &w_init, &b_init);
    assert!(
        (loss_actual - loss_check).abs() < 1e-4,
        "loss mismatch: bwd graph {loss_actual} vs fwd-only {loss_check}"
    );

    let eps = 1e-3f32;
    let mut w_perturbed = w_init.clone();
    let mut gw_numerical = vec![0f32; w_init.len()];
    for i in 0..w_init.len() {
        let saved = w_perturbed[i];
        w_perturbed[i] = saved + eps;
        let lp = run_loss(&mut fwd_arena, &w_perturbed, &b_init);
        w_perturbed[i] = saved - eps;
        let lm = run_loss(&mut fwd_arena, &w_perturbed, &b_init);
        w_perturbed[i] = saved;
        gw_numerical[i] = (lp - lm) / (2.0 * eps);
    }
    for (i, (a, n)) in gw_actual.iter().zip(&gw_numerical).enumerate() {
        assert!(
            (a - n).abs() < 5e-3,
            "grad_w[{i}]: analytical {a} vs numerical {n}"
        );
    }

    let mut b_perturbed = b_init.clone();
    let mut gb_numerical = vec![0f32; b_init.len()];
    for i in 0..b_init.len() {
        let saved = b_perturbed[i];
        b_perturbed[i] = saved + eps;
        let lp = run_loss(&mut fwd_arena, &w_init, &b_perturbed);
        b_perturbed[i] = saved - eps;
        let lm = run_loss(&mut fwd_arena, &w_init, &b_perturbed);
        b_perturbed[i] = saved;
        gb_numerical[i] = (lp - lm) / (2.0 * eps);
    }
    for (i, (a, n)) in gb_actual.iter().zip(&gb_numerical).enumerate() {
        assert!(
            (a - n).abs() < 5e-3,
            "grad_b[{i}]: analytical {a} vs numerical {n}"
        );
    }
}

/// Reduce::Mean specifically — verifies the 1/N scaling in the VJP.
/// The same dense+SCE graph but with Mean instead of Sum on the loss.
#[test]
fn dense_sce_mean_reduce_gradient_matches_numerical() {
    use rlx_ir::Philox4x32;
    let bs = 3usize;
    let k_in = 2usize;
    let c = 4usize;
    let mut rng = Philox4x32::new(13);
    let mut x = vec![0f32; bs * k_in];
    rng.fill_normal(&mut x);
    let mut w_init = vec![0f32; k_in * c];
    rng.fill_normal(&mut w_init);
    let labels: Vec<f32> = (0..bs).map(|i| (i % c) as f32).collect();

    let f = DType::F32;
    let mut fwd = Graph::new("dense_sce_mean");
    let xn = fwd.input("x", Shape::new(&[bs, k_in], f));
    let lb = fwd.input("labels", Shape::new(&[bs], f));
    let wp = fwd.param("w", Shape::new(&[k_in, c], f));
    let mm = fwd.matmul(xn, wp, Shape::new(&[bs, c], f));
    let loss_per = fwd.softmax_cross_entropy_with_logits(mm, lb);
    let loss = fwd.add_node(
        Op::Reduce {
            op: ReduceOp::Mean,
            axes: vec![0],
            keep_dim: false,
        },
        vec![loss_per],
        Shape::from_dims(&[], f),
    );
    fwd.set_outputs(vec![loss]);

    let bwd_graph = rlx_opt::autodiff::grad_with_loss(&fwd, &[wp]);
    let d_out = bwd_graph
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
        .map(|n| n.id)
        .unwrap();

    let (sched, mut arena) = prepare(
        &bwd_graph,
        &[(xn, &x), (lb, &labels), (wp, &w_init), (d_out, &[1.0])],
    );
    execute_thunks(&sched, arena.raw_buf_mut());

    let outs = &bwd_graph.outputs;
    let loss_id = outs[0];
    let gw_id = outs[1];
    let _ = read_arena(&arena, loss_id, 1)[0];
    let gw_actual = read_arena(&arena, gw_id, k_in * c);

    let plan = rlx_opt::memory::plan_memory(&fwd);
    let mut fwd_arena = crate::arena::Arena::from_plan(plan);
    let fwd_sched = compile_thunks(&fwd, &fwd_arena);
    write_arena(&mut fwd_arena, xn, &x);
    write_arena(&mut fwd_arena, lb, &labels);

    let run_loss = |arena: &mut crate::arena::Arena, w: &[f32]| -> f32 {
        write_arena(arena, wp, w);
        execute_thunks(&fwd_sched, arena.raw_buf_mut());
        read_arena(arena, loss, 1)[0]
    };

    let eps = 1e-3f32;
    let mut wp_p = w_init.clone();
    let mut gw_num = vec![0f32; w_init.len()];
    for i in 0..w_init.len() {
        let s = wp_p[i];
        wp_p[i] = s + eps;
        let lp = run_loss(&mut fwd_arena, &wp_p);
        wp_p[i] = s - eps;
        let lm = run_loss(&mut fwd_arena, &wp_p);
        wp_p[i] = s;
        gw_num[i] = (lp - lm) / (2.0 * eps);
    }
    for (i, (a, n)) in gw_actual.iter().zip(&gw_num).enumerate() {
        assert!((a - n).abs() < 5e-3, "mean reduce grad_w[{i}]: {a} vs {n}");
    }
}
/// The full TinyConv-MNIST forward path (downsized) plumbed
/// through grad_with_loss. Validates that Conv, Pool(Max), ReLU,
/// Reshape, MatMul, Add (broadcast), SCE, Reduce(Mean) VJPs all
/// compose into a graph that produces correct gradients.
#[test]
fn tinyconv_full_gradient_matches_numerical() {
    use rlx_ir::Philox4x32;
    // Tiny shapes so finite differences finish in <1s.
    let n = 1usize;
    let c_in = 1usize;
    let h = 6usize;
    let w_in = 6usize;
    let c_mid = 2usize; // first conv output channels
    let kh = 3;
    let kw = 3;
    let h1 = h - kh + 1; // 4
    let w1 = w_in - kw + 1; // 4
    let h2 = h1 / 2;
    let w2 = w1 / 2; // 2 × 2 after 2× pool
    let flat = c_mid * h2 * w2; // 8
    let num_classes = 3usize;

    let mut rng = Philox4x32::new(31);
    let mut x = vec![0f32; n * c_in * h * w_in];
    rng.fill_normal(&mut x);
    let mut wc = vec![0f32; c_mid * c_in * kh * kw];
    rng.fill_normal(&mut wc);
    for v in wc.iter_mut() {
        *v *= 0.2;
    }
    // Shift conv-bias well away from the ReLU zero-boundary. Without
    // this, an ε-perturbation of bc[c] can flip the ReLU mask on a
    // pre-activation that happened to land near zero — making the
    // central-difference numerical gradient discontinuous and
    // diverge from the analytical (which assumes local smoothness).
    // +5.0 keeps every pre-activation positive for any random init
    // produced by Philox seed 31 with the wc/x scales used here, so
    // ReLU acts as an identity and finite differences are exact.
    let bc: Vec<f32> = (0..c_mid).map(|i| 5.0 + 0.1 * i as f32).collect();
    let mut wfc = vec![0f32; flat * num_classes];
    rng.fill_normal(&mut wfc);
    for v in wfc.iter_mut() {
        *v *= 0.5;
    }
    let mut bfc = vec![0f32; num_classes];
    rng.fill_normal(&mut bfc);
    let labels: Vec<f32> = vec![1.0]; // batch=1

    let f = DType::F32;
    let mut fwd = Graph::new("tinyconv");
    let xn = fwd.input("x", Shape::new(&[n, c_in, h, w_in], f));
    let lb = fwd.input("labels", Shape::new(&[n], f));
    let wcp = fwd.param("wc", Shape::new(&[c_mid, c_in, kh, kw], f));
    let bcp = fwd.param("bc", Shape::new(&[c_mid], f));
    let wfp = fwd.param("wfc", Shape::new(&[flat, num_classes], f));
    let bfp = fwd.param("bfc", Shape::new(&[num_classes], f));

    // conv: [n, c_in, h, w] → [n, c_mid, h1, w1]
    let conv = fwd.add_node(
        Op::Conv {
            kernel_size: vec![kh, kw],
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        },
        vec![xn, wcp],
        Shape::new(&[n, c_mid, h1, w1], f),
    );
    // Bias add: expand bc[c_mid] up to the full [n, c_mid, h1, w1]
    // shape so the Add becomes a plain element-wise op. Going through
    // an explicit Reshape→Expand instead of relying on the Add to
    // broadcast `[1, C, 1, 1]` → `[N, C, H, W]` works around a known
    // limitation of `rlx-cpu`'s `Op::Binary` lowering: it dispatches
    // on `out_len % rhs_len == 0` and treats `rhs` as a last-axis
    // bias, which produces `bc[0], bc[1], bc[0], bc[1], …` alternating
    // across all positions instead of channel-broadcasting. Going
    // through Expand (a real broadcast thunk) avoids that path
    // entirely. The autodiff still exercises `unbroadcast` because
    // `Op::Expand`'s VJP reduces over the broadcast axes.
    let bc_4d = fwd.add_node(
        Op::Reshape {
            new_shape: vec![1, c_mid as i64, 1, 1],
        },
        vec![bcp],
        Shape::new(&[1, c_mid, 1, 1], f),
    );
    let bc_expanded = fwd.add_node(
        Op::Expand {
            target_shape: vec![n as i64, c_mid as i64, h1 as i64, w1 as i64],
        },
        vec![bc_4d],
        Shape::new(&[n, c_mid, h1, w1], f),
    );
    let conv_b = fwd.binary(
        BinaryOp::Add,
        conv,
        bc_expanded,
        Shape::new(&[n, c_mid, h1, w1], f),
    );
    let relu = fwd.activation(Activation::Relu, conv_b, Shape::new(&[n, c_mid, h1, w1], f));
    let pool = fwd.add_node(
        Op::Pool {
            kind: ReduceOp::Max,
            kernel_size: vec![2, 2],
            stride: vec![2, 2],
            padding: vec![0, 0],
        },
        vec![relu],
        Shape::new(&[n, c_mid, h2, w2], f),
    );
    let flatn = fwd.add_node(
        Op::Reshape {
            new_shape: vec![n as i64, flat as i64],
        },
        vec![pool],
        Shape::new(&[n, flat], f),
    );
    let mm = fwd.matmul(flatn, wfp, Shape::new(&[n, num_classes], f));
    let logits = fwd.binary(BinaryOp::Add, mm, bfp, Shape::new(&[n, num_classes], f));
    let loss_per = fwd.softmax_cross_entropy_with_logits(logits, lb);
    let loss = fwd.add_node(
        Op::Reduce {
            op: ReduceOp::Mean,
            axes: vec![0],
            keep_dim: false,
        },
        vec![loss_per],
        Shape::from_dims(&[], f),
    );
    fwd.set_outputs(vec![loss]);

    let bwd_graph = rlx_opt::autodiff::grad_with_loss(&fwd, &[wcp, bcp, wfp, bfp]);
    let d_out = bwd_graph
        .nodes()
        .iter()
        .find(|n| matches!(&n.op, Op::Input { name } if name == "d_output"))
        .map(|n| n.id)
        .unwrap();

    let (sched, mut arena) = prepare(
        &bwd_graph,
        &[
            (xn, &x),
            (lb, &labels),
            (wcp, &wc),
            (bcp, &bc),
            (wfp, &wfc),
            (bfp, &bfc),
            (d_out, &[1.0]),
        ],
    );
    execute_thunks(&sched, arena.raw_buf_mut());

    let outs = bwd_graph.outputs.clone();
    let loss_id = outs[0];
    let g_wc_id = outs[1];
    let g_bc_id = outs[2];
    let g_wfc_id = outs[3];
    let g_bfc_id = outs[4];
    let loss_actual = read_arena(&arena, loss_id, 1)[0];
    let g_wc = read_arena(&arena, g_wc_id, wc.len());
    let g_bc = read_arena(&arena, g_bc_id, bc.len());
    let g_wfc = read_arena(&arena, g_wfc_id, wfc.len());
    let g_bfc = read_arena(&arena, g_bfc_id, bfc.len());

    // Forward-only arena for finite differences.
    let plan = rlx_opt::memory::plan_memory(&fwd);
    let mut fwd_arena = crate::arena::Arena::from_plan(plan);
    let fwd_sched = compile_thunks(&fwd, &fwd_arena);
    write_arena(&mut fwd_arena, xn, &x);
    write_arena(&mut fwd_arena, lb, &labels);

    // Closure variant: we need to set all four params each call so
    // perturbations to one don't leak between sweeps.
    let run_loss = |arena: &mut crate::arena::Arena,
                    wc: &[f32],
                    bc: &[f32],
                    wfc: &[f32],
                    bfc: &[f32]|
     -> f32 {
        write_arena(arena, wcp, wc);
        write_arena(arena, bcp, bc);
        write_arena(arena, wfp, wfc);
        write_arena(arena, bfp, bfc);
        execute_thunks(&fwd_sched, arena.raw_buf_mut());
        read_arena(arena, loss, 1)[0]
    };

    let loss_check = run_loss(&mut fwd_arena, &wc, &bc, &wfc, &bfc);
    assert!(
        (loss_actual - loss_check).abs() < 1e-4,
        "tinyconv loss mismatch: bwd {loss_actual} vs fwd {loss_check}"
    );

    let eps = 1e-3f32;
    let check_grad = |arena: &mut crate::arena::Arena,
                      name: &str,
                      analytical: &[f32],
                      mut perturb: Box<
        dyn FnMut(&mut [f32], usize, f32, &mut crate::arena::Arena) -> f32 + '_,
    >,
                      n: usize| {
        for i in 0..n {
            let lp = perturb(&mut analytical.to_vec(), i, eps, arena);
            let lm = perturb(&mut analytical.to_vec(), i, -eps, arena);
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (analytical[i] - num).abs() < 5e-3,
                "{name}[{i}]: analytical {} vs numerical {num}",
                analytical[i]
            );
        }
    };

    // Helper to perturb one param and run forward. Kept as a
    // reference for the explicit per-param sweep pattern below.
    #[allow(unused_macros)]
    macro_rules! sweep {
        ($name:expr, $base:expr, $analytical:expr, $set_param:ident) => {{
            let n = $base.len();
            for i in 0..n {
                let mut p = $base.clone();
                let s = p[i];
                p[i] = s + eps;
                let lp = {
                    let $set_param = &p;
                    run_loss(&mut fwd_arena, &wc, &bc, &wfc, &bfc).max(f32::NEG_INFINITY);
                    // Reset others, set the one being swept, run.
                    // (the macro receives one of the four params via $set_param)
                    let _ = $set_param;
                    // Fall through to the explicit per-param helper:
                    0.0_f32
                };
                let _ = lp;
            }
        }};
    }
    let _ = check_grad; // silence unused (sweep! macro is intentionally\n        // unused — kept as reference for the per-param sweep pattern below)

    // Per-param sweeps (explicit, not macro — clearer).
    for i in 0..wc.len() {
        let mut p = wc.clone();
        let s = p[i];
        p[i] = s + eps;
        let lp = run_loss(&mut fwd_arena, &p, &bc, &wfc, &bfc);
        p[i] = s - eps;
        let lm = run_loss(&mut fwd_arena, &p, &bc, &wfc, &bfc);
        let num = (lp - lm) / (2.0 * eps);
        assert!(
            (g_wc[i] - num).abs() < 5e-3,
            "g_wc[{i}]: {} vs {num}",
            g_wc[i]
        );
    }
    for i in 0..bc.len() {
        let mut p = bc.clone();
        let s = p[i];
        p[i] = s + eps;
        let lp = run_loss(&mut fwd_arena, &wc, &p, &wfc, &bfc);
        p[i] = s - eps;
        let lm = run_loss(&mut fwd_arena, &wc, &p, &wfc, &bfc);
        let num = (lp - lm) / (2.0 * eps);
        assert!(
            (g_bc[i] - num).abs() < 5e-3,
            "g_bc[{i}]: {} vs {num}",
            g_bc[i]
        );
    }
    for i in 0..wfc.len() {
        let mut p = wfc.clone();
        let s = p[i];
        p[i] = s + eps;
        let lp = run_loss(&mut fwd_arena, &wc, &bc, &p, &bfc);
        p[i] = s - eps;
        let lm = run_loss(&mut fwd_arena, &wc, &bc, &p, &bfc);
        let num = (lp - lm) / (2.0 * eps);
        assert!(
            (g_wfc[i] - num).abs() < 5e-3,
            "g_wfc[{i}]: {} vs {num}",
            g_wfc[i]
        );
    }
    for i in 0..bfc.len() {
        let mut p = bfc.clone();
        let s = p[i];
        p[i] = s + eps;
        let lp = run_loss(&mut fwd_arena, &wc, &bc, &wfc, &p);
        p[i] = s - eps;
        let lm = run_loss(&mut fwd_arena, &wc, &bc, &wfc, &p);
        let num = (lp - lm) / (2.0 * eps);
        assert!(
            (g_bfc[i] - num).abs() < 5e-3,
            "g_bfc[{i}]: {} vs {num}",
            g_bfc[i]
        );
    }
}

/// Negative case: a Narrow whose output has multiple consumers
/// must NOT be fused (we can't elide its write — something else
/// reads it).
#[test]
fn narrow_rope_skips_when_narrow_has_multiple_consumers() {
    let f = DType::F32;
    let mut g = Graph::new("nr_skip");
    let qkv = g.input("qkv", Shape::new(&[16, 8, 192], f));
    let cos = g.input("cos", Shape::new(&[16], f));
    let sin = g.input("sin", Shape::new(&[16], f));
    let q = g.narrow_(qkv, 2, 0, 64);
    let q_rope = g.rope(q, cos, sin, 16);
    // Second consumer of `q` blocks the fusion.
    let q_dup = g.activation(rlx_ir::op::Activation::Relu, q, Shape::new(&[16, 8, 64], f));
    g.set_outputs(vec![q_rope, q_dup]);

    let plan = rlx_opt::memory::plan_memory(&g);
    let arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);

    let narrow_count = sched
        .thunks
        .iter()
        .filter(|t| matches!(t, Thunk::Narrow { .. }))
        .count();
    assert!(
        narrow_count >= 1,
        "Narrow with multiple consumers must NOT be fused away"
    );
}

// ── Op::CustomFn (custom_vjp / custom_jvp) tests ──
//
// Validates: forward execution inlines fwd_body; VJP rule inlines
// vjp_body in place of recursing into fwd_body; JVP rule inlines
// jvp_body. Each test deliberately picks a body whose AD-via-tracing
// would yield a *different* gradient than the override, so we know
// the override actually fired.

/// Forward only: CustomFn wrapping `f(x) = x + c` (c=1 inside body)
/// without override AD bodies. Verifies the body is compiled,
/// constants in the body fill correctly, and the output lands at
/// the outer node's slot.
#[test]
fn custom_fn_forward_inlines_body() {
    let s = Shape::new(&[3], DType::F32);

    // Body: f(x) = x + 1
    let mut body = Graph::new("addone_body");
    let x = body.input("x", s.clone());
    let one_data: Vec<u8> = (0..3).flat_map(|_| 1.0_f32.to_le_bytes()).collect();
    let one = body.add_node(Op::Constant { data: one_data }, vec![], s.clone());
    let y = body.binary(BinaryOp::Add, x, one, s.clone());
    body.set_outputs(vec![y]);

    let mut g = Graph::new("custom_fn_outer");
    let xin = g.input("x_in", s.clone());
    let cf = g.custom_fn(vec![xin], body, None, None);
    g.set_outputs(vec![cf]);

    let xs = vec![10.0_f32, 20.0, 30.0];
    let (sched, mut arena) = prepare(&g, &[(xin, &xs)]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let got = read_arena(&arena, cf, 3);
    assert_eq!(got, vec![11.0, 21.0, 31.0]);
}

/// Locate an Op::Input or Op::Param by name in a graph.
fn find_named(graph: &Graph, want: &str) -> NodeId {
    for n in graph.nodes() {
        let name = match &n.op {
            Op::Input { name } | Op::Param { name } => Some(name.as_str()),
            _ => None,
        };
        if name == Some(want) {
            return n.id;
        }
    }
    panic!("no node named {want:?} in graph");
}

/// VJP override: f(x) = x but vjp_body returns 2 * d_output, so the
/// reported gradient should be 2 — different from the natural 1
/// you'd get by recursing into the identity body.
#[test]
fn custom_fn_vjp_overrides_natural_gradient() {
    use rlx_opt::autodiff::grad_with_loss;
    let s = Shape::new(&[1], DType::F32);

    let mut fwd = Graph::new("id_fwd");
    let x = fwd.input("x", s.clone());
    fwd.set_outputs(vec![x]);

    let mut vjp_g = Graph::new("id_vjp");
    let _x_p = vjp_g.input("x", s.clone());
    let _y_p = vjp_g.input("primal_output", s.clone());
    let dy = vjp_g.input("d_output", s.clone());
    let two_data: Vec<u8> = 2.0_f32.to_le_bytes().to_vec();
    let two = vjp_g.add_node(Op::Constant { data: two_data }, vec![], s.clone());
    let dx = vjp_g.binary(BinaryOp::Mul, dy, two, s.clone());
    vjp_g.set_outputs(vec![dx]);

    let mut g = Graph::new("outer");
    let xp = g.param("x", s.clone());
    let cf = g.custom_fn(vec![xp], fwd, Some(vjp_g), None);
    g.set_outputs(vec![cf]);

    let bwd = grad_with_loss(&g, &[xp]);
    assert_eq!(bwd.outputs.len(), 2, "expect [loss, dx]");

    let xb = find_named(&bwd, "x");
    let dout = find_named(&bwd, "d_output");
    let (sched, mut arena) = prepare(&bwd, &[(xb, &[7.0]), (dout, &[1.0])]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let loss = read_arena(&arena, bwd.outputs[0], 1);
    let dx_v = read_arena(&arena, bwd.outputs[1], 1);
    assert!((loss[0] - 7.0).abs() < 1e-6, "loss should be 7.0");
    assert!(
        (dx_v[0] - 2.0).abs() < 1e-6,
        "vjp override should yield dx=2.0, got {} (natural autodiff would give 1.0)",
        dx_v[0]
    );
}

/// VJP override: f(a, b) = a*b with vjp_body returning
/// (b * d_output, a * d_output). Validates routing of multiple
/// primals + d_output through the override; matches the natural
/// autodiff-of-Mul gradient (b, a).
#[test]
fn custom_fn_vjp_two_inputs_matches_mul_autodiff() {
    use rlx_opt::autodiff::grad_with_loss;
    let s = Shape::new(&[1], DType::F32);

    let mut fwd = Graph::new("mul_fwd");
    let a_f = fwd.input("a", s.clone());
    let b_f = fwd.input("b", s.clone());
    let y_f = fwd.binary(BinaryOp::Mul, a_f, b_f, s.clone());
    fwd.set_outputs(vec![y_f]);

    let mut vjp_g = Graph::new("mul_vjp");
    let a_v = vjp_g.input("a", s.clone());
    let b_v = vjp_g.input("b", s.clone());
    let _y_v = vjp_g.input("primal_output", s.clone());
    let dy_v = vjp_g.input("d_output", s.clone());
    let da = vjp_g.binary(BinaryOp::Mul, b_v, dy_v, s.clone());
    let db = vjp_g.binary(BinaryOp::Mul, a_v, dy_v, s.clone());
    vjp_g.set_outputs(vec![da, db]);

    let mut g = Graph::new("outer");
    let ap = g.param("a", s.clone());
    let bp = g.param("b", s.clone());
    let cf = g.custom_fn(vec![ap, bp], fwd, Some(vjp_g), None);
    g.set_outputs(vec![cf]);

    let bwd = grad_with_loss(&g, &[ap, bp]);
    assert_eq!(bwd.outputs.len(), 3, "expect [loss, da, db]");

    let ab = find_named(&bwd, "a");
    let bb = find_named(&bwd, "b");
    let dout = find_named(&bwd, "d_output");
    let (sched, mut arena) = prepare(&bwd, &[(ab, &[3.0]), (bb, &[5.0]), (dout, &[1.0])]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let loss = read_arena(&arena, bwd.outputs[0], 1);
    let da_v = read_arena(&arena, bwd.outputs[1], 1);
    let db_v = read_arena(&arena, bwd.outputs[2], 1);
    assert!((loss[0] - 15.0).abs() < 1e-5);
    assert!(
        (da_v[0] - 5.0).abs() < 1e-5,
        "da should be b=5.0, got {}",
        da_v[0]
    );
    assert!(
        (db_v[0] - 3.0).abs() < 1e-5,
        "db should be a=3.0, got {}",
        db_v[0]
    );
}

/// JVP override: f(x) = x but jvp_body returns 2 * tangent_0.
/// Forward-mode tangent should be 2x the seed (1.0) → 2.0.
#[test]
fn custom_fn_jvp_overrides_natural_tangent() {
    use rlx_opt::autodiff_fwd::jvp;
    let s = Shape::new(&[1], DType::F32);

    let mut fwd = Graph::new("id_fwd");
    let x = fwd.input("x", s.clone());
    fwd.set_outputs(vec![x]);

    let mut jvp_g = Graph::new("id_jvp");
    let _x_p = jvp_g.input("x", s.clone());
    let tx = jvp_g.input("tangent_0", s.clone());
    let two_data: Vec<u8> = 2.0_f32.to_le_bytes().to_vec();
    let two = jvp_g.add_node(Op::Constant { data: two_data }, vec![], s.clone());
    let ty = jvp_g.binary(BinaryOp::Mul, tx, two, s.clone());
    jvp_g.set_outputs(vec![ty]);

    let mut g = Graph::new("outer");
    let xin = g.input("x_in", s.clone());
    let cf = g.custom_fn(vec![xin], fwd, None, Some(jvp_g));
    g.set_outputs(vec![cf]);

    let fwd_g = jvp(&g, &[xin]);
    assert_eq!(fwd_g.outputs.len(), 2, "expect [primal_y, tangent_y]");

    let xb = find_named(&fwd_g, "x_in");
    let tan = find_named(&fwd_g, "tangent_x_in");
    let (sched, mut arena) = prepare(&fwd_g, &[(xb, &[7.0]), (tan, &[1.0])]);
    execute_thunks(&sched, arena.raw_buf_mut());
    let y = read_arena(&arena, fwd_g.outputs[0], 1);
    let ty_v = read_arena(&arena, fwd_g.outputs[1], 1);
    assert!((y[0] - 7.0).abs() < 1e-6);
    assert!(
        (ty_v[0] - 2.0).abs() < 1e-6,
        "jvp override should yield t_y=2.0 (natural autodiff would give 1.0), got {}",
        ty_v[0]
    );
}

/// IR-level basic test: `DType::C64` is wired through the dtype
/// table — `size_bytes() == 8`, `is_complex()` reports true, and
/// a `[2]`-shaped C64 buffer in the arena occupies the expected
/// 16 bytes.
#[test]
fn c64_dtype_storage_layout() {
    assert_eq!(
        DType::C64.size_bytes(),
        8,
        "C64 should be 8 bytes (f32 real + f32 imag)"
    );
    assert!(DType::C64.is_complex());
    assert!(!DType::C64.is_float());

    // A length-2 C64 buffer should have shape size_bytes = 16.
    let s = Shape::new(&[2], DType::C64);
    assert_eq!(s.size_bytes().unwrap(), 16);
}

// ── C64 element-wise binary kernel witnesses (2026-05-17) ──────
//
// Build a tiny graph: Input `a` + Input `b` (both C64 [2]),
// output = a OP b. Run through CompileResult and compare against
// the closed-form complex arithmetic on the four chosen pairs.

fn run_c64_binary(op: BinaryOp, a: &[(f32, f32)], b: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let n = a.len();
    let s = Shape::new(&[n], DType::C64);
    let mut g = Graph::new("c64_bin");
    let in_a = g.input("a", s.clone());
    let in_b = g.input("b", s.clone());
    let out = g.binary(op, in_a, in_b, s.clone());
    g.set_outputs(vec![out]);

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);

    let a_off = arena.byte_offset(in_a);
    let b_off = arena.byte_offset(in_b);
    let out_off = arena.byte_offset(out);
    // Interleave [re_0, im_0, re_1, im_1, ...] in the f32 buffer.
    let buf = arena.raw_buf_mut();
    unsafe {
        let pa = buf.as_mut_ptr().add(a_off) as *mut f32;
        let pb = buf.as_mut_ptr().add(b_off) as *mut f32;
        for (i, &(re, im)) in a.iter().enumerate() {
            *pa.add(2 * i) = re;
            *pa.add(2 * i + 1) = im;
        }
        for (i, &(re, im)) in b.iter().enumerate() {
            *pb.add(2 * i) = re;
            *pb.add(2 * i + 1) = im;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let raw_out: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(out_off) as *const f32;
        (0..(2 * n)).map(|i| *p.add(i)).collect()
    };
    (0..n)
        .map(|i| (raw_out[2 * i], raw_out[2 * i + 1]))
        .collect()
}

#[track_caller]
fn assert_close_c(got: (f32, f32), expected: (f32, f32), tol: f32, label: &str) {
    let dr = (got.0 - expected.0).abs();
    let di = (got.1 - expected.1).abs();
    assert!(
        dr < tol && di < tol,
        "[{label}] got ({:+.4}, {:+.4}), expected ({:+.4}, {:+.4})",
        got.0,
        got.1,
        expected.0,
        expected.1
    );
}

#[test]
fn c64_binary_add_matches_complex_arithmetic() {
    let a = [(1.0_f32, 2.0_f32), (3.0_f32, -1.0_f32)];
    let b = [(4.0_f32, -1.0_f32), (0.5_f32, 0.5_f32)];
    let out = run_c64_binary(BinaryOp::Add, &a, &b);
    assert_close_c(out[0], (5.0, 1.0), 1e-6, "add[0]");
    assert_close_c(out[1], (3.5, -0.5), 1e-6, "add[1]");
}

#[test]
fn c64_binary_sub_matches_complex_arithmetic() {
    let a = [(5.0_f32, 1.0_f32)];
    let b = [(2.0_f32, 3.0_f32)];
    let out = run_c64_binary(BinaryOp::Sub, &a, &b);
    assert_close_c(out[0], (3.0, -2.0), 1e-6, "sub");
}

#[test]
fn c64_binary_mul_matches_complex_arithmetic() {
    // (1 + 2i)(3 + 4i) = 3 + 4i + 6i + 8i² = -5 + 10i.
    let a = [(1.0_f32, 2.0_f32)];
    let b = [(3.0_f32, 4.0_f32)];
    let out = run_c64_binary(BinaryOp::Mul, &a, &b);
    assert_close_c(out[0], (-5.0, 10.0), 1e-5, "mul");
}

#[test]
fn c64_binary_div_matches_complex_arithmetic() {
    // (1 + 2i) / (3 + 4i) = ((1·3 + 2·4) + (2·3 − 1·4)i) / 25
    //                     = (11 + 2i) / 25
    //                     = 0.44 + 0.08i
    let a = [(1.0_f32, 2.0_f32)];
    let b = [(3.0_f32, 4.0_f32)];
    let out = run_c64_binary(BinaryOp::Div, &a, &b);
    assert_close_c(out[0], (0.44, 0.08), 1e-5, "div");
}

#[test]
fn c64_binary_mul_identity_one_is_no_op() {
    // (a + bi) · (1 + 0i) = a + bi.
    let a = [(3.5_f32, -1.25_f32), (-2.0_f32, 7.0_f32)];
    let b = [(1.0_f32, 0.0_f32), (1.0_f32, 0.0_f32)];
    let out = run_c64_binary(BinaryOp::Mul, &a, &b);
    assert_close_c(out[0], a[0], 1e-6, "mul·1[0]");
    assert_close_c(out[1], a[1], 1e-6, "mul·1[1]");
}

#[test]
fn c64_binary_mul_by_i_rotates_90_degrees() {
    // (a + bi) · i = (a + bi)(0 + i) = -b + ai. 90° CCW rotation.
    let a = [(1.0_f32, 0.0_f32)];
    let b = [(0.0_f32, 1.0_f32)];
    let out = run_c64_binary(BinaryOp::Mul, &a, &b);
    assert_close_c(out[0], (0.0, 1.0), 1e-6, "1·i");
}

#[test]
fn c64_binary_div_by_self_gives_unity() {
    let a = [(2.5_f32, -1.5_f32), (-0.7_f32, 4.2_f32)];
    let out = run_c64_binary(BinaryOp::Div, &a, &a);
    assert_close_c(out[0], (1.0, 0.0), 1e-5, "div_self[0]");
    assert_close_c(out[1], (1.0, 0.0), 1e-5, "div_self[1]");
}

#[test]
#[should_panic(expected = "C64: complex max/min/pow")]
fn c64_binary_max_is_rejected_at_lowering() {
    run_c64_binary(BinaryOp::Max, &[(1.0_f32, 2.0_f32)], &[(3.0_f32, 4.0_f32)]);
}

fn run_c64_activation(act: Activation, a: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let n = a.len();
    let s = Shape::new(&[n], DType::C64);
    let mut g = Graph::new("c64_act");
    let in_a = g.input("a", s.clone());
    let out = g.activation(act, in_a, s.clone());
    g.set_outputs(vec![out]);
    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    let a_off = arena.byte_offset(in_a);
    let out_off = arena.byte_offset(out);
    let buf = arena.raw_buf_mut();
    unsafe {
        let pa = buf.as_mut_ptr().add(a_off) as *mut f32;
        for (i, &(re, im)) in a.iter().enumerate() {
            *pa.add(2 * i) = re;
            *pa.add(2 * i + 1) = im;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let raw: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(out_off) as *const f32;
        (0..(2 * n)).map(|i| *p.add(i)).collect()
    };
    (0..n).map(|i| (raw[2 * i], raw[2 * i + 1])).collect()
}

#[test]
fn c64_activation_neg_negates_both_components() {
    let inp = [(3.5_f32, -1.25_f32), (-2.0_f32, 0.0_f32)];
    let out = run_c64_activation(Activation::Neg, &inp);
    assert_close_c(out[0], (-3.5, 1.25), 1e-6, "neg[0]");
    assert_close_c(out[1], (2.0, 0.0), 1e-6, "neg[1]");
}

#[test]
fn c64_activation_exp_matches_euler() {
    // exp(0 + i·π) = -1 + 0i.
    // exp(1 + 0i) = e ≈ 2.71828.
    let inp = [(0.0_f32, std::f32::consts::PI), (1.0_f32, 0.0_f32)];
    let out = run_c64_activation(Activation::Exp, &inp);
    assert_close_c(out[0], (-1.0, 0.0), 1e-5, "exp(iπ)");
    assert_close_c(out[1], (std::f32::consts::E, 0.0), 1e-5, "exp(1)");
}

#[test]
fn c64_activation_log_matches_principal_branch() {
    // log(1 + 0i) = 0.
    // log(0 + i) = log(1) + i·π/2 = 0 + i·π/2.
    // log(-1 + 0i) = 0 + i·π.
    let inp = [(1.0_f32, 0.0_f32), (0.0_f32, 1.0_f32), (-1.0_f32, 0.0_f32)];
    let out = run_c64_activation(Activation::Log, &inp);
    assert_close_c(out[0], (0.0, 0.0), 1e-5, "log(1)");
    assert_close_c(out[1], (0.0, std::f32::consts::FRAC_PI_2), 1e-5, "log(i)");
    assert_close_c(out[2], (0.0, std::f32::consts::PI), 1e-5, "log(-1)");
}

#[test]
fn c64_activation_sqrt_squared_recovers_input() {
    // For positive-real-part inputs, sqrt(z)² should equal z exactly
    // to f32 noise.
    let inp = [(4.0_f32, 0.0_f32), (3.0_f32, 4.0_f32)];
    let roots = run_c64_activation(Activation::Sqrt, &inp);
    // sqrt(4) = 2 + 0i; sqrt(3+4i) = 2 + i (since (2+i)² = 4+4i-1 = 3+4i).
    assert_close_c(roots[0], (2.0, 0.0), 1e-5, "sqrt(4)");
    assert_close_c(roots[1], (2.0, 1.0), 1e-5, "sqrt(3+4i)");
}

#[test]
#[should_panic(expected = "no natural complex extension")]
fn c64_activation_relu_is_rejected_at_lowering() {
    run_c64_activation(Activation::Relu, &[(1.0_f32, 2.0_f32)]);
}

// ── ComplexNormSq + Wirtinger backward witnesses ───────────────

/// Forward `|z|²`: returns `[n]` f32.
fn run_complex_norm_sq(z: &[(f32, f32)]) -> Vec<f32> {
    let n = z.len();
    let mut g = Graph::new("cns_fwd");
    let in_z = g.input("z", Shape::new(&[n], DType::C64));
    let out = g.complex_norm_sq(in_z);
    g.set_outputs(vec![out]);
    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    let z_off = arena.byte_offset(in_z);
    let out_off = arena.byte_offset(out);
    let buf = arena.raw_buf_mut();
    unsafe {
        let pz = buf.as_mut_ptr().add(z_off) as *mut f32;
        for (i, &(re, im)) in z.iter().enumerate() {
            *pz.add(2 * i) = re;
            *pz.add(2 * i + 1) = im;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    unsafe {
        let p = arena.raw_buf().as_ptr().add(out_off) as *const f32;
        (0..n).map(|i| *p.add(i)).collect()
    }
}

/// Backward: given z and upstream g, return dz = g·z element-wise (C64).
fn run_complex_norm_sq_bwd(z: &[(f32, f32)], g: &[f32]) -> Vec<(f32, f32)> {
    let n = z.len();
    let mut gr = Graph::new("cns_bwd");
    let in_z = gr.input("z", Shape::new(&[n], DType::C64));
    let in_g = gr.input("g", Shape::new(&[n], DType::F32));
    let out = gr.complex_norm_sq_backward(in_z, in_g);
    gr.set_outputs(vec![out]);
    let plan = rlx_opt::memory::plan_memory(&gr);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&gr, &arena);
    let z_off = arena.byte_offset(in_z);
    let g_off = arena.byte_offset(in_g);
    let out_off = arena.byte_offset(out);
    let buf = arena.raw_buf_mut();
    unsafe {
        let pz = buf.as_mut_ptr().add(z_off) as *mut f32;
        let pg = buf.as_mut_ptr().add(g_off) as *mut f32;
        for (i, &(re, im)) in z.iter().enumerate() {
            *pz.add(2 * i) = re;
            *pz.add(2 * i + 1) = im;
        }
        for (i, &v) in g.iter().enumerate() {
            *pg.add(i) = v;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    unsafe {
        let p = arena.raw_buf().as_ptr().add(out_off) as *const f32;
        (0..n).map(|i| (*p.add(2 * i), *p.add(2 * i + 1))).collect()
    }
}

#[test]
fn complex_norm_sq_matches_textbook() {
    // |3 + 4i|² = 9 + 16 = 25.
    // |1 + 0i|² = 1.
    // |0 + 0i|² = 0.
    let z = [(3.0_f32, 4.0_f32), (1.0_f32, 0.0_f32), (0.0_f32, 0.0_f32)];
    let out = run_complex_norm_sq(&z);
    assert!((out[0] - 25.0).abs() < 1e-5);
    assert!((out[1] - 1.0).abs() < 1e-6);
    assert!(out[2].abs() < 1e-6);
}

#[test]
fn complex_norm_sq_backward_matches_wirtinger_formula() {
    // Wirtinger: ∂|z|²/∂z̄ = z. With upstream g = 1, dz = z.
    let z = [(3.0_f32, 4.0_f32), (1.5_f32, -2.5_f32)];
    let g = [1.0_f32, 1.0_f32];
    let dz = run_complex_norm_sq_bwd(&z, &g);
    assert_close_c(dz[0], z[0], 1e-6, "dz[0] = g·z[0]");
    assert_close_c(dz[1], z[1], 1e-6, "dz[1] = g·z[1]");
}

#[test]
fn complex_norm_sq_backward_scales_with_upstream() {
    // With upstream g[i] ≠ 1: dz[i] = g[i]·z[i].
    let z = [(2.0_f32, 1.0_f32), (-1.0_f32, 3.0_f32)];
    let g = [0.5_f32, -2.0_f32];
    let dz = run_complex_norm_sq_bwd(&z, &g);
    assert_close_c(dz[0], (1.0, 0.5), 1e-6, "g=0.5 · (2,1)");
    assert_close_c(dz[1], (2.0, -6.0), 1e-6, "g=-2 · (-1,3)");
}

/// Multi-output Op::CustomFn via the concat-with-Narrow design
/// (rlx-ir::Graph::custom_fn_multi). Build a custom_fn whose
/// fwd_body returns two outputs (x², 2x), then materialize each
/// via the MultiOutputHandle and verify both numerically.
#[test]
fn custom_fn_multi_extracts_each_subgraph_output() {
    use rlx_ir::ops::special::MultiOutputHandle;

    let _ = MultiOutputHandle {
        source: NodeId(0),
        sub_shapes: vec![],
        offsets: vec![],
    }; // import sanity

    // Inner body: input x [3] f32, outputs (x², 2x) both [3] f32.
    let mut body = Graph::new("multi_body");
    let s3 = Shape::new(&[3], DType::F32);
    let x = body.input("x", s3.clone());
    let x_sq = body.binary(BinaryOp::Mul, x, x, s3.clone());
    let two = body.add_node(
        Op::Constant {
            data: vec![
                2.0_f32.to_le_bytes(),
                2.0_f32.to_le_bytes(),
                2.0_f32.to_le_bytes(),
            ]
            .into_iter()
            .flatten()
            .collect(),
        },
        vec![],
        s3.clone(),
    );
    let two_x = body.binary(BinaryOp::Mul, two, x, s3.clone());
    body.set_outputs(vec![x_sq, two_x]);

    // Outer graph: feed in_x → custom_fn_multi → handle.output(0/1).
    let mut outer = Graph::new("multi_outer");
    let in_x = outer.input("xin", s3.clone());
    let handle = outer.custom_fn_multi(vec![in_x], body);
    assert_eq!(handle.n_outputs(), 2);
    let out0 = handle.output(&mut outer, 0); // x²
    let out1 = handle.output(&mut outer, 1); // 2x
    outer.set_outputs(vec![out0, out1]);

    let plan = rlx_opt::memory::plan_memory(&outer);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&outer, &arena);
    let xin_off = arena.byte_offset(in_x);
    let out0_off = arena.byte_offset(out0);
    let out1_off = arena.byte_offset(out1);
    let xs = [1.0_f32, 2.0, 3.0];
    unsafe {
        let p = arena.raw_buf_mut().as_mut_ptr().add(xin_off) as *mut f32;
        for (i, &v) in xs.iter().enumerate() {
            *p.add(i) = v;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let out0_v: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(out0_off) as *const f32;
        (0..3).map(|i| *p.add(i)).collect()
    };
    let out1_v: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(out1_off) as *const f32;
        (0..3).map(|i| *p.add(i)).collect()
    };
    // x² = [1, 4, 9]; 2x = [2, 4, 6].
    for i in 0..3 {
        assert!(
            (out0_v[i] - xs[i] * xs[i]).abs() < 1e-5,
            "out0[{i}] = {} != x² = {}",
            out0_v[i],
            xs[i] * xs[i]
        );
        assert!(
            (out1_v[i] - 2.0 * xs[i]).abs() < 1e-5,
            "out1[{i}] = {} != 2x = {}",
            out1_v[i],
            2.0 * xs[i]
        );
    }
}

#[test]
fn complex_norm_sq_gradient_matches_finite_difference() {
    // Numerical sanity: perturb z[0].re by ε, observe Δ|z|² ≈ 2·re·ε.
    let z = [(3.0_f32, 4.0_f32)];
    let eps = 1e-3_f32;
    let v0 = run_complex_norm_sq(&z)[0];
    let z_pert = [(3.0_f32 + eps, 4.0_f32)];
    let v1 = run_complex_norm_sq(&z_pert)[0];
    let fd_re = (v1 - v0) / eps;
    let analytic_re = 2.0 * z[0].0;
    assert!((fd_re - analytic_re).abs() < 1e-2);

    // ∂/∂im at z = (3, 4) is 2·im = 8.
    let z_pert_im = [(3.0_f32, 4.0_f32 + eps)];
    let v2 = run_complex_norm_sq(&z_pert_im)[0];
    let fd_im = (v2 - v0) / eps;
    let analytic_im = 2.0 * z[0].1;
    assert!((fd_im - analytic_im).abs() < 1e-2);

    // Compare with the Wirtinger backward at upstream g = 1.
    // Wirtinger ∂/∂z̄ = z gives dz = (re, im). The "real
    // gradient" wrt (re, im) is 2·(re, im), i.e. 2·dz = (2·re,
    // 2·im) — that's the factor 2 difference between Wirtinger
    // ∂/∂z̄ and the real-vector gradient on (re, im).
    let dz = run_complex_norm_sq_bwd(&z, &[1.0_f32]);
    assert!((2.0 * dz[0].0 - analytic_re).abs() < 1e-5);
    assert!((2.0 * dz[0].1 - analytic_im).abs() < 1e-5);
}

/// Direct regression test for the 5-D mid-shape singleton broadcast
/// (SAM rel_pos pattern: `[bh, h, w, 1, w] + [bh, h, w, h, w]`).
/// The SAM port worked around this by `concat`-tiling the rhs; this
/// test verifies the in-graph broadcast path is bit-correct.
#[test]
fn binary_full_5d_mid_singleton_broadcast() {
    let bh = 2usize;
    let h = 3;
    let w = 4;
    let f = DType::F32;

    let mut g = Graph::new("bcast_5d");
    let lhs = g.input("lhs", Shape::new(&[bh, h, w, h, w], f));
    // rhs shape with size-1 at axis 3 (mid-shape singleton).
    let rhs = g.input("rhs", Shape::new(&[bh, h, w, 1, w], f));
    let out = g.binary(BinaryOp::Add, lhs, rhs, Shape::new(&[bh, h, w, h, w], f));
    g.set_outputs(vec![out]);

    // Deterministic data.
    let lhs_data: Vec<f32> = (0..bh * h * w * h * w).map(|i| i as f32 * 0.01).collect();
    let rhs_data: Vec<f32> = (0..bh * h * w * w)
        .map(|i| (i as f32 + 100.0) * 0.01)
        .collect();

    // Compute expected output by hand.
    let mut expected = vec![0f32; bh * h * w * h * w];
    for b_ in 0..bh {
        for hq in 0..h {
            for wq in 0..w {
                for hk in 0..h {
                    for wk in 0..w {
                        let li = (((b_ * h + hq) * w + wq) * h + hk) * w + wk;
                        // rhs has hk dim = 1, so it's always index 0 there.
                        let ri = ((b_ * h + hq) * w + wq) * w + wk;
                        expected[li] = lhs_data[li] + rhs_data[ri];
                    }
                }
            }
        }
    }

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    let lhs_off = arena.byte_offset(lhs);
    let rhs_off = arena.byte_offset(rhs);
    let out_off = arena.byte_offset(out);
    let buf = arena.raw_buf_mut();
    unsafe {
        let p = buf.as_mut_ptr().add(lhs_off) as *mut f32;
        for (i, &v) in lhs_data.iter().enumerate() {
            *p.add(i) = v;
        }
        let p = buf.as_mut_ptr().add(rhs_off) as *mut f32;
        for (i, &v) in rhs_data.iter().enumerate() {
            *p.add(i) = v;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let actual: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(out_off) as *const f32;
        (0..bh * h * w * h * w).map(|i| *p.add(i)).collect()
    };

    // Bit-exact check.
    let mut max_diff = 0f32;
    let mut max_idx = 0;
    for i in 0..actual.len() {
        let d = (actual[i] - expected[i]).abs();
        if d > max_diff {
            max_diff = d;
            max_idx = i;
        }
    }
    assert!(
        max_diff < 1e-6,
        "5D mid-shape singleton broadcast wrong: max |Δ| = {max_diff} at idx {max_idx} \
             (actual={}, expected={})",
        actual[max_idx],
        expected[max_idx]
    );
}

#[test]
fn layer_norm2d_and_conv_transpose2d_kernels() {
    let mut out = vec![0f32; 8];
    crate::kernels::layer_norm2d_nchw(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[1.0, 1.0],
        &[0.0, 0.0],
        &mut out,
        1,
        2,
        2,
        2,
        1e-5,
    );
    let mean0: f32 = (1.0 + 3.0) / 2.0;
    assert!((out[0] - mean0).abs() > 0.1);

    let mut up = vec![0f32; 4];
    crate::kernels::conv_transpose2d_nchw(
        &[2.0],
        &[1.0, 0.0, 0.0, 1.0],
        &mut up,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        0,
        0,
        1,
        1,
        1,
    );
    assert!((up[0] - 2.0).abs() < 1e-5);
    assert!((up[3] - 2.0).abs() < 1e-5);
}

/// End-to-end native-low-precision GEMM oracle: build the full
/// ScaledQuantScale → ScaledQuantize → ScaledMatMul pipeline through the
/// thunk path and check the f32-accumulated result tracks a plain f32 TN
/// matmul (cosine), for every format × scale-layout. This exercises shape
/// inference, memory planning, the build arms, and both execute paths.
#[test]
fn scaled_matmul_oracle_matches_f32() {
    use rlx_ir::ScaledFormat::*;
    use rlx_ir::{ScaleLayout, ScaledFormat};

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_scaled(
        lhs: &[f32],
        rhs: &[f32],
        m: usize,
        k: usize,
        n: usize,
        lf: ScaledFormat,
        rf: ScaledFormat,
        layout: ScaleLayout,
    ) -> Vec<f32> {
        let f = DType::F32;
        let u8t = DType::U8;
        let mut g = Graph::new("scaled");
        let lhs_in = g.input("lhs", Shape::new(&[m, k], f));
        let rhs_in = g.input("rhs", Shape::new(&[n, k], f));
        let (ls_shape, rs_shape) = match layout {
            ScaleLayout::PerTensor => (Shape::new(&[1], f), Shape::new(&[1], f)),
            _ => {
                let nb = k.div_ceil(layout.block() as usize);
                (Shape::new(&[m, nb], u8t), Shape::new(&[n, nb], u8t))
            }
        };
        let ls = g.add_node(
            Op::ScaledQuantScale {
                format: lf,
                scale_layout: layout,
            },
            vec![lhs_in],
            ls_shape,
        );
        let lq = g.add_node(
            Op::ScaledQuantize {
                format: lf,
                scale_layout: layout,
            },
            vec![lhs_in, ls],
            Shape::new(&[m, k], u8t),
        );
        let rs = g.add_node(
            Op::ScaledQuantScale {
                format: rf,
                scale_layout: layout,
            },
            vec![rhs_in],
            rs_shape,
        );
        let rq = g.add_node(
            Op::ScaledQuantize {
                format: rf,
                scale_layout: layout,
            },
            vec![rhs_in, rs],
            Shape::new(&[n, k], u8t),
        );
        let out = g.add_node(
            Op::ScaledMatMul {
                lhs_format: lf,
                rhs_format: rf,
                scale_layout: layout,
                has_bias: false,
            },
            vec![lq, rq, ls, rs],
            Shape::new(&[m, n], f),
        );
        g.set_outputs(vec![out]);

        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);
        let lhs_off = arena.byte_offset(lhs_in);
        let rhs_off = arena.byte_offset(rhs_in);
        let out_off = arena.byte_offset(out);
        let buf = arena.raw_buf_mut();
        unsafe {
            let lp = buf.as_mut_ptr().add(lhs_off) as *mut f32;
            for (i, &v) in lhs.iter().enumerate() {
                *lp.add(i) = v;
            }
            let rp = buf.as_mut_ptr().add(rhs_off) as *mut f32;
            for (i, &v) in rhs.iter().enumerate() {
                *rp.add(i) = v;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        unsafe {
            let p = arena.raw_buf().as_ptr().add(out_off) as *const f32;
            (0..m * n).map(|i| *p.add(i)).collect()
        }
    }

    let (m, k, n) = (4usize, 64usize, 8usize);
    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin() * 1.5).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.07).cos() * 1.2).collect();
    let mut reference = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += lhs[i * k + p] * rhs[j * k + p];
            }
            reference[i * n + j] = acc;
        }
    }

    // Per-tensor scaling across all 7 element formats.
    let cases = [
        (F8E4M3, 0.999f32),
        (F8E5M2, 0.99),
        (F8E4M3Fnuz, 0.999),
        (F8E5M2Fnuz, 0.99),
        (F6E2M3, 0.99),
        (F6E3M2, 0.98),
        (F4E2M1, 0.90),
    ];
    for (fmt, thresh) in cases {
        let out = run_scaled(&lhs, &rhs, m, k, n, fmt, fmt, ScaleLayout::PerTensor);
        let c = cosine(&out, &reference);
        assert!(c >= thresh, "{fmt} per-tensor cosine {c} < {thresh}");
    }

    // Block layouts: MX E8M0 (FP8) and NVFP4 (FP4) — finer scaling, higher fidelity.
    let out_mx = run_scaled(&lhs, &rhs, m, k, n, F8E4M3, F8E4M3, ScaleLayout::mx());
    let c_mx = cosine(&out_mx, &reference);
    assert!(c_mx >= 0.999, "mx-e8m0 e4m3 cosine {c_mx}");
    let out_nv = run_scaled(&lhs, &rhs, m, k, n, F4E2M1, F4E2M1, ScaleLayout::nvfp4());
    let c_nv = cosine(&out_nv, &reference);
    assert!(c_nv >= 0.95, "nvfp4 e2m1 cosine {c_nv}");
}

/// `Op::ScaledDequantize` (`decode(code)·scale`) must invert
/// `Op::ScaledQuantize`: a quantize→dequantize round-trip reconstructs the
/// original f32 within the format's resolution. Exercises the standalone
/// dequantizer the `ScaledMatMul` backward graph relies on.
#[test]
fn scaled_dequantize_inverts_quantize() {
    use rlx_ir::ScaledFormat::*;
    use rlx_ir::{ScaleLayout, ScaledFormat};

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    fn roundtrip(x: &[f32], rows: usize, cols: usize, fmt: ScaledFormat) -> Vec<f32> {
        let f = DType::F32;
        let u8t = DType::U8;
        let layout = ScaleLayout::PerTensor;
        let mut g = Graph::new("dequant_rt");
        let x_in = g.input("x", Shape::new(&[rows, cols], f));
        let scale = g.add_node(
            Op::ScaledQuantScale {
                format: fmt,
                scale_layout: layout,
            },
            vec![x_in],
            Shape::new(&[1], f),
        );
        let codes = g.add_node(
            Op::ScaledQuantize {
                format: fmt,
                scale_layout: layout,
            },
            vec![x_in, scale],
            Shape::new(&[rows, cols], u8t),
        );
        let recon = g.add_node(
            Op::ScaledDequantize {
                format: fmt,
                scale_layout: layout,
            },
            vec![codes, scale],
            Shape::new(&[rows, cols], f),
        );
        g.set_outputs(vec![recon]);

        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);
        let x_off = arena.byte_offset(x_in);
        let r_off = arena.byte_offset(recon);
        unsafe {
            let p = arena.raw_buf_mut().as_mut_ptr().add(x_off) as *mut f32;
            for (i, &v) in x.iter().enumerate() {
                *p.add(i) = v;
            }
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        unsafe {
            let p = arena.raw_buf().as_ptr().add(r_off) as *const f32;
            (0..rows * cols).map(|i| *p.add(i)).collect()
        }
    }

    let (rows, cols) = (4usize, 16usize);
    let x: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.21).sin() * 1.7)
        .collect();
    // FP8 reconstructs tightly; FP4 is coarse but still strongly correlated.
    for (fmt, min_cos) in [(F8E4M3, 0.999f32), (F8E5M2, 0.99), (F4E2M1, 0.93)] {
        let recon = roundtrip(&x, rows, cols, fmt);
        assert_eq!(recon.len(), x.len());
        assert!(
            recon.iter().all(|v| v.is_finite()),
            "{fmt:?} produced non-finite"
        );
        let c = cosine(&recon, &x);
        assert!(c >= min_cos, "{fmt:?} round-trip cosine {c} < {min_cos}");
    }
}

/// A parameterized `Custom` minifloat (`f4e3m0` — 3 exp, 0 mant: a signed
/// power-of-two grid) must run end-to-end through the real quantize →
/// dequantize graph on the CPU executor. Values drawn from the format's own
/// grid — with ±16 present so the per-tensor scale is exactly 1 — must
/// reconstruct bit-exactly, proving the generic decode/encode path that the
/// graph, the CPU backend, and the Metal host-fallback all share works for
/// an arbitrary (exp, mant) split, not just the seven named formats.
#[test]
fn scaled_custom_f4e3m0_round_trip_is_exact() {
    use rlx_ir::{ScaleLayout, ScaledFormat};

    let fmt = ScaledFormat::custom(3, 0); // f4e3m0
    assert_eq!(fmt.to_string(), "f4e3m0");
    let (rows, cols) = (3usize, 8usize);
    // Grid: 0 and ±{0.25,0.5,1,2,4,8,16}. ±16 present → amax 16 → scale 1.
    let grid = [16.0f32, -8.0, 4.0, -2.0, 1.0, -0.5, 0.25, 0.0];
    let x: Vec<f32> = (0..rows * cols).map(|i| grid[i % grid.len()]).collect();

    let f = DType::F32;
    let u8t = DType::U8;
    let layout = ScaleLayout::PerTensor;
    let mut g = Graph::new("f4e3m0_rt");
    let x_in = g.input("x", Shape::new(&[rows, cols], f));
    let scale = g.add_node(
        Op::ScaledQuantScale {
            format: fmt,
            scale_layout: layout,
        },
        vec![x_in],
        Shape::new(&[1], f),
    );
    let codes = g.add_node(
        Op::ScaledQuantize {
            format: fmt,
            scale_layout: layout,
        },
        vec![x_in, scale],
        Shape::new(&[rows, cols], u8t),
    );
    let recon = g.add_node(
        Op::ScaledDequantize {
            format: fmt,
            scale_layout: layout,
        },
        vec![codes, scale],
        Shape::new(&[rows, cols], f),
    );
    g.set_outputs(vec![recon]);

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    let x_off = arena.byte_offset(x_in);
    let r_off = arena.byte_offset(recon);
    unsafe {
        let p = arena.raw_buf_mut().as_mut_ptr().add(x_off) as *mut f32;
        for (i, &v) in x.iter().enumerate() {
            *p.add(i) = v;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let recon_vals: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(r_off) as *const f32;
        (0..rows * cols).map(|i| *p.add(i)).collect()
    };
    assert_eq!(recon_vals, x, "f4e3m0 grid values must round-trip exactly");
}

/// The ergonomic `Graph::scaled_matmul` builder — one call composes the whole
/// ScaledQuantScale→ScaledQuantize→ScaledMatMul chain from a `ScaledFormat` +
/// `ScaleLayout` — must produce a graph that runs and tracks the f32 matmul.
#[test]
fn scaled_matmul_builder_helper_tracks_f32() {
    use rlx_ir::{ScaleLayout, ScaledFormat};

    let (m, k, n) = (4usize, 64usize, 8usize);
    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin() * 1.5).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.07).cos() * 1.2).collect();

    let mut g = Graph::new("scaled_builder");
    let lhs_in = g.input("lhs", Shape::new(&[m, k], DType::F32));
    let rhs_in = g.input("rhs", Shape::new(&[n, k], DType::F32));
    // One call instead of hand-wiring five ops.
    let y = g.scaled_matmul(
        lhs_in,
        rhs_in,
        ScaledFormat::custom(3, 0),
        ScaleLayout::mx(),
    );
    g.set_outputs(vec![y]);

    // Builder wired the expected op chain.
    let count = |k: rlx_ir::OpKind| g.nodes().iter().filter(|nd| nd.op.kind() == k).count();
    assert_eq!(count(rlx_ir::OpKind::ScaledMatMul), 1);
    assert_eq!(count(rlx_ir::OpKind::ScaledQuantize), 2);
    assert_eq!(count(rlx_ir::OpKind::ScaledQuantScale), 2);

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    let lhs_off = arena.byte_offset(lhs_in);
    let rhs_off = arena.byte_offset(rhs_in);
    let y_off = arena.byte_offset(y);
    unsafe {
        let p = arena.raw_buf_mut().as_mut_ptr();
        let lp = p.add(lhs_off) as *mut f32;
        for (i, &v) in lhs.iter().enumerate() {
            *lp.add(i) = v;
        }
        let rp = p.add(rhs_off) as *mut f32;
        for (i, &v) in rhs.iter().enumerate() {
            *rp.add(i) = v;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let out: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(y_off) as *const f32;
        (0..m * n).map(|i| *p.add(i)).collect()
    };

    // Reference f32 matmul (TN).
    let mut reference = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += lhs[i * k + p] * rhs[j * k + p];
            }
            reference[i * n + j] = acc;
        }
    }
    let dot: f32 = out.iter().zip(&reference).map(|(a, b)| a * b).sum();
    let na = out.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = reference.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb);
    assert!(cos >= 0.9, "scaled_matmul builder cosine {cos} < 0.9");
}

/// `Op::Fma` computes the single-rounded `a*b + c` elementwise. Verify it
/// matches `f32::mul_add` (the hardware FMA), and that the `LowerFma`
/// fallback (`Mul` + `Add`) stays close on a benign case.
#[test]
fn fma_matches_mul_add() {
    let f = DType::F32;
    let n = 9usize;
    let a: Vec<f32> = vec![1.5, -2.0, 0.0, 3.25, -1.1, 7.0, -0.5, 2.2, 9.9];
    let b: Vec<f32> = vec![2.0, 0.5, 4.0, -1.0, 6.0, -2.5, 8.0, -3.3, 0.1];
    let c: Vec<f32> = vec![0.25, 1.0, -3.0, 2.0, -0.5, 4.0, 1.5, -2.2, 0.0];

    let mut g = Graph::new("fma");
    let an = g.input("a", Shape::new(&[n], f));
    let bn = g.input("b", Shape::new(&[n], f));
    let cn = g.input("c", Shape::new(&[n], f));
    let out = g.add_node(Op::Fma, vec![an, bn, cn], Shape::new(&[n], f));
    g.set_outputs(vec![out]);

    let actual = run_graph(&g, &[(an, &a), (bn, &b), (cn, &c)], out, n);
    for i in 0..n {
        let expected = a[i].mul_add(b[i], c[i]);
        assert!(
            (actual[i] - expected).abs() <= f32::EPSILON * (1.0 + expected.abs()),
            "fma[{i}]: {} vs mul_add {expected}",
            actual[i]
        );
    }
}

/// End-to-end AMP-FP8: take a plain `MatMul` graph, run the
/// `insert_scaled_matmul` compile pass, then execute the rewritten graph on
/// CPU and confirm it tracks the f32 matmul. Proves the pass emits a
/// runnable, numerically-sound graph (rhs transpose + dynamic quantize).
#[test]
fn scaled_quant_pass_runs_end_to_end() {
    use rlx_opt::rlx_compile::scaled_quant_insert::{ScaledQuantConfig, insert_scaled_matmul};

    let f = DType::F32;
    let (m, k, n) = (3usize, 16usize, 5usize);
    let mut g = Graph::new("amp_fp8");
    let x = g.input("x", Shape::new(&[m, k], f));
    let w = g.param("w", Shape::new(&[k, n], f));
    let mm = g.matmul(x, w, Shape::new(&[m, n], f));
    g.set_outputs(vec![mm]);

    let g = insert_scaled_matmul(g, ScaledQuantConfig::fp8_e4m3());

    let x_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.11).sin()).collect();
    let w_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.05).cos()).collect();
    // reference NN matmul: out[i,j] = Σ_p x[i,p]·w[p,j]
    let mut reference = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += x_data[i * k + p] * w_data[p * n + j];
            }
            reference[i * n + j] = acc;
        }
    }

    // Locate the (preserved) input/param + output nodes in the rewritten graph.
    let mut x_id = None;
    let mut w_id = None;
    for node in g.nodes() {
        match &node.op {
            Op::Input { name } if name == "x" => x_id = Some(node.id),
            Op::Param { name } if name == "w" => w_id = Some(node.id),
            _ => {}
        }
    }
    let (x_id, w_id) = (x_id.unwrap(), w_id.unwrap());
    let out_id = g.outputs[0];

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    let x_off = arena.byte_offset(x_id);
    let w_off = arena.byte_offset(w_id);
    let out_off = arena.byte_offset(out_id);
    let buf = arena.raw_buf_mut();
    unsafe {
        let xp = buf.as_mut_ptr().add(x_off) as *mut f32;
        for (i, &v) in x_data.iter().enumerate() {
            *xp.add(i) = v;
        }
        let wp = buf.as_mut_ptr().add(w_off) as *mut f32;
        for (i, &v) in w_data.iter().enumerate() {
            *wp.add(i) = v;
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let actual: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(out_off) as *const f32;
        (0..m * n).map(|i| *p.add(i)).collect()
    };

    let dot: f32 = actual.iter().zip(&reference).map(|(a, b)| a * b).sum();
    let na = actual.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = reference.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos = dot / (na * nb);
    assert!(cos >= 0.999, "AMP-fp8 e2e cosine {cos}");
}

// ── DiT modulation ops (AdaLayerNorm / GatedResidual) ───────────────────

/// `Op::AdaLayerNorm` native CPU kernel must match the reference
/// `norm(x)·(1+scale)+shift` for both LayerNorm and RMSNorm flavors, with
/// the `[B,1,D]` modulation broadcast over the sequence axis.
#[test]
fn ada_layer_norm_matches_reference() {
    use rlx_ir::infer::GraphExt;
    use rlx_ir::op::AdaNormKind;
    let f = DType::F32;
    let (bch, s, d) = (2usize, 5usize, 8usize);
    let eps = 1e-5f32;

    let mut seed: u32 = 0x1234_5678;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) - 0.5
    };
    let x: Vec<f32> = (0..bch * s * d).map(|_| rand()).collect();
    let scale: Vec<f32> = (0..bch * d).map(|_| rand()).collect(); // [B,1,D]
    let shift: Vec<f32> = (0..bch * d).map(|_| rand()).collect();

    for &ln in &[true, false] {
        // Reference.
        let mut expected = vec![0f32; bch * s * d];
        for b in 0..bch {
            for si in 0..s {
                let row = &x[(b * s + si) * d..(b * s + si) * d + d];
                let (mean, inv) = if ln {
                    let mean = row.iter().sum::<f32>() / d as f32;
                    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
                    (mean, 1.0 / (var + eps).sqrt())
                } else {
                    let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
                    (0f32, 1.0 / (ms + eps).sqrt())
                };
                for di in 0..d {
                    let n = (row[di] - mean) * inv;
                    expected[(b * s + si) * d + di] =
                        n * (1.0 + scale[b * d + di]) + shift[b * d + di];
                }
            }
        }

        // RLX.
        let mut g = Graph::new("ada");
        let xn = g.input("x", Shape::new(&[bch, s, d], f));
        let scn = g.input("scale", Shape::new(&[bch, 1, d], f));
        let shn = g.input("shift", Shape::new(&[bch, 1, d], f));
        let norm = if ln {
            AdaNormKind::LayerNorm
        } else {
            AdaNormKind::RmsNorm
        };
        let out = g.ada_layer_norm(xn, scn, shn, norm, eps);
        g.set_outputs(vec![out]);

        let plan = rlx_opt::memory::plan_memory(&g);
        let mut arena = crate::arena::Arena::from_plan(plan);
        let sched = compile_thunks(&g, &arena);
        let xoff = arena.byte_offset(xn);
        let scoff = arena.byte_offset(scn);
        let shoff = arena.byte_offset(shn);
        let ooff = arena.byte_offset(out);
        let buf = arena.raw_buf_mut();
        unsafe {
            let cp = |dst: *mut f32, data: &[f32]| {
                for (i, &v) in data.iter().enumerate() {
                    *dst.add(i) = v;
                }
            };
            cp(buf.as_mut_ptr().add(xoff) as *mut f32, &x);
            cp(buf.as_mut_ptr().add(scoff) as *mut f32, &scale);
            cp(buf.as_mut_ptr().add(shoff) as *mut f32, &shift);
        }
        execute_thunks(&sched, arena.raw_buf_mut());
        let actual: Vec<f32> = unsafe {
            let p = arena.raw_buf().as_ptr().add(ooff) as *const f32;
            (0..bch * s * d).map(|i| *p.add(i)).collect()
        };
        for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
            assert!(
                (e - a).abs() < 1e-4,
                "ada layer_norm={ln} mismatch at {i}: expected {e}, got {a}"
            );
        }
    }
}

/// `Op::GatedResidual` native CPU kernel must match `x + gate·y` with the
/// `[B,1,D]` gate broadcast over the sequence axis.
#[test]
fn gated_residual_matches_reference() {
    use rlx_ir::infer::GraphExt;
    let f = DType::F32;
    let (bch, s, d) = (2usize, 4usize, 6usize);

    let mut seed: u32 = 0xdead_beef;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) - 0.5
    };
    let x: Vec<f32> = (0..bch * s * d).map(|_| rand()).collect();
    let y: Vec<f32> = (0..bch * s * d).map(|_| rand()).collect();
    let gate: Vec<f32> = (0..bch * d).map(|_| rand()).collect(); // [B,1,D]

    let mut expected = vec![0f32; bch * s * d];
    for b in 0..bch {
        for si in 0..s {
            for di in 0..d {
                let idx = (b * s + si) * d + di;
                expected[idx] = x[idx] + gate[b * d + di] * y[idx];
            }
        }
    }

    let mut g = Graph::new("gated");
    let xn = g.input("x", Shape::new(&[bch, s, d], f));
    let yn = g.input("y", Shape::new(&[bch, s, d], f));
    let gn = g.input("gate", Shape::new(&[bch, 1, d], f));
    let out = g.gated_residual(xn, yn, gn);
    g.set_outputs(vec![out]);

    let plan = rlx_opt::memory::plan_memory(&g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(&g, &arena);
    let xoff = arena.byte_offset(xn);
    let yoff = arena.byte_offset(yn);
    let goff = arena.byte_offset(gn);
    let ooff = arena.byte_offset(out);
    let buf = arena.raw_buf_mut();
    unsafe {
        let cp = |dst: *mut f32, data: &[f32]| {
            for (i, &v) in data.iter().enumerate() {
                *dst.add(i) = v;
            }
        };
        cp(buf.as_mut_ptr().add(xoff) as *mut f32, &x);
        cp(buf.as_mut_ptr().add(yoff) as *mut f32, &y);
        cp(buf.as_mut_ptr().add(goff) as *mut f32, &gate);
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let actual: Vec<f32> = unsafe {
        let p = arena.raw_buf().as_ptr().add(ooff) as *const f32;
        (0..bch * s * d).map(|i| *p.add(i)).collect()
    };
    for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
        assert!(
            (e - a).abs() < 1e-5,
            "gated_residual mismatch at {i}: expected {e}, got {a}"
        );
    }
}

/// Compile + run a single-output graph on CPU, loading each named input, and
/// return the output as a `Vec<f32>`. Used to compare a native fused op with
/// its `unfuse` decomposition (the path every non-CPU backend takes).
#[cfg(test)]
fn run_graph_cpu(g: &Graph, inputs: &[(&str, &[f32])], out_len: usize) -> Vec<f32> {
    let plan = rlx_opt::memory::plan_memory(g);
    let mut arena = crate::arena::Arena::from_plan(plan);
    let sched = compile_thunks(g, &arena);
    // Materialize Op::Constant data into the arena (the runtime's loader does
    // this in production; e.g. the decomposition's gamma=ones / beta=zeros).
    for n in g.nodes() {
        if let Op::Constant { data } = &n.op {
            let off = arena.byte_offset(n.id);
            let buf = arena.raw_buf_mut();
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), buf.as_mut_ptr().add(off), data.len());
            }
        }
    }
    for (name, data) in inputs {
        let id = g
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, Op::Input { name: nm } if nm.as_str() == *name))
            .unwrap_or_else(|| panic!("input {name} not found"))
            .id;
        let off = arena.byte_offset(id);
        let buf = arena.raw_buf_mut();
        unsafe {
            let dst = buf.as_mut_ptr().add(off) as *mut f32;
            for (i, &v) in data.iter().enumerate() {
                *dst.add(i) = v;
            }
        }
    }
    execute_thunks(&sched, arena.raw_buf_mut());
    let ooff = arena.byte_offset(g.outputs[0]);
    unsafe {
        let p = arena.raw_buf().as_ptr().add(ooff) as *const f32;
        (0..out_len).map(|i| *p.add(i)).collect()
    }
}

/// The `unfuse` decomposition of `Op::AdaLayerNorm` (norm → mul(1+scale) →
/// add(shift)) — the path every non-CPU backend runs — must match the native
/// fused kernel bit-for-bit-close.
#[test]
fn ada_layer_norm_decompose_matches_native() {
    use rlx_ir::infer::GraphExt;
    use rlx_ir::op::AdaNormKind;
    let f = DType::F32;
    let (bch, s, d) = (2usize, 5usize, 8usize);
    let eps = 1e-5f32;

    let mut seed: u32 = 0x0bad_c0de;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) - 0.5
    };
    let x: Vec<f32> = (0..bch * s * d).map(|_| rand()).collect();
    let scale: Vec<f32> = (0..bch * d).map(|_| rand()).collect();
    let shift: Vec<f32> = (0..bch * d).map(|_| rand()).collect();

    for &ln in &[true, false] {
        let norm = if ln {
            AdaNormKind::LayerNorm
        } else {
            AdaNormKind::RmsNorm
        };
        let build = || {
            let mut g = Graph::new("ada");
            let xn = g.input("x", Shape::new(&[bch, s, d], f));
            let scn = g.input("scale", Shape::new(&[bch, 1, d], f));
            let shn = g.input("shift", Shape::new(&[bch, 1, d], f));
            let out = g.ada_layer_norm(xn, scn, shn, norm, eps);
            g.set_outputs(vec![out]);
            g
        };
        let inputs: &[(&str, &[f32])] = &[("x", &x), ("scale", &scale), ("shift", &shift)];

        let native = run_graph_cpu(&build(), inputs, bch * s * d);
        // Mirror a non-CPU backend: unfuse to primitives, then legalize the
        // broadcasts the decomposition introduces (the real backend pipeline
        // runs both before lowering).
        let decomposed = rlx_opt::rlx_compile::legalize_broadcast::run(
            rlx_opt::rlx_fusion::unfuse::unfuse_dit_modulation(build()),
        );
        assert!(
            !decomposed
                .nodes()
                .iter()
                .any(|n| matches!(n.op, Op::AdaLayerNorm { .. })),
            "unfuse left an AdaLayerNorm node"
        );
        let decomp = run_graph_cpu(&decomposed, inputs, bch * s * d);

        for (i, (nv, dv)) in native.iter().zip(&decomp).enumerate() {
            assert!(
                (nv - dv).abs() < 1e-4,
                "ada ln={ln} native vs decompose mismatch at {i}: {nv} vs {dv}"
            );
        }
    }
}

/// The `unfuse` decomposition of `Op::GatedResidual` (mul(gate,y) → add(x))
/// must match the native fused kernel.
#[test]
fn gated_residual_decompose_matches_native() {
    use rlx_ir::infer::GraphExt;
    let f = DType::F32;
    let (bch, s, d) = (2usize, 4usize, 6usize);

    let mut seed: u32 = 0xfeed_face;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) - 0.5
    };
    let x: Vec<f32> = (0..bch * s * d).map(|_| rand()).collect();
    let y: Vec<f32> = (0..bch * s * d).map(|_| rand()).collect();
    let gate: Vec<f32> = (0..bch * d).map(|_| rand()).collect();

    let build = || {
        let mut g = Graph::new("gated");
        let xn = g.input("x", Shape::new(&[bch, s, d], f));
        let yn = g.input("y", Shape::new(&[bch, s, d], f));
        let gn = g.input("gate", Shape::new(&[bch, 1, d], f));
        let out = g.gated_residual(xn, yn, gn);
        g.set_outputs(vec![out]);
        g
    };
    let inputs: &[(&str, &[f32])] = &[("x", &x), ("y", &y), ("gate", &gate)];

    let native = run_graph_cpu(&build(), inputs, bch * s * d);
    let decomposed = rlx_opt::rlx_compile::legalize_broadcast::run(
        rlx_opt::rlx_fusion::unfuse::unfuse_dit_modulation(build()),
    );
    assert!(
        !decomposed
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GatedResidual)),
        "unfuse left a GatedResidual node"
    );
    let decomp = run_graph_cpu(&decomposed, inputs, bch * s * d);

    for (i, (nv, dv)) in native.iter().zip(&decomp).enumerate() {
        assert!(
            (nv - dv).abs() < 1e-5,
            "gated_residual native vs decompose mismatch at {i}: {nv} vs {dv}"
        );
    }
}
