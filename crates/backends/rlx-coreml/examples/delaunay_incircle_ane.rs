// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// SPECIAL VERSION FOR THE APPLE NEURAL ENGINE (ANE) of the Delaunay in-circle predicate,
// with a SPEED-vs-PRECISION config API.
//
// The in-circle PREDICATE is a batch of 4-point determinants = pure elementwise tensor
// arithmetic, which the ANE runs. This builds it as an rlx IR Graph, lowers to CoreML
// (MIL → .mlpackage, ComputeUnits::CpuAndNeuralEngine → ANE), and validates the sign vs an
// exact i128 reference. PER-QUAD normalization (translate by d, scale each test so
// max|·−d|≈TARGET) keeps every degree-4 determinant O(1): no f16 overflow, full mantissa.
//
// `IncircleConfig` picks the speed↔precision point:
//   Fast      — pure f16 on the ANE, no fallback          (fastest; ~99.95% sign-correct)
//   DoubleF16 — "f16+f16 ≈ f32": each value a (hi,lo) f16 pair, compensated arithmetic
//               (TwoSum/TwoProduct) → ~22 mantissa bits, ALL on the ANE, NO host fallback
//               (100% correct on real data; still not a GUARANTEED exact predicate)
//   Filtered  — f16 on the ANE + EXACT i128 host fallback for any |det| inside the f16
//               error bound                                (guaranteed 100%; ~0.05% on host)
//   Exact     — exact i128 on the host                     (100%; 0% on NPU, slowest)
// `filter_k` tunes the certified bound (larger ⇒ safer ⇒ more fallback ⇒ slower).
//   cargo run -p rlx-coreml --example delaunay_incircle_ane --release [-- K]
// CoreML/ANE is Apple-only; wrap the example so non-Apple hosts still compile
// (with a stub `main`) instead of failing with "main function not found".
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod imp {
    use half::f16;
    use rlx_coreml::{ComputeUnits, CoremlExecutable};
    use rlx_ir::op::BinaryOp;
    use rlx_ir::{DType, Graph, NodeId, Op, Shape};

    /// f16 machine epsilon (11-bit mantissa).
    const F16_EPS: f64 = 1.0 / 2048.0;

    /// Speed↔precision configuration for the NPU in-circle predicate.
    #[derive(Clone, Copy)]
    struct IncircleConfig {
        mode: Mode,
        /// per-quad normalization magnitude (f16 conditioning; ~4 keeps det O(1) w/o overflow).
        target_scale: f64,
        /// certified error-bound multiplier for `Filtered` (K in K·ε·perm); larger ⇒ more fallback.
        filter_k: f64,
    }
    #[derive(Clone, Copy, PartialEq)]
    enum Mode {
        Fast,
        DoubleF16,
        Filtered,
        Exact,
    }
    impl IncircleConfig {
        fn fast() -> Self {
            Self {
                mode: Mode::Fast,
                target_scale: 4.0,
                filter_k: 0.0,
            }
        }
        fn filtered() -> Self {
            Self {
                mode: Mode::Filtered,
                target_scale: 4.0,
                filter_k: 8.0,
            }
        }
        fn filtered_k(filter_k: f64) -> Self {
            Self {
                mode: Mode::Filtered,
                target_scale: 4.0,
                filter_k,
            }
        }
        fn exact() -> Self {
            Self {
                mode: Mode::Exact,
                target_scale: 4.0,
                filter_k: 0.0,
            }
        }
        fn double_f16() -> Self {
            Self {
                mode: Mode::DoubleF16,
                target_scale: 4.0,
                filter_k: 0.0,
            }
        }
        fn label(&self) -> &'static str {
            match self.mode {
                Mode::Fast => "Fast     (pure f16 on ANE)",
                Mode::DoubleF16 => "DoubleF16(f16+f16≈f32 on ANE)",
                Mode::Filtered => "Filtered (f16 ANE + exact fallback)",
                Mode::Exact => "Exact    (i128 host)",
            }
        }
    }

    // exact i128 in-circle sign (ground truth / host fallback)
    fn incircle_exact(a: [i64; 2], b: [i64; 2], c: [i64; 2], d: [i64; 2]) -> i32 {
        let (ax, ay) = ((a[0] - d[0]) as i128, (a[1] - d[1]) as i128);
        let (bx, by) = ((b[0] - d[0]) as i128, (b[1] - d[1]) as i128);
        let (cx, cy) = ((c[0] - d[0]) as i128, (c[1] - d[1]) as i128);
        let (a2, b2, c2) = (ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy);
        (a2 * (bx * cy - cx * by) - b2 * (ax * cy - cx * ay) + c2 * (ax * by - bx * ay)).signum()
            as i32
    }

    /// The in-circle determinant batch [B] as an rlx graph (F16 or F32 datapath).
    fn build_graph(b: usize, f16: bool) -> Graph {
        let mut g = Graph::new("incircle");
        let names = ["Ax", "Ay", "Bx", "By", "Cx", "Cy", "Dx", "Dy"];
        let mut inp: Vec<rlx_ir::NodeId> = names
            .iter()
            .map(|n| g.input(*n, Shape::new(&[b], DType::F32)))
            .collect();
        let dt = if f16 { DType::F16 } else { DType::F32 };
        if f16 {
            for v in inp.iter_mut() {
                *v = g.append_node(
                    Op::Cast { to: DType::F16 },
                    vec![*v],
                    Shape::new(&[b], DType::F16),
                    None,
                );
            }
        }
        let bin = |g: &mut Graph, op, x, y| g.binary(op, x, y, Shape::new(&[b], dt));
        let (ax0, ay0, bx0, by0, cx0, cy0, dx, dy) = (
            inp[0], inp[1], inp[2], inp[3], inp[4], inp[5], inp[6], inp[7],
        );
        let ax = bin(&mut g, BinaryOp::Sub, ax0, dx);
        let ay = bin(&mut g, BinaryOp::Sub, ay0, dy);
        let bx = bin(&mut g, BinaryOp::Sub, bx0, dx);
        let by = bin(&mut g, BinaryOp::Sub, by0, dy);
        let cx = bin(&mut g, BinaryOp::Sub, cx0, dx);
        let cy = bin(&mut g, BinaryOp::Sub, cy0, dy);
        let sq = |g: &mut Graph, x, y| {
            let xx = g.binary(BinaryOp::Mul, x, x, Shape::new(&[b], dt));
            let yy = g.binary(BinaryOp::Mul, y, y, Shape::new(&[b], dt));
            g.binary(BinaryOp::Add, xx, yy, Shape::new(&[b], dt))
        };
        let a2 = sq(&mut g, ax, ay);
        let b2 = sq(&mut g, bx, by);
        let c2 = sq(&mut g, cx, cy);
        let minor = |g: &mut Graph, p, s, q, r| {
            let ps = g.binary(BinaryOp::Mul, p, s, Shape::new(&[b], dt));
            let qr = g.binary(BinaryOp::Mul, q, r, Shape::new(&[b], dt));
            g.binary(BinaryOp::Sub, ps, qr, Shape::new(&[b], dt))
        };
        let m_bc = minor(&mut g, bx, cy, cx, by);
        let m_ac = minor(&mut g, ax, cy, cx, ay);
        let m_ab = minor(&mut g, ax, by, bx, ay);
        let t1 = bin(&mut g, BinaryOp::Mul, a2, m_bc);
        let t2 = bin(&mut g, BinaryOp::Mul, b2, m_ac);
        let t3 = bin(&mut g, BinaryOp::Mul, c2, m_ab);
        let s1 = bin(&mut g, BinaryOp::Sub, t1, t2);
        let mut det = bin(&mut g, BinaryOp::Add, s1, t3);
        if f16 {
            det = g.append_node(
                Op::Cast { to: DType::F32 },
                vec![det],
                Shape::new(&[b], DType::F32),
                None,
            );
        }
        g.set_outputs(vec![det]);
        g
    }

    // ---- DOUBLE-f16 ("f16+f16" ≈ f32) compensated arithmetic, built from f16 ops only ----
    // Each value is a pair (hi, lo) with hi = fl16(x), lo = x − hi, giving ~22 mantissa bits
    // (near f32) while every op stays f16 — so the ANE can run it. Uses Knuth TwoSum + Dekker
    // TwoProduct (error-free transforms); no constants (65·a via bit-doublings, −x via x−2x)
    // so nothing gets constant-folded away. Whether the *hardware* preserves the extra bits
    // depends on CoreML/ANE not fusing (FMA) or widening these ops — measured below.
    fn fh(b: usize) -> Shape {
        Shape::new(&[b], DType::F16)
    }
    fn fa(g: &mut Graph, b: usize, x: NodeId, y: NodeId) -> NodeId {
        g.binary(BinaryOp::Add, x, y, fh(b))
    }
    fn fs(g: &mut Graph, b: usize, x: NodeId, y: NodeId) -> NodeId {
        g.binary(BinaryOp::Sub, x, y, fh(b))
    }
    fn fm(g: &mut Graph, b: usize, x: NodeId, y: NodeId) -> NodeId {
        g.binary(BinaryOp::Mul, x, y, fh(b))
    }
    fn fneg(g: &mut Graph, b: usize, x: NodeId) -> NodeId {
        let x2 = fa(g, b, x, x);
        fs(g, b, x, x2)
    } // x−2x=−x
    fn two_sum(g: &mut Graph, b: usize, a: NodeId, c: NodeId) -> (NodeId, NodeId) {
        let s = fa(g, b, a, c);
        let bb = fs(g, b, s, a);
        let sa = fs(g, b, s, bb);
        let ea = fs(g, b, a, sa);
        let ec = fs(g, b, c, bb);
        (s, fa(g, b, ea, ec))
    }
    fn fast_two_sum(g: &mut Graph, b: usize, a: NodeId, c: NodeId) -> (NodeId, NodeId) {
        let s = fa(g, b, a, c);
        let t = fs(g, b, s, a);
        (s, fs(g, b, c, t))
    }
    fn split65(g: &mut Graph, b: usize, a: NodeId) -> (NodeId, NodeId) {
        let d2 = fa(g, b, a, a);
        let d4 = fa(g, b, d2, d2);
        let d8 = fa(g, b, d4, d4);
        let d16 = fa(g, b, d8, d8);
        let d32 = fa(g, b, d16, d16);
        let d64 = fa(g, b, d32, d32);
        let c = fa(g, b, a, d64); // 65·a (f16-rounded, exact enough for Dekker)
        let cma = fs(g, b, c, a);
        let hi = fs(g, b, c, cma);
        let lo = fs(g, b, a, hi);
        (hi, lo)
    }
    fn two_prod(g: &mut Graph, b: usize, a: NodeId, c: NodeId) -> (NodeId, NodeId) {
        let p = fm(g, b, a, c);
        let (ah, al) = split65(g, b, a);
        let (ch, cl) = split65(g, b, c);
        let ahch = fm(g, b, ah, ch);
        let e0 = fs(g, b, ahch, p);
        let ahcl = fm(g, b, ah, cl);
        let e1 = fa(g, b, e0, ahcl);
        let alch = fm(g, b, al, ch);
        let e2 = fa(g, b, e1, alch);
        let alcl = fm(g, b, al, cl);
        (p, fa(g, b, e2, alcl))
    }
    fn p_add(
        g: &mut Graph,
        b: usize,
        ah: NodeId,
        al: NodeId,
        bh: NodeId,
        bl: NodeId,
    ) -> (NodeId, NodeId) {
        let (s, e) = two_sum(g, b, ah, bh);
        let albl = fa(g, b, al, bl);
        let e = fa(g, b, e, albl);
        fast_two_sum(g, b, s, e)
    }
    fn p_sub(
        g: &mut Graph,
        b: usize,
        ah: NodeId,
        al: NodeId,
        bh: NodeId,
        bl: NodeId,
    ) -> (NodeId, NodeId) {
        let nh = fneg(g, b, bh);
        let nl = fneg(g, b, bl);
        p_add(g, b, ah, al, nh, nl)
    }
    fn p_mul(
        g: &mut Graph,
        b: usize,
        ah: NodeId,
        al: NodeId,
        bh: NodeId,
        bl: NodeId,
    ) -> (NodeId, NodeId) {
        let (p, e) = two_prod(g, b, ah, bh);
        let ahbl = fm(g, b, ah, bl);
        let albh = fm(g, b, al, bh);
        let cross = fa(g, b, ahbl, albh);
        let e = fa(g, b, e, cross);
        fast_two_sum(g, b, p, e)
    }

    /// In-circle determinant in double-f16 (inputs are per-quad-normalized coord PAIRS; D=0).
    fn build_graph_df16(b: usize) -> Graph {
        let mut g = Graph::new("incircle_df16");
        let names = [
            "Axh", "Axl", "Ayh", "Ayl", "Bxh", "Bxl", "Byh", "Byl", "Cxh", "Cxl", "Cyh", "Cyl",
        ];
        let inp: Vec<(NodeId, NodeId)> = (0..6)
            .map(|j| {
                let h = g.input(names[2 * j], Shape::new(&[b], DType::F32));
                let l = g.input(names[2 * j + 1], Shape::new(&[b], DType::F32));
                let hf = g.append_node(Op::Cast { to: DType::F16 }, vec![h], fh(b), None);
                let lf = g.append_node(Op::Cast { to: DType::F16 }, vec![l], fh(b), None);
                (hf, lf)
            })
            .collect();
        let (ax, ay, bx, by, cx, cy) = (inp[0], inp[1], inp[2], inp[3], inp[4], inp[5]);
        let pm = |g: &mut Graph, u: (NodeId, NodeId), v: (NodeId, NodeId)| {
            p_mul(g, b, u.0, u.1, v.0, v.1)
        };
        let pa = |g: &mut Graph, u: (NodeId, NodeId), v: (NodeId, NodeId)| {
            p_add(g, b, u.0, u.1, v.0, v.1)
        };
        let ps = |g: &mut Graph, u: (NodeId, NodeId), v: (NodeId, NodeId)| {
            p_sub(g, b, u.0, u.1, v.0, v.1)
        };
        let axx = pm(&mut g, ax, ax);
        let ayy = pm(&mut g, ay, ay);
        let a2 = pa(&mut g, axx, ayy);
        let bxx = pm(&mut g, bx, bx);
        let byy = pm(&mut g, by, by);
        let b2 = pa(&mut g, bxx, byy);
        let cxx = pm(&mut g, cx, cx);
        let cyy = pm(&mut g, cy, cy);
        let c2 = pa(&mut g, cxx, cyy);
        let m_bc = {
            let l = pm(&mut g, bx, cy);
            let r = pm(&mut g, cx, by);
            ps(&mut g, l, r)
        };
        let m_ac = {
            let l = pm(&mut g, ax, cy);
            let r = pm(&mut g, cx, ay);
            ps(&mut g, l, r)
        };
        let m_ab = {
            let l = pm(&mut g, ax, by);
            let r = pm(&mut g, bx, ay);
            ps(&mut g, l, r)
        };
        let t1 = pm(&mut g, a2, m_bc);
        let t2 = pm(&mut g, b2, m_ac);
        let t3 = pm(&mut g, c2, m_ab);
        let det = {
            let s = ps(&mut g, t1, t2);
            pa(&mut g, s, t3)
        };
        // reconstruct hi+lo in f32 for the sign
        let hf = g.append_node(
            Op::Cast { to: DType::F32 },
            vec![det.0],
            Shape::new(&[b], DType::F32),
            None,
        );
        let lf = g.append_node(
            Op::Cast { to: DType::F32 },
            vec![det.1],
            Shape::new(&[b], DType::F32),
            None,
        );
        let out = g.binary(BinaryOp::Add, hf, lf, Shape::new(&[b], DType::F32));
        g.set_outputs(vec![out]);
        g
    }

    /// Split an f64 value into an f16 pair (hi, lo) as f32s exactly representable in f16.
    fn split_pair(x: f64) -> (f32, f32) {
        let hi = f16::from_f64(x);
        let lo = f16::from_f64(x - hi.to_f64());
        (hi.to_f32(), lo.to_f32())
    }

    /// Run the whole batch through an ANE-sized graph in `chunk`-wide slices (a single flat
    /// batch trips the ANE compiler; chunking keeps every run on the Neural Engine).
    fn run_batch(
        e: &mut CoremlExecutable,
        cols: &[Vec<f32>],
        b: usize,
        chunk: usize,
        names: &[&str],
    ) -> Vec<f32> {
        let mut all = Vec::with_capacity(b);
        let mut start = 0;
        while start < b {
            let end = (start + chunk).min(b);
            let ins: Vec<Vec<f32>> = (0..names.len())
                .map(|j| {
                    let mut v = cols[j][start..end].to_vec();
                    v.resize(chunk, 0.0);
                    v
                })
                .collect();
            let refs: Vec<(&str, &[f32])> = names
                .iter()
                .zip(&ins)
                .map(|(n, v)| (*n, v.as_slice()))
                .collect();
            let out = e.run(&refs).unwrap().remove(0);
            all.extend_from_slice(&out[..end - start]);
            start = end;
        }
        all
    }

    pub fn run() {
        let k = std::env::args()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .unwrap_or(48usize);
        let span = 15_000_000i64;
        let mut s = 0x1234_5678u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) as i64
        };
        let mut seen = std::collections::HashSet::new();
        let mut p: Vec<[i64; 2]> = vec![];
        while p.len() < k {
            let q = [rnd() % span, rnd() % span];
            if seen.insert(q) {
                p.push(q);
            }
        }

        // all (triangle, test-point) in-circle tests = the dense leaf batch
        let mut quad: Vec<[[i64; 2]; 4]> = vec![];
        let mut exact: Vec<i32> = vec![];
        for a in 0..k {
            for b in (a + 1)..k {
                for c in (b + 1)..k {
                    for d in 0..k {
                        if d == a || d == b || d == c {
                            continue;
                        }
                        quad.push([p[a], p[b], p[c], p[d]]);
                        exact.push(incircle_exact(p[a], p[b], p[c], p[d]));
                    }
                }
            }
        }
        let b = quad.len();

        // per-quad normalization (target_scale) + the per-test error-bound magnitude `perm`
        // (permanent of the determinant), used by the Filtered mode's certified fallback test.
        let target = IncircleConfig::filtered().target_scale;
        let mut cols: Vec<Vec<f32>> = (0..8).map(|_| Vec::with_capacity(b)).collect();
        let mut cols_pair: Vec<Vec<f32>> = (0..12).map(|_| Vec::with_capacity(b)).collect(); // f16 pairs
        let mut perm: Vec<f64> = Vec::with_capacity(b);
        for q in &quad {
            let (a, bb, c, d) = (q[0], q[1], q[2], q[3]);
            let df = [
                a[0] - d[0],
                a[1] - d[1],
                bb[0] - d[0],
                bb[1] - d[1],
                c[0] - d[0],
                c[1] - d[1],
            ];
            let r = df.iter().map(|v| v.unsigned_abs()).max().unwrap().max(1) as f64;
            let sc = target / r;
            let n: Vec<f64> = df.iter().map(|&v| v as f64 * sc).collect();
            for j in 0..6 {
                cols[j].push(n[j] as f32);
                let (hi, lo) = split_pair(n[j]); // f16 hi+lo ≈ f32 for the DoubleF16 path
                cols_pair[2 * j].push(hi);
                cols_pair[2 * j + 1].push(lo);
            }
            cols[6].push(0.0);
            cols[7].push(0.0);
            let (ax, ay, bx, by, cx, cy) = (n[0], n[1], n[2], n[3], n[4], n[5]);
            let (a2, b2, c2) = (ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy);
            perm.push(
                a2 * ((bx * cy).abs() + (cx * by).abs())
                    + b2 * ((ax * cy).abs() + (cx * ay).abs())
                    + c2 * ((ax * by).abs() + (bx * ay).abs()),
            );
        }
        let names = ["Ax", "Ay", "Bx", "By", "Cx", "Cy", "Dx", "Dy"];
        let names_pair = [
            "Axh", "Axl", "Ayh", "Ayl", "Bxh", "Bxl", "Byh", "Byl", "Cxh", "Cxl", "Cyh", "Cyl",
        ];
        let chunk = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(16384usize);

        // NPU passes, run once each (timed: warmup + best-of) and reused across configs:
        //   det16  — plain f16 (the fast filter)
        //   det_df — DOUBLE-f16 (f16+f16 ≈ f32), pure NPU
        let bench = |f: &mut dyn FnMut() -> Vec<f32>, runs: usize| -> (Vec<f32>, f64) {
            let out = f(); // warmup + captured result
            let mut best = f64::INFINITY;
            for _ in 0..runs {
                let t = std::time::Instant::now();
                let _ = std::hint::black_box(f());
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            (out, best)
        };
        let mut e16 = CoremlExecutable::compile_with_units(
            build_graph(chunk, true),
            ComputeUnits::CpuAndNeuralEngine,
        );
        let (det16, t_f16) = bench(&mut || run_batch(&mut e16, &cols, b, chunk, &names), 8);
        let mut edf = CoremlExecutable::compile_with_units(
            build_graph_df16(chunk),
            ComputeUnits::CpuAndNeuralEngine,
        );
        let (det_df, t_df16) = bench(
            &mut || run_batch(&mut edf, &cols_pair, b, chunk, &names_pair),
            8,
        );
        // host exact-predicate cost (for Filtered fallback + Exact)
        let t_host_all = {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                let mut acc = 0i64;
                for q in &quad {
                    acc += incircle_exact(q[0], q[1], q[2], q[3]) as i64;
                }
                std::hint::black_box(acc);
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            best
        };
        let host_per = t_host_all / b as f64; // ms per exact test

        println!("tile K={k} span={span:.0e}  in-circle tests B={b}  (ANE, {chunk}-wide chunks)");
        println!(
            "raw: ANE f16 {t_f16:.2} ms | ANE double-f16 {t_df16:.2} ms | host i128 all {t_host_all:.2} ms\n"
        );
        println!(
            "{:<38} {:>12} {:>9} {:>13} {:>10} {:>12}",
            "config", "sign-correct", "% correct", "host fallback", "time ms", "M tests/s"
        );
        for cfg in [
            IncircleConfig::fast(),
            IncircleConfig::double_f16(),
            IncircleConfig::filtered_k(2.0),
            IncircleConfig::filtered_k(8.0),
            IncircleConfig::exact(),
        ] {
            let mut agree = 0usize;
            let mut fb = 0usize;
            for i in 0..b {
                let sd = match cfg.mode {
                    Mode::Fast => det16[i].signum() as i32 * (det16[i] != 0.0) as i32,
                    Mode::DoubleF16 => det_df[i].signum() as i32 * (det_df[i] != 0.0) as i32,
                    Mode::Exact => {
                        fb += 1;
                        exact[i]
                    }
                    Mode::Filtered => {
                        // certified: trust f16 only when |det| beats the f16 error bound.
                        if (det16[i] as f64).abs() > cfg.filter_k * F16_EPS * perm[i] {
                            det16[i].signum() as i32 * (det16[i] != 0.0) as i32
                        } else {
                            fb += 1;
                            exact[i]
                        }
                    }
                };
                if sd == exact[i] {
                    agree += 1;
                }
            }
            let lbl = if cfg.mode == Mode::Filtered {
                format!("Filtered k={:<4} (f16 ANE+exact fb)", cfg.filter_k)
            } else {
                cfg.label().to_string()
            };
            // wall-time model: NPU pass (once) + host exact recompute for the fallback subset.
            let ms = match cfg.mode {
                Mode::Fast => t_f16,
                Mode::DoubleF16 => t_df16,
                Mode::Filtered => t_f16 + fb as f64 * host_per,
                Mode::Exact => t_host_all,
            };
            let mtps = b as f64 / ms / 1000.0;
            println!(
                "{:<38} {:>7}/{b} {:>8.3}% {:>7} ({:>4.2}%) {:>9.2} {:>11.1}",
                lbl,
                agree,
                100.0 * agree as f64 / b as f64,
                fb,
                100.0 * fb as f64 / b as f64,
                ms,
                mtps
            );
        }
        println!(
            "\nSpeed↔precision: Fast (f16) fastest but ~0.05% wrong; DoubleF16 (f16+f16) 100% on"
        );
        println!(
            "real data all-ANE, more ops; Filtered 100% guaranteed (NPU + tiny host fallback);"
        );
        println!("Exact all-host. Note: time is dominated by per-chunk CoreML call overhead.");
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn main() {
    imp::run();
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn main() {
    eprintln!("delaunay_incircle_ane requires macOS/iOS (CoreML / Apple Neural Engine)");
}
