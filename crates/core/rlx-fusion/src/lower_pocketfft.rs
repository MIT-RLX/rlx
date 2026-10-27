// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lower `Op::Fft` with [`FftAlgo::PocketFft`] into elementwise passes, so a
//! backend without a pocketfft kernel (every GPU) computes the **same bits** as
//! the CPU row kernel.
//!
//! The plan — factors, twiddles, Bluestein chirps — comes from
//! `rlx_pocketfft::engine` ([`C2cPlan::spec`]), rounded exactly as the engine
//! uses it. Each FFTPACK pass becomes slices of a `[B, l1, ip, ido]` view, the
//! pass's butterfly as `Binary` / `Neg` / `Fma` nodes over whole `[B, l1, ido]`
//! tensors (one node per scalar operation of the engine, vectorized over every
//! `(k, i)`), and a concat back to `[B, ip, l1, ido]`. Twiddles skip `i = 0`
//! exactly as pocketfft does (the untwiddled column is concatenated in), which
//! keeps signed zeros. `Contraction::None` emits `mul` + `add` instead of `Fma`.
//!
//! Exactness rests on the backend rounding each `Binary` and `Fma` as IEEE does
//! (no contraction of a `mul` feeding an `add`, a single-rounding `Fma`) —
//! measured for Metal, MLX and wgpu by `tests/elementwise_rounding_probe.rs`.
//!
//! Declines (the node stays, and legalization reports it): `C64` input, dynamic
//! shapes, and plans with a generic factor above [`MAX_GENERIC_FACTOR`] (its
//! pass is `O(ip²)` nodes).

// pocketfft's literals are written to 34 digits; copying them verbatim is what
// guarantees the same rounding (a 16-digit spelling of 0.309… rounded differently).
#![allow(clippy::excessive_precision)]

use std::collections::HashMap;

use crate::rewriter::{MatchRewrite, RewriteCtx};
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::*;
use rlx_pocketfft::engine::{C2cPlan, C2cSpec, Contraction, Cx, Options, PassSpec, Twiddles};

/// The largest generic (non 2/3/4/5/7/8/11) factor lowered; its pass emits about
/// `2·ip²` nodes.
pub const MAX_GENERIC_FACTOR: usize = 64;

pub struct LowerPocketFft;

fn options(algo: FftAlgo) -> Option<Options> {
    let FftAlgo::PocketFft {
        contraction,
        twiddles,
    } = algo
    else {
        return None;
    };
    Some(Options {
        contraction: match contraction {
            FftContraction::Platform => Contraction::Platform,
            FftContraction::Clang => Contraction::Clang,
            FftContraction::None => Contraction::None,
        },
        twiddles: match twiddles {
            FftTwiddles::Platform => Twiddles::Platform,
            FftTwiddles::AppleSinCos => Twiddles::AppleSinCos,
            FftTwiddles::Separate => Twiddles::Separate,
        },
    })
}

/// The plan as `f64` (exact for both element types).
fn spec(n: usize, dt: DType, opt: Options) -> C2cSpec<f64> {
    fn up<T: Copy + Into<f64>>(v: &[Cx<T>]) -> Vec<Cx<f64>> {
        v.iter().map(|c| Cx::new(c.r.into(), c.i.into())).collect()
    }
    fn up_p<T: Copy + Into<f64>>(p: &[PassSpec<T>]) -> Vec<PassSpec<f64>> {
        p.iter()
            .map(|p| PassSpec {
                ip: p.ip,
                l1: p.l1,
                ido: p.ido,
                tw: up(&p.tw),
                tws: up(&p.tws),
            })
            .collect()
    }
    fn up_s<T: Copy + Into<f64>>(s: C2cSpec<T>) -> C2cSpec<f64> {
        match s {
            C2cSpec::Pack(v) => C2cSpec::Pack(up_p(&v)),
            C2cSpec::Blue { n2, bk, bkf, plan } => C2cSpec::Blue {
                n2,
                bk: up(&bk),
                bkf: up(&bkf),
                plan: up_p(&plan),
            },
        }
    }
    match dt {
        DType::F32 => up_s(C2cPlan::<f32>::new(n, opt).spec()),
        _ => C2cPlan::<f64>::new(n, opt).spec(),
    }
}

fn max_generic(s: &C2cSpec<f64>) -> usize {
    let passes = match s {
        C2cSpec::Pack(p) => p,
        C2cSpec::Blue { plan, .. } => plan,
    };
    passes
        .iter()
        .map(|p| {
            if matches!(p.ip, 2 | 3 | 4 | 5 | 7 | 8 | 11) {
                0
            } else {
                p.ip
            }
        })
        .max()
        .unwrap_or(0)
}

impl MatchRewrite for LowerPocketFft {
    fn name(&self) -> &str {
        "lower_pocketfft"
    }

    fn trigger_kinds(&self) -> &[OpKind] {
        &[OpKind::FftPocketFft]
    }

    fn matches(&self, _graph: &Graph, node: &Node) -> bool {
        node.op.kind() == OpKind::FftPocketFft
            && matches!(node.shape.dtype(), DType::F32 | DType::F64)
            && node.shape.dims().iter().all(|d| d.is_static())
    }

    fn rewrite(&self, node: &Node, ctx: &mut RewriteCtx) -> Option<NodeId> {
        let Op::Fft {
            inverse,
            norm,
            algo,
        } = node.op
        else {
            return None;
        };
        let opt = options(algo)?;
        let dt = node.shape.dtype();
        let dims: Vec<usize> = node
            .shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect();
        let n2x = *dims.last()?;
        let n = n2x / 2;
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let plan = spec(n, dt, opt);
        if max_generic(&plan) > MAX_GENERIC_FACTOR {
            return None;
        }
        // pocketfft's `fct`: the norm scale rounded to the element type
        let scale = match (norm, inverse) {
            (FftNorm::Ortho, _) => rlx_pocketfft::norm_fct(1, n),
            (FftNorm::Forward, true) => rlx_pocketfft::norm_fct(2, n),
            _ => 1.0,
        };
        let fct = if dt == DType::F32 {
            scale as f32 as f64
        } else {
            scale
        };
        let x = ctx.input(0);
        let mut b = Builder {
            ctx,
            dt,
            fused: opt.contraction.fused(),
            consts: HashMap::new(),
        };
        let flat = b.reshape(x, &[rows, n2x]);
        let v = V {
            r: b.narrow(flat, 1, 0, n),
            i: b.narrow(flat, 1, n, n),
        };
        let y = b.c2c(v, &plan, !inverse, fct, rows, n);
        let out = b.concat(&[y.r, y.i], 1);
        Some(b.reshape(out, &dims))
    }
}

#[derive(Clone, Copy)]
struct V {
    r: NodeId,
    i: NodeId,
}

struct Builder<'a, 'b> {
    ctx: &'a mut RewriteCtx<'b>,
    dt: DType,
    fused: bool,
    /// `(value bits, dims)` → an expanded constant, so repeated coefficients
    /// share one node.
    consts: HashMap<(Vec<u64>, Vec<usize>), NodeId>,
}

impl Builder<'_, '_> {
    fn dims(&self, x: NodeId) -> Vec<usize> {
        self.ctx
            .out
            .node(x)
            .shape
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect()
    }
    fn shape(&self, d: &[usize]) -> Shape {
        Shape::new(d, self.dt)
    }
    fn bin(&mut self, op: BinaryOp, a: NodeId, b: NodeId) -> NodeId {
        let s = self.ctx.out.node(a).shape.clone();
        self.ctx.emit(Op::Binary(op), vec![a, b], s)
    }
    fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinaryOp::Add, a, b)
    }
    fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinaryOp::Sub, a, b)
    }
    fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.bin(BinaryOp::Mul, a, b)
    }
    fn neg(&mut self, a: NodeId) -> NodeId {
        let s = self.ctx.out.node(a).shape.clone();
        self.ctx.emit(Op::Activation(Activation::Neg), vec![a], s)
    }
    /// `a*b + c` with one rounding (two under `Contraction::None`).
    fn fma(&mut self, a: NodeId, b: NodeId, c: NodeId) -> NodeId {
        if self.fused {
            let s = self.ctx.out.node(a).shape.clone();
            self.ctx.emit(Op::Fma, vec![a, b, c], s)
        } else {
            let p = self.mul(a, b);
            self.add(p, c)
        }
    }
    fn reshape(&mut self, x: NodeId, d: &[usize]) -> NodeId {
        let s = self.shape(d);
        self.ctx.emit(
            Op::Reshape {
                new_shape: d.iter().map(|&v| v as i64).collect(),
            },
            vec![x],
            s,
        )
    }
    fn narrow(&mut self, x: NodeId, axis: usize, start: usize, len: usize) -> NodeId {
        let mut d = self.dims(x);
        d[axis] = len;
        let s = self.shape(&d);
        self.ctx.emit(Op::Narrow { axis, start, len }, vec![x], s)
    }
    fn concat(&mut self, xs: &[NodeId], axis: usize) -> NodeId {
        if xs.len() == 1 {
            return xs[0];
        }
        let mut d = self.dims(xs[0]);
        d[axis] = xs.iter().map(|&x| self.dims(x)[axis]).sum();
        let s = self.shape(&d);
        self.ctx.emit(Op::Concat { axis }, xs.to_vec(), s)
    }
    /// A constant of `cdims` (values in `f64`, stored as the element type),
    /// expanded to `target`.
    fn konst(&mut self, vals: &[f64], cdims: &[usize], target: &[usize]) -> NodeId {
        let key = (
            vals.iter()
                .map(|v| v.to_bits())
                .chain(cdims.iter().map(|&d| d as u64))
                .collect(),
            target.to_vec(),
        );
        if let Some(&id) = self.consts.get(&key) {
            return id;
        }
        let data: Vec<u8> = match self.dt {
            DType::F32 => vals
                .iter()
                .flat_map(|&v| (v as f32).to_le_bytes())
                .collect(),
            _ => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        let s = self.shape(cdims);
        let mut id = self.ctx.emit(Op::Constant { data }, vec![], s);
        if cdims != target {
            let s = self.shape(target);
            id = self.ctx.emit(
                Op::Expand {
                    target_shape: target.iter().map(|&v| v as i64).collect(),
                },
                vec![id],
                s,
            );
        }
        self.consts.insert(key, id);
        id
    }
    /// The scalar `v` broadcast to `like`'s shape.
    fn scalar(&mut self, v: f64, like: NodeId) -> NodeId {
        let t = self.dims(like);
        let ones = vec![1; t.len()];
        self.konst(&[v], &ones, &t)
    }
    /// `x * s` (a scalar coefficient, unfused).
    fn muls(&mut self, x: NodeId, s: f64) -> NodeId {
        let k = self.scalar(s, x);
        self.mul(x, k)
    }

    // ── complex helpers (the engine's `cmplx` operators) ──

    fn vadd(&mut self, a: V, b: V) -> V {
        V {
            r: self.add(a.r, b.r),
            i: self.add(a.i, b.i),
        }
    }
    fn vsub(&mut self, a: V, b: V) -> V {
        V {
            r: self.sub(a.r, b.r),
            i: self.sub(a.i, b.i),
        }
    }
    fn vscale(&mut self, a: V, s: f64) -> V {
        V {
            r: self.muls(a.r, s),
            i: self.muls(a.i, s),
        }
    }
    fn vnarrow(&mut self, a: V, axis: usize, start: usize, len: usize) -> V {
        V {
            r: self.narrow(a.r, axis, start, len),
            i: self.narrow(a.i, axis, start, len),
        }
    }
    fn vreshape(&mut self, a: V, d: &[usize]) -> V {
        V {
            r: self.reshape(a.r, d),
            i: self.reshape(a.i, d),
        }
    }
    fn vconcat(&mut self, xs: &[V], axis: usize) -> V {
        let r: Vec<NodeId> = xs.iter().map(|v| v.r).collect();
        let i: Vec<NodeId> = xs.iter().map(|v| v.i).collect();
        V {
            r: self.concat(&r, axis),
            i: self.concat(&i, axis),
        }
    }
    /// `special_mul<fwd>(v, w)`.
    fn smul(&mut self, fwd: bool, v: V, w: V) -> V {
        if fwd {
            let p = self.mul(v.i, w.i);
            let r = self.fma(v.r, w.r, p);
            let q = self.mul(v.r, w.i);
            let q = self.neg(q);
            let i = self.fma(v.i, w.r, q);
            V { r, i }
        } else {
            let p = self.mul(v.i, w.i);
            let p = self.neg(p);
            let r = self.fma(v.r, w.r, p);
            let q = self.mul(v.i, w.r);
            let i = self.fma(v.r, w.i, q);
            V { r, i }
        }
    }
    fn rotx90(&mut self, fwd: bool, a: V) -> V {
        if fwd {
            V {
                r: a.i,
                i: self.neg(a.r),
            }
        } else {
            V {
                r: self.neg(a.i),
                i: a.r,
            }
        }
    }
    fn rotx45(&mut self, fwd: bool, a: V) -> V {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let (r, i) = if fwd {
            (self.add(a.r, a.i), self.sub(a.i, a.r))
        } else {
            (self.sub(a.r, a.i), self.add(a.i, a.r))
        };
        V {
            r: self.muls(r, h),
            i: self.muls(i, h),
        }
    }
    fn rotx135(&mut self, fwd: bool, a: V) -> V {
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let nr = self.neg(a.r);
        let (r, i) = if fwd {
            (self.sub(a.i, a.r), self.sub(nr, a.i))
        } else {
            (self.sub(nr, a.i), self.sub(a.r, a.i))
        };
        V {
            r: self.muls(r, h),
            i: self.muls(i, h),
        }
    }
    /// `c0*v0 + c1*v1 + …` as clang contracts it: `fma(c0, v0, c1*v1)`, then
    /// `fma(ck, vk, acc)`.
    fn chain(&mut self, terms: &[(f64, NodeId)]) -> NodeId {
        let k1 = self.scalar(terms[1].0, terms[1].1);
        let p = self.mul(k1, terms[1].1);
        let k0 = self.scalar(terms[0].0, terms[0].1);
        let mut acc = self.fma(k0, terms[0].1, p);
        for &(c, v) in &terms[2..] {
            let k = self.scalar(c, v);
            acc = self.fma(k, v, acc);
        }
        acc
    }
    /// `fma(c_last, v_last, … fma(c0, v0, base))`.
    fn chain_on(&mut self, base: NodeId, terms: &[(f64, NodeId)]) -> NodeId {
        let mut acc = base;
        for &(c, v) in terms {
            let k = self.scalar(c, v);
            acc = self.fma(k, v, acc);
        }
        acc
    }

    /// `u` twiddled by pass twiddle row `x` for `i ≥ 1` (`i = 0` untouched).
    fn tw(&mut self, fwd: bool, u: V, p: &PassSpec<f64>, x: usize, b: usize) -> V {
        let ido = p.ido;
        if ido == 1 {
            return u;
        }
        let w = &p.tw[x * (ido - 1)..(x + 1) * (ido - 1)];
        let target = [b, p.l1, ido - 1];
        let wr: Vec<f64> = w.iter().map(|c| c.r).collect();
        let wi: Vec<f64> = w.iter().map(|c| c.i).collect();
        let wv = V {
            r: self.konst(&wr, &[1, 1, ido - 1], &target),
            i: self.konst(&wi, &[1, 1, ido - 1], &target),
        };
        let u0 = self.vnarrow(u, 2, 0, 1);
        let u1 = self.vnarrow(u, 2, 1, ido - 1);
        let t = self.smul(fwd, u1, wv);
        self.vconcat(&[u0, t], 2)
    }

    /// `cfftp::exec(c, fct, fwd)` on `[b, n]`.
    fn c2c(&mut self, v: V, plan: &C2cSpec<f64>, fwd: bool, fct: f64, b: usize, n: usize) -> V {
        match plan {
            C2cSpec::Pack(passes) => self.pack(v, passes, fwd, fct, b, n),
            C2cSpec::Blue { n2, bk, bkf, plan } => self.blue(v, *n2, bk, bkf, plan, fwd, fct, b, n),
        }
    }

    fn pack(
        &mut self,
        mut v: V,
        passes: &[PassSpec<f64>],
        fwd: bool,
        fct: f64,
        b: usize,
        n: usize,
    ) -> V {
        for p in passes {
            let x = self.vreshape(v, &[b, p.l1, p.ip, p.ido]);
            let cc: Vec<V> = (0..p.ip)
                .map(|j| {
                    let s = self.vnarrow(x, 2, j, 1);
                    self.vreshape(s, &[b, p.l1, p.ido])
                })
                .collect();
            let ch = match p.ip {
                2 => self.pass2(fwd, &cc, p, b),
                3 => self.pass3(fwd, &cc, p, b),
                4 => self.pass4(fwd, &cc, p, b),
                5 => self.pass5(fwd, &cc, p, b),
                7 => self.pass7(fwd, &cc, p, b),
                8 => self.pass8(fwd, &cc, p, b),
                11 => self.pass11(fwd, &cc, p, b),
                _ => self.passg(fwd, &cc, p, b),
            };
            let parts: Vec<V> = ch
                .iter()
                .map(|c| self.vreshape(*c, &[b, 1, p.l1, p.ido]))
                .collect();
            let y = self.vconcat(&parts, 1);
            v = self.vreshape(y, &[b, n]);
        }
        if fct != 1.0 {
            v = self.vscale(v, fct);
        }
        v
    }

    fn pass2(&mut self, fwd: bool, cc: &[V], p: &PassSpec<f64>, b: usize) -> Vec<V> {
        let c0 = self.vadd(cc[0], cc[1]);
        let d = self.vsub(cc[0], cc[1]);
        let c1 = self.tw(fwd, d, p, 0, b);
        vec![c0, c1]
    }

    fn pass3(&mut self, fwd: bool, cc: &[V], p: &PassSpec<f64>, b: usize) -> Vec<V> {
        let tw1r = -0.5;
        let tw1i =
            if fwd { -1.0 } else { 1.0 } * 0.866_025_403_784_438_646_763_723_170_752_936_2_f64;
        let tw1i = self.cast(tw1i);
        let t0 = cc[0];
        let t1 = self.vadd(cc[1], cc[2]);
        let t2 = self.vsub(cc[1], cc[2]);
        let c0 = self.vadd(t0, t1);
        let s = self.vscale(t1, tw1r);
        let ca = self.vadd(t0, s);
        let ni = self.neg(t2.i);
        let cb = V {
            r: self.muls(ni, tw1i),
            i: self.muls(t2.r, tw1i),
        };
        let a = self.vadd(ca, cb);
        let d = self.vsub(ca, cb);
        let c1 = self.tw(fwd, a, p, 0, b);
        let c2 = self.tw(fwd, d, p, 1, b);
        vec![c0, c1, c2]
    }

    fn pass4(&mut self, fwd: bool, cc: &[V], p: &PassSpec<f64>, b: usize) -> Vec<V> {
        let t2 = self.vadd(cc[0], cc[2]);
        let t1 = self.vsub(cc[0], cc[2]);
        let t3 = self.vadd(cc[1], cc[3]);
        let t4 = self.vsub(cc[1], cc[3]);
        let t4 = self.rotx90(fwd, t4);
        let c0 = self.vadd(t2, t3);
        let u1 = self.vadd(t1, t4);
        let u2 = self.vsub(t2, t3);
        let u3 = self.vsub(t1, t4);
        let c1 = self.tw(fwd, u1, p, 0, b);
        let c2 = self.tw(fwd, u2, p, 1, b);
        let c3 = self.tw(fwd, u3, p, 2, b);
        vec![c0, c1, c2, c3]
    }

    /// A literal as `T0(…L)` rounds it (to the element type).
    fn cast(&self, v: f64) -> f64 {
        if self.dt == DType::F32 {
            v as f32 as f64
        } else {
            v
        }
    }

    fn pass5(&mut self, fwd: bool, cc: &[V], p: &PassSpec<f64>, b: usize) -> Vec<V> {
        let s = if fwd { -1.0 } else { 1.0 };
        let tw1r = self.cast(0.309_016_994_374_947_424_102_293_417_182_819_1);
        let tw1i = s * self.cast(0.951_056_516_295_153_572_116_439_333_379_382_1);
        let tw2r = self.cast(-0.809_016_994_374_947_424_102_293_417_182_819_1);
        let tw2i = s * self.cast(0.587_785_252_292_473_129_168_705_954_639_072_8);
        let t0 = cc[0];
        let t1 = self.vadd(cc[1], cc[4]);
        let t4 = self.vsub(cc[1], cc[4]);
        let t2 = self.vadd(cc[2], cc[3]);
        let t3 = self.vsub(cc[2], cc[3]);
        let r = self.add(t0.r, t1.r);
        let i = self.add(t0.i, t1.i);
        let mut ch = vec![
            V {
                r: self.add(r, t2.r),
                i: self.add(i, t2.i)
            };
            5
        ];
        for (u1, u2, twar, twbr, twai, twbi) in [
            (1, 4, tw1r, tw2r, tw1i, tw2i),
            (2, 3, tw2r, tw1r, tw2i, -tw1i),
        ] {
            let ca = V {
                r: self.chain_on(t0.r, &[(twar, t1.r), (twbr, t2.r)]),
                i: self.chain_on(t0.i, &[(twar, t1.i), (twbr, t2.i)]),
            };
            let cbr = self.chain(&[(twai, t4.i), (twbi, t3.i)]);
            let cb = V {
                r: self.neg(cbr),
                i: self.chain(&[(twai, t4.r), (twbi, t3.r)]),
            };
            let a = self.vadd(ca, cb);
            let d = self.vsub(ca, cb);
            ch[u1] = self.tw(fwd, a, p, u1 - 1, b);
            ch[u2] = self.tw(fwd, d, p, u2 - 1, b);
        }
        ch
    }

    fn pass7(&mut self, fwd: bool, cc: &[V], p: &PassSpec<f64>, b: usize) -> Vec<V> {
        let s = if fwd { -1.0 } else { 1.0 };
        let tw1r = self.cast(0.623_489_801_858_733_530_525_004_884_004_239_8);
        let tw1i = s * self.cast(0.781_831_482_468_029_808_708_444_526_674_057_8);
        let tw2r = self.cast(-0.222_520_933_956_314_404_288_902_564_496_794_8);
        let tw2i = s * self.cast(0.974_927_912_181_823_607_018_131_682_993_931_2);
        let tw3r = self.cast(-0.900_968_867_902_419_126_236_102_319_507_445_1);
        let tw3i = s * self.cast(0.433_883_739_117_558_120_475_768_332_848_359);
        let t1 = cc[0];
        let t2 = self.vadd(cc[1], cc[6]);
        let t7 = self.vsub(cc[1], cc[6]);
        let t3 = self.vadd(cc[2], cc[5]);
        let t6 = self.vsub(cc[2], cc[5]);
        let t4 = self.vadd(cc[3], cc[4]);
        let t5 = self.vsub(cc[3], cc[4]);
        let mut r = self.add(t1.r, t2.r);
        let mut i = self.add(t1.i, t2.i);
        r = self.add(r, t3.r);
        i = self.add(i, t3.i);
        let mut ch = vec![
            V {
                r: self.add(r, t4.r),
                i: self.add(i, t4.i)
            };
            7
        ];
        let steps = [
            (1, 6, [tw1r, tw2r, tw3r], [tw1i, tw2i, tw3i]),
            (2, 5, [tw2r, tw3r, tw1r], [tw2i, -tw3i, -tw1i]),
            (3, 4, [tw3r, tw1r, tw2r], [tw3i, -tw1i, tw2i]),
        ];
        for (u1, u2, x, y) in steps {
            let ca = V {
                r: self.chain_on(t1.r, &[(x[0], t2.r), (x[1], t3.r), (x[2], t4.r)]),
                i: self.chain_on(t1.i, &[(x[0], t2.i), (x[1], t3.i), (x[2], t4.i)]),
            };
            let cbr = self.chain(&[(y[0], t7.i), (y[1], t6.i), (y[2], t5.i)]);
            let cb = V {
                r: self.neg(cbr),
                i: self.chain(&[(y[0], t7.r), (y[1], t6.r), (y[2], t5.r)]),
            };
            let da = self.vadd(ca, cb);
            let db = self.vsub(ca, cb);
            ch[u1] = self.tw(fwd, da, p, u1 - 1, b);
            ch[u2] = self.tw(fwd, db, p, u2 - 1, b);
        }
        ch
    }

    fn pass8(&mut self, fwd: bool, cc: &[V], p: &PassSpec<f64>, b: usize) -> Vec<V> {
        let a1 = self.vadd(cc[1], cc[5]);
        let a5 = self.vsub(cc[1], cc[5]);
        let a3 = self.vadd(cc[3], cc[7]);
        let a7 = self.vsub(cc[3], cc[7]);
        let a7 = self.rotx90(fwd, a7);
        let (a1, a3) = (self.vadd(a1, a3), self.vsub(a1, a3));
        let a3 = self.rotx90(fwd, a3);
        let (a5, a7) = (self.vadd(a5, a7), self.vsub(a5, a7));
        let a5 = self.rotx45(fwd, a5);
        let a7 = self.rotx135(fwd, a7);
        let a0 = self.vadd(cc[0], cc[4]);
        let a4 = self.vsub(cc[0], cc[4]);
        let a2 = self.vadd(cc[2], cc[6]);
        let a6 = self.vsub(cc[2], cc[6]);
        let (a0, a2) = (self.vadd(a0, a2), self.vsub(a0, a2));
        let mut ch = vec![self.vadd(a0, a1); 8];
        let u = self.vsub(a0, a1);
        ch[4] = self.tw(fwd, u, p, 3, b);
        let u = self.vadd(a2, a3);
        ch[2] = self.tw(fwd, u, p, 1, b);
        let u = self.vsub(a2, a3);
        ch[6] = self.tw(fwd, u, p, 5, b);
        let a6 = self.rotx90(fwd, a6);
        let (a4, a6) = (self.vadd(a4, a6), self.vsub(a4, a6));
        let u = self.vadd(a4, a5);
        ch[1] = self.tw(fwd, u, p, 0, b);
        let u = self.vsub(a4, a5);
        ch[5] = self.tw(fwd, u, p, 4, b);
        let u = self.vadd(a6, a7);
        ch[3] = self.tw(fwd, u, p, 2, b);
        let u = self.vsub(a6, a7);
        ch[7] = self.tw(fwd, u, p, 6, b);
        ch
    }

    fn pass11(&mut self, fwd: bool, cc: &[V], p: &PassSpec<f64>, b: usize) -> Vec<V> {
        let s = if fwd { -1.0 } else { 1.0 };
        let r: Vec<f64> = [
            0.841_253_532_831_181_168_861_811_648_919_367_7,
            0.415_415_013_001_886_425_529_274_149_229_623_2,
            -0.142_314_838_273_285_140_443_792_668_616_369_7,
            -0.654_860_733_945_285_064_056_925_072_466_293_6,
            -0.959_492_973_614_497_389_890_368_057_066_327_7,
        ]
        .iter()
        .map(|&v| self.cast(v))
        .collect();
        let im: Vec<f64> = [
            0.540_640_817_455_597_582_107_635_954_318_691_7,
            0.909_631_995_354_518_371_411_715_383_079_028_5,
            0.989_821_441_880_932_732_376_092_037_776_718_8,
            0.755_749_574_354_258_283_774_035_843_972_344_4,
            0.281_732_556_841_429_697_711_417_915_346_616_9,
        ]
        .iter()
        .map(|&v| s * self.cast(v))
        .collect();
        let (tw1r, tw2r, tw3r, tw4r, tw5r) = (r[0], r[1], r[2], r[3], r[4]);
        let (tw1i, tw2i, tw3i, tw4i, tw5i) = (im[0], im[1], im[2], im[3], im[4]);
        let steps = [
            (
                1,
                10,
                [tw1r, tw2r, tw3r, tw4r, tw5r],
                [tw1i, tw2i, tw3i, tw4i, tw5i],
            ),
            (
                2,
                9,
                [tw2r, tw4r, tw5r, tw3r, tw1r],
                [tw2i, tw4i, -tw5i, -tw3i, -tw1i],
            ),
            (
                3,
                8,
                [tw3r, tw5r, tw2r, tw1r, tw4r],
                [tw3i, -tw5i, -tw2i, tw1i, tw4i],
            ),
            (
                4,
                7,
                [tw4r, tw3r, tw1r, tw5r, tw2r],
                [tw4i, -tw3i, tw1i, tw5i, -tw2i],
            ),
            (
                5,
                6,
                [tw5r, tw1r, tw4r, tw2r, tw3r],
                [tw5i, -tw1i, tw4i, -tw2i, tw3i],
            ),
        ];
        let t1 = cc[0];
        let mut tp = vec![t1];
        let mut tm = Vec::new();
        for k in 1..=5 {
            tp.push(self.vadd(cc[k], cc[11 - k]));
            tm.push(self.vsub(cc[k], cc[11 - k]));
        }
        // tp = [t1, t2..t6]; tm = [t11, t10, t9, t8, t7]
        let mut r0 = t1.r;
        let mut i0 = t1.i;
        for t in &tp[1..] {
            r0 = self.add(r0, t.r);
            i0 = self.add(i0, t.i);
        }
        let mut ch = vec![V { r: r0, i: i0 }; 11];
        for (u1, u2, x, y) in steps {
            // complex operators: never fused
            let mut ca = t1;
            for k in 0..5 {
                let s = self.vscale(tp[k + 1], x[k]);
                ca = self.vadd(ca, s);
            }
            let ti: Vec<(f64, NodeId)> = (0..5).map(|k| (y[k], tm[k].i)).collect();
            let tr: Vec<(f64, NodeId)> = (0..5).map(|k| (y[k], tm[k].r)).collect();
            let cbr = self.chain(&ti);
            let cb = V {
                r: self.neg(cbr),
                i: self.chain(&tr),
            };
            let da = self.vadd(ca, cb);
            let db = self.vsub(ca, cb);
            ch[u1] = self.tw(fwd, da, p, u1 - 1, b);
            ch[u2] = self.tw(fwd, db, p, u2 - 1, b);
        }
        ch
    }

    /// `passg`: `cc[j]` are the `[b, l1, ido]` slices; returns the output slices.
    fn passg(&mut self, fwd: bool, cc: &[V], p: &PassSpec<f64>, b: usize) -> Vec<V> {
        let ip = p.ip;
        let ipph = ip.div_ceil(2);
        let wal: Vec<Cx<f64>> = (0..ip)
            .map(|i| {
                if i == 0 {
                    Cx::new(1.0, 0.0)
                } else {
                    Cx::new(p.tws[i].r, if fwd { -p.tws[i].i } else { p.tws[i].i })
                }
            })
            .collect();
        // ch[j]: the PM'd copies
        let mut ch = cc.to_vec();
        for j in 1..ipph {
            let jc = ip - j;
            ch[j] = self.vadd(cc[j], cc[jc]);
            ch[jc] = self.vsub(cc[j], cc[jc]);
        }
        let mut out = vec![ch[0]; ip];
        let mut acc = ch[0];
        for c in &ch[1..ipph] {
            acc = self.vadd(acc, *c);
        }
        out[0] = acc;
        for l in 1..ipph {
            let lc = ip - l;
            let mut x = V {
                r: self.chain_on(ch[0].r, &[(wal[l].r, ch[1].r), (wal[2 * l].r, ch[2].r)]),
                i: self.chain_on(ch[0].i, &[(wal[l].r, ch[1].i), (wal[2 * l].r, ch[2].i)]),
            };
            let mut y = V {
                r: self.chain(&[(-wal[l].i, ch[ip - 1].i), (-wal[2 * l].i, ch[ip - 2].i)]),
                i: self.chain(&[(wal[l].i, ch[ip - 1].r), (wal[2 * l].i, ch[ip - 2].r)]),
            };
            let mut iwal = 2 * l;
            let (mut j, mut jc) = (3, ip - 3);
            while j + 1 < ipph {
                iwal += l;
                if iwal > ip {
                    iwal -= ip;
                }
                let xw = wal[iwal];
                iwal += l;
                if iwal > ip {
                    iwal -= ip;
                }
                let xw2 = wal[iwal];
                let t = self.chain(&[(xw.r, ch[j].r), (xw2.r, ch[j + 1].r)]);
                x.r = self.add(x.r, t);
                let t = self.chain(&[(xw.r, ch[j].i), (xw2.r, ch[j + 1].i)]);
                x.i = self.add(x.i, t);
                let t = self.chain(&[(xw.i, ch[jc].i), (xw2.i, ch[jc - 1].i)]);
                y.r = self.sub(y.r, t);
                let t = self.chain(&[(xw.i, ch[jc].r), (xw2.i, ch[jc - 1].r)]);
                y.i = self.add(y.i, t);
                j += 2;
                jc -= 2;
            }
            while j < ipph {
                iwal += l;
                if iwal > ip {
                    iwal -= ip;
                }
                let xw = wal[iwal];
                x.r = self.chain_on(x.r, &[(xw.r, ch[j].r)]);
                x.i = self.chain_on(x.i, &[(xw.r, ch[j].i)]);
                y.r = self.chain_on(y.r, &[(-xw.i, ch[jc].i)]);
                y.i = self.chain_on(y.i, &[(xw.i, ch[jc].r)]);
                j += 1;
                jc -= 1;
            }
            out[l] = x;
            out[lc] = y;
        }
        // shuffling and twiddling
        for j in 1..ipph {
            let jc = ip - j;
            let (t1, t2) = (out[j], out[jc]);
            let a = self.vadd(t1, t2);
            let d = self.vsub(t1, t2);
            out[j] = self.tw(fwd, a, p, j - 1, b);
            out[jc] = self.tw(fwd, d, p, jc - 1, b);
        }
        out
    }

    /// `fftblue::fft<fwd>(c, fct)`.
    #[allow(clippy::too_many_arguments)]
    fn blue(
        &mut self,
        c: V,
        n2: usize,
        bk: &[Cx<f64>],
        bkf: &[Cx<f64>],
        plan: &[PassSpec<f64>],
        fwd: bool,
        fct: f64,
        b: usize,
        n: usize,
    ) -> V {
        let table = |s: &mut Self, t: &[Cx<f64>], len: usize| -> V {
            let r: Vec<f64> = t.iter().map(|c| c.r).collect();
            let i: Vec<f64> = t.iter().map(|c| c.i).collect();
            V {
                r: s.konst(&r, &[1, len], &[b, len]),
                i: s.konst(&i, &[1, len], &[b, len]),
            }
        };
        let bkv = table(self, bk, n);
        let mut akf = self.smul(fwd, c, bkv);
        if n2 > n {
            // padding: akf[0]*T0(0), sign included
            let a0 = self.vnarrow(akf, 1, 0, 1);
            let z = self.vscale(a0, 0.0);
            let t = [b, n2 - n];
            let pad = V {
                r: self.ctx.emit(
                    Op::Expand {
                        target_shape: t.iter().map(|&v| v as i64).collect(),
                    },
                    vec![z.r],
                    Shape::new(&t, self.dt),
                ),
                i: self.ctx.emit(
                    Op::Expand {
                        target_shape: t.iter().map(|&v| v as i64).collect(),
                    },
                    vec![z.i],
                    Shape::new(&t, self.dt),
                ),
            };
            akf = self.vconcat(&[akf, pad], 1);
        }
        akf = self.pack(akf, plan, true, 1.0, b, n2);
        let full: Vec<Cx<f64>> = (0..n2).map(|m| bkf[m.min(n2 - m)]).collect();
        let bkfv = table(self, &full, n2);
        akf = self.smul(!fwd, akf, bkfv);
        akf = self.pack(akf, plan, false, 1.0, b, n2);
        let head = self.vnarrow(akf, 1, 0, n);
        let out = self.smul(fwd, head, bkv);
        if fct != 1.0 {
            self.vscale(out, fct)
        } else {
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_every_pocketfft_node_and_leaves_native() {
        let mut g = Graph::new("pf");
        let x = g.input("x", Shape::new(&[3, 2 * 60], DType::F32));
        let y = g.fft_algo(x, false, FftNorm::Backward, FftAlgo::SCIPY);
        let z = g.fft(y, true);
        g.set_outputs(vec![z]);
        let out = crate::rewriter::apply_match_rewrite(&LowerPocketFft, g).graph;
        let kinds: Vec<OpKind> = out.nodes().iter().map(|n| n.op.kind()).collect();
        assert!(!kinds.contains(&OpKind::FftPocketFft));
        assert_eq!(kinds.iter().filter(|k| **k == OpKind::Fft).count(), 1);
        // SciPy's build contracts `a*b + c` only where its compiler does
        // (Apple clang); the lowering mirrors that, so `Fma` appears on Apple
        // targets and `mul` + `add` elsewhere.
        assert_eq!(
            kinds.contains(&OpKind::Fma),
            Contraction::Platform.fused(),
            "Fma presence must follow the platform contraction rule"
        );
    }
}
