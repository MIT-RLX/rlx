// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! MoE FFN placeholder for Gemma 4 recipes (dense path ignores this).

/// Routed-expert FFN stage (stub — wired when `rlx-flow` MoE kernels land).
#[derive(Debug, Clone)]
pub struct MoeFfnStage {
    pub prefix: String,
    pub num_experts: usize,
    pub top_k: usize,
    pub n_embd: usize,
    pub n_ff: usize,
}

impl MoeFfnStage {
    pub fn hf(
        prefix: impl Into<String>,
        num_experts: usize,
        top_k: usize,
        n_embd: usize,
        n_ff: usize,
    ) -> Self {
        Self {
            prefix: prefix.into(),
            num_experts,
            top_k,
            n_embd,
            n_ff,
        }
    }
}

/// A Qwen3-family sparse MoE MLP (Qwen3-30B-A3B, Qwen3-VL-MoE): a router
/// (`mlp.gate`, softmax over `num_experts`), the `top_k` experts each token
/// picks (weights renormalized to sum to 1 when `norm_topk_prob`), each a
/// SwiGLU of `intermediate` width, summed by weight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3Moe {
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate: usize,
    pub norm_topk_prob: bool,
}

/// A layer's MLP: the dense SwiGLU's three linears, or its experts.
pub(crate) enum Mlp<L> {
    Dense(L, L, L),
    Experts(MoeWeights),
}

/// A MoE layer's weights as graph params: the router `[hidden, experts]`,
/// the experts' gate-and-up `[experts, hidden, 2·intermediate]` (gate first)
/// and down `[experts, intermediate, hidden]` — `GroupedMatMul`'s layout.
/// With `packed`, gate-and-up and down are instead quant banks of `[out, in]`
/// slabs (`[2·intermediate, hidden]` and `[hidden, intermediate]` per expert)
/// for `DequantGroupedMatMul` in that scheme.
pub(crate) struct MoeWeights {
    router: rlx_ir::HirNodeId,
    gate_up: rlx_ir::HirNodeId,
    down: rlx_ir::HirNodeId,
    packed: Option<rlx_ir::quant::QuantScheme>,
}

/// The experts as packed banks, when the source hands their linears over
/// packed (an FP8 safetensors or K-quant checkpoint, one tensor per expert).
/// Each expert's gate and up rows are concatenated into one `[2I, H]` slab —
/// a packed row is whole blocks, so rows concatenate byte for byte. `None`
/// when expert 0's gate is not packed (or absent: a fused checkpoint), which
/// leaves every tensor for the f32 path.
fn load_moe_packed(
    ctx: &mut crate::context::FlowCtx<'_>,
    lp: &str,
    moe: &Qwen3Moe,
    hidden: usize,
) -> anyhow::Result<
    Option<(
        rlx_ir::HirNodeId,
        rlx_ir::HirNodeId,
        rlx_ir::quant::QuantScheme,
    )>,
> {
    use rlx_ir::{DType, Shape};
    let (e, i) = (moe.num_experts, moe.intermediate);
    let key = |x: usize, p: &str| format!("{lp}.mlp.experts.{x}.{p}_proj.weight");
    let Some(first) = ctx.weights.take_packed(&key(0, "gate")).ok().flatten() else {
        return Ok(None);
    };
    let scheme = first.scheme;
    let mut first = Some(first);
    let (mut gu, mut dn) = (Vec::new(), Vec::new());
    for x in 0..e {
        let mut take = |p: &str, out: usize, inp: usize| -> anyhow::Result<Vec<u8>> {
            let k = key(x, p);
            let w = match (p, x, first.take()) {
                ("gate", 0, Some(w)) => w,
                _ => ctx.weights.take_packed(&k)?.ok_or_else(|| {
                    anyhow::anyhow!("{k}: expert 0's gate is packed but this linear is not")
                })?,
            };
            if w.scheme != scheme || (w.out_dim, w.in_dim) != (out, inp) {
                anyhow::bail!(
                    "{k}: {} [{}, {}], expected {scheme} [{out}, {inp}]",
                    w.scheme,
                    w.out_dim,
                    w.in_dim
                );
            }
            Ok(w.w_q)
        };
        gu.extend(take("gate", i, hidden)?);
        gu.extend(take("up", i, hidden)?);
        dn.extend(take("down", hidden, i)?);
    }
    let mut param = |name: String, bytes: Vec<u8>| {
        let id = ctx
            .hir()
            .param(&name, Shape::new(&[bytes.len()], DType::U8));
        ctx.state.typed_params.push((name, bytes, DType::U8));
        id
    };
    let gate_up = param(format!("{lp}.mlp.experts.gate_up.packed"), gu);
    let down = param(format!("{lp}.mlp.experts.down.packed"), dn);
    Ok(Some((gate_up, down, scheme)))
}

/// `x` (`[e, a, b]`) with each `[a, b]` slab transposed to `[b, a]`.
fn transpose_slabs(x: &[f32], e: usize, a: usize, b: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for k in 0..e {
        let (src, dst) = (
            &x[k * a * b..(k + 1) * a * b],
            &mut out[k * a * b..(k + 1) * a * b],
        );
        for i in 0..a {
            for j in 0..b {
                dst[j * a + i] = src[i * b + j];
            }
        }
    }
    out
}

/// Layer `lp`'s MoE weights, in either checkpoint layout: fused
/// (`mlp.experts.gate_up_proj` / `down_proj`, `[E, H, 2I]` / `[E, I, H]` as
/// stored by the original releases — taken as they are — or `[E, 2I, H]` /
/// `[E, H, I]` as transformers 5 saves them, transposed), or one tensor per
/// expert (`mlp.experts.{e}.gate_proj.weight`, …), stacked.
pub(crate) fn load_moe(
    ctx: &mut crate::context::FlowCtx<'_>,
    lp: &str,
    moe: &Qwen3Moe,
    hidden: usize,
) -> anyhow::Result<MoeWeights> {
    use rlx_ir::{DType, Shape};
    let (e, i) = (moe.num_experts, moe.intermediate);
    let router = ctx.load_param(&format!("{lp}.mlp.gate.weight"), true)?;
    if let Some((gate_up, down, scheme)) = load_moe_packed(ctx, lp, moe, hidden)? {
        return Ok(MoeWeights {
            router,
            gate_up,
            down,
            packed: Some(scheme),
        });
    }
    let f = |d: &[usize]| Shape::new(d, DType::F32);
    let gu_key = format!("{lp}.mlp.experts.gate_up_proj");
    // Fused when the checkpoint has the fused tensor (taking it is the probe:
    // not every weight source answers `has`).
    let fused = ctx.weights.take(&gu_key, false).ok();
    let (gate_up, down) = if let Some((gu, gs)) = fused {
        let dn_key = format!("{lp}.mlp.experts.down_proj");
        let (dn, ds) = ctx.weights.take(&dn_key, false)?;
        let gate_up = match gs[..] {
            [ee, a, b] if ee == e && a == hidden && b == 2 * i => {
                ctx.synth_param(&gu_key, gu, f(&[e, hidden, 2 * i]))
            }
            [ee, a, b] if ee == e && a == 2 * i && b == hidden => ctx.synth_param(
                &format!("{gu_key}.t"),
                transpose_slabs(&gu, e, a, b),
                f(&[e, hidden, 2 * i]),
            ),
            _ => anyhow::bail!("{gu_key} is {gs:?}; {e} experts of {hidden}×{}", 2 * i),
        };
        let down = match ds[..] {
            [ee, a, b] if ee == e && a == i && b == hidden => {
                ctx.synth_param(&dn_key, dn, f(&[e, i, hidden]))
            }
            [ee, a, b] if ee == e && a == hidden && b == i => ctx.synth_param(
                &format!("{dn_key}.t"),
                transpose_slabs(&dn, e, a, b),
                f(&[e, i, hidden]),
            ),
            _ => anyhow::bail!("{dn_key} is {ds:?}; {e} experts of {i}×{hidden}"),
        };
        (gate_up, down)
    } else {
        // One `[out, in]` linear per expert: gate and up side by side.
        let mut gu = vec![0f32; e * hidden * 2 * i];
        let mut dn = vec![0f32; e * i * hidden];
        for x in 0..e {
            let p = format!("{lp}.mlp.experts.{x}");
            let (g, _) = ctx.weights.take(&format!("{p}.gate_proj.weight"), false)?;
            let (u, _) = ctx.weights.take(&format!("{p}.up_proj.weight"), false)?;
            let (d, _) = ctx.weights.take(&format!("{p}.down_proj.weight"), false)?;
            let slab = &mut gu[x * hidden * 2 * i..(x + 1) * hidden * 2 * i];
            for k in 0..hidden {
                for n in 0..i {
                    slab[k * 2 * i + n] = g[n * hidden + k];
                    slab[k * 2 * i + i + n] = u[n * hidden + k];
                }
            }
            let slab = &mut dn[x * i * hidden..(x + 1) * i * hidden];
            for k in 0..i {
                for n in 0..hidden {
                    slab[k * hidden + n] = d[n * i + k];
                }
            }
        }
        (
            ctx.synth_param(
                &format!("{lp}.mlp.experts.gate_up.stacked"),
                gu,
                f(&[e, hidden, 2 * i]),
            ),
            ctx.synth_param(
                &format!("{lp}.mlp.experts.down.stacked"),
                dn,
                f(&[e, i, hidden]),
            ),
        )
    };
    Ok(MoeWeights {
        router,
        gate_up,
        down,
        packed: None,
    })
}

/// The MoE MLP of `x` (`hidden_shape`, `[.., hidden]`): each token's
/// `top_k` experts through `GroupedMatMul`, weighted and summed.
pub(crate) fn emit_moe(
    gb: &mut rlx_ir::hir::HirMut<'_>,
    x: rlx_ir::HirNodeId,
    w: &MoeWeights,
    moe: &Qwen3Moe,
    hidden_shape: &rlx_ir::Shape,
) -> rlx_ir::HirNodeId {
    use rlx_ir::HirGraphExt;
    use rlx_ir::{DType, Op, Shape};
    let dims: Vec<usize> = hidden_shape
        .dims()
        .iter()
        .map(|d| d.unwrap_static())
        .collect();
    let h = *dims.last().expect("a hidden dim");
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let (k, i) = (moe.top_k.max(1), moe.intermediate);
    let f = |d: &[usize]| Shape::new(d, DType::F32);
    let x2 = gb.reshape_(x, vec![rows as i64, h as i64]);
    let logits = gb.mm(x2, w.router);
    let probs = gb.sm(logits, -1);
    let top = gb.add_node(Op::TopK { k }, vec![probs], f(&[rows, k]));
    let mut weights = gb.add_node(
        Op::GatherElements { axis: 1 },
        vec![probs, top],
        f(&[rows, k]),
    );
    if moe.norm_topk_prob {
        let total = gb.sum(weights, vec![1], true);
        weights = gb.div(weights, total);
    }
    let mut acc = None;
    for slot in 0..k {
        let idx = gb.narrow_(top, 1, slot, 1);
        let idx = gb.reshape_(idx, vec![rows as i64]);
        let wt = gb.narrow_(weights, 1, slot, 1);
        let grouped = match w.packed {
            Some(scheme) => Op::DequantGroupedMatMul { scheme },
            None => Op::GroupedMatMul,
        };
        let gu = gb.add_node(grouped.clone(), vec![x2, w.gate_up, idx], f(&[rows, 2 * i]));
        let gate = gb.narrow_(gu, 1, 0, i);
        let up = gb.narrow_(gu, 1, i, i);
        let gate = gb.silu(gate);
        let act = gb.mul(gate, up);
        let out = gb.add_node(grouped, vec![act, w.down, idx], f(&[rows, h]));
        let out = gb.mul(out, wt);
        acc = Some(match acc {
            None => out,
            Some(a) => gb.add(a, out),
        });
    }
    let reshaped: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
    gb.reshape_(acc.expect("top_k >= 1"), reshaped)
}
