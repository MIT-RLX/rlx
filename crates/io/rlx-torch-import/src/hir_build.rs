// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Interpret a [`Lowered`] program into a live `HirModule` (for run + verify).

use crate::call::*;
use anyhow::{Result, anyhow};
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::BinaryOp;
use rlx_ir::{HirGraphExt, HirModule};
use std::collections::HashMap;

pub fn build_hir(lo: &Lowered) -> Result<HirModule> {
    let mut hir = HirModule::new(lo.name.clone());
    let mut b = HirMut::new(&mut hir);
    let mut vm: HashMap<String, HirNodeId> = HashMap::new();

    for i in &lo.inputs {
        // Integer inputs (e.g. token ids) are fed as f32 and cast to their real
        // dtype inside the graph. Small ints are exact in f32, and this keeps
        // the f32-arena host surface (CPU + CUDA + Metal) able to accept them
        // (those backends only widen F32/F16/BF16 on the I/O boundary).
        if crate::call::is_float_dtype(i.dtype) {
            let id = b.input(i.name.clone(), i.hir_shape(i.dtype));
            vm.insert(i.name.clone(), id);
        } else {
            let f = b.input(i.name.clone(), i.hir_shape(rlx_ir::DType::F32));
            let casted = b.cast(f, i.dtype);
            vm.insert(i.name.clone(), casted);
        }
    }
    for p in lo.params.iter().chain(lo.zero_params.iter()) {
        // Integer params (e.g. BatchNorm `num_batches_tracked`, position-id
        // buffers) are declared f32 and cast to their real dtype — same reason
        // as integer inputs: the f32 arena (CPU/CUDA/Metal) only holds f32.
        if crate::call::is_float_dtype(p.dtype) {
            let id = b.param(p.key.clone(), shape_of(&p.shape, p.dtype));
            vm.insert(p.value_id.clone(), id);
        } else {
            let f = b.param(p.key.clone(), shape_of(&p.shape, rlx_ir::DType::F32));
            let casted = b.cast(f, p.dtype);
            vm.insert(p.value_id.clone(), casted);
        }
    }

    for ins in &lo.instrs {
        let get = |vm: &HashMap<String, HirNodeId>, name: &str| -> Result<HirNodeId> {
            vm.get(name)
                .copied()
                .ok_or_else(|| anyhow!("unresolved value {name:?} (instr {})", ins.result))
        };
        let out = match &ins.call {
            Call::Mm(a, c) => b.mm(get(&vm, a)?, get(&vm, c)?),
            Call::Binary(op, a, c) => {
                let (x, y) = (get(&vm, a)?, get(&vm, c)?);
                match op {
                    BinaryOp::Add => b.add(x, y),
                    BinaryOp::Sub => b.sub(x, y),
                    BinaryOp::Mul => b.mul(x, y),
                    BinaryOp::Div => b.div(x, y),
                    other => return Err(anyhow!("binary {other:?} not wired")),
                }
            }
            Call::Act(act, a) => {
                let x = get(&vm, a)?;
                let s = b.shape(x).clone();
                b.activation(*act, x, s)
            }
            Call::Ln {
                x,
                gamma,
                beta,
                eps,
            } => b.ln(get(&vm, x)?, get(&vm, gamma)?, get(&vm, beta)?, *eps),
            Call::RmsNorm {
                x,
                gamma,
                beta,
                eps,
            } => b.rms_norm(get(&vm, x)?, get(&vm, gamma)?, get(&vm, beta)?, *eps),
            Call::Reshape { x, shape } => b.reshape_(get(&vm, x)?, shape.clone()),
            Call::Transpose { x, perm } => b.transpose_(get(&vm, x)?, perm.clone()),
            Call::Narrow {
                x,
                axis,
                start,
                len,
            } => b.narrow_(get(&vm, x)?, *axis, *start, *len),
            Call::Concat { xs, axis } => {
                let ids = xs.iter().map(|n| get(&vm, n)).collect::<Result<Vec<_>>>()?;
                b.concat_(ids, *axis)
            }
            Call::Gather {
                table,
                indices,
                axis,
            } => b.gather_(get(&vm, table)?, get(&vm, indices)?, *axis),
            Call::Softmax { x, axis } => b.sm(get(&vm, x)?, *axis),
            Call::Reduce {
                op,
                x,
                axes,
                keep_dim,
            } => match op {
                rlx_ir::op::ReduceOp::Mean => b.mean(get(&vm, x)?, axes.clone(), *keep_dim),
                rlx_ir::op::ReduceOp::Sum => b.sum(get(&vm, x)?, axes.clone(), *keep_dim),
                other => return Err(anyhow!("reduce {other:?} not wired")),
            },
            Call::Cast { x, to } => b.cast(get(&vm, x)?, *to),
            Call::Conv2d {
                x,
                weight,
                kernel,
                stride,
                padding,
                groups,
                out,
                out_dtype,
            } => b.conv2d(
                get(&vm, x)?,
                get(&vm, weight)?,
                *kernel,
                *stride,
                *padding,
                *groups,
                shape_of(out, *out_dtype),
            ),
            Call::Attention {
                q,
                k,
                v,
                num_heads,
                head_dim,
                mask,
                out,
                out_dtype,
            } => b.attention_kind(
                get(&vm, q)?,
                get(&vm, k)?,
                get(&vm, v)?,
                *num_heads,
                *head_dim,
                *mask,
                shape_of(out, *out_dtype),
            ),
            Call::Rope {
                x,
                cos,
                sin,
                head_dim,
            } => b.rope(get(&vm, x)?, get(&vm, cos)?, get(&vm, sin)?, *head_dim),
            Call::Full {
                value,
                shape,
                dtype,
            } => {
                let numel: usize = shape.iter().product();
                let data = fill_bytes(*value, *dtype, numel)?;
                b.add_node(
                    rlx_ir::Op::Constant { data },
                    vec![],
                    shape_of(shape, *dtype),
                )
            }
            Call::ConvTranspose2d {
                x,
                weight,
                kernel,
                stride,
                padding,
                dilation,
                output_padding,
                groups,
                out,
                out_dtype,
            } => b.conv_transpose2d(
                get(&vm, x)?,
                get(&vm, weight)?,
                *kernel,
                *stride,
                *padding,
                *dilation,
                *output_padding,
                *groups,
                shape_of(out, *out_dtype),
            ),
            Call::AttentionBias {
                q,
                k,
                v,
                bias,
                num_heads,
                head_dim,
                out,
                out_dtype,
            } => b.attention_bias(
                get(&vm, q)?,
                get(&vm, k)?,
                get(&vm, v)?,
                get(&vm, bias)?,
                *num_heads,
                *head_dim,
                shape_of(out, *out_dtype),
            ),
            Call::Iota { rows, step, dtype } => {
                let mut data = Vec::with_capacity(rows * 8);
                for r in 0..*rows {
                    let v = (r as i64) * step;
                    match dtype {
                        rlx_ir::DType::I64 => data.extend_from_slice(&v.to_le_bytes()),
                        rlx_ir::DType::I32 => data.extend_from_slice(&(v as i32).to_le_bytes()),
                        _ => data.extend_from_slice(&(v as f32).to_le_bytes()),
                    }
                }
                b.add_node(
                    rlx_ir::Op::Constant { data },
                    vec![],
                    shape_of(&[*rows, 1], *dtype),
                )
            }
            Call::Arange {
                start,
                step,
                len,
                dtype,
            } => {
                let mut data = Vec::new();
                for i in 0..*len {
                    let v = start + (i as f64) * step;
                    match dtype {
                        rlx_ir::DType::I64 => data.extend_from_slice(&(v as i64).to_le_bytes()),
                        rlx_ir::DType::I32 => data.extend_from_slice(&(v as i32).to_le_bytes()),
                        _ => data.extend_from_slice(&(v as f32).to_le_bytes()),
                    }
                }
                b.add_node(
                    rlx_ir::Op::Constant { data },
                    vec![],
                    shape_of(&[*len], *dtype),
                )
            }
            Call::GridSample {
                input,
                grid,
                mode,
                pad,
                align_corners,
                ..
            } => b.grid_sample2d(
                get(&vm, input)?,
                get(&vm, grid)?,
                *mode,
                *pad,
                *align_corners,
            ),
            Call::Resize {
                x,
                out_h,
                out_w,
                align_corners,
                cubic,
                antialias,
                ..
            } => {
                let x = get(&vm, x)?;
                match (*cubic, *antialias) {
                    (false, false) => b.resize_bilinear2d(x, *out_h, *out_w, *align_corners),
                    (true, false) => b.resize_bicubic2d(x, *out_h, *out_w, *align_corners),
                    (false, true) => b.resize_bilinear2d_aa(x, *out_h, *out_w, *align_corners),
                    (true, true) => b.resize_bicubic2d_aa(x, *out_h, *out_w, *align_corners),
                }
            }
            Call::Node(node) => {
                let mut resolve = |name: &str| get(&vm, name);
                node.build(&mut b, &mut resolve)?
            }
            Call::Alias(src) => {
                let id = get(&vm, src)?;
                vm.insert(ins.result.clone(), id);
                continue;
            }
        };
        vm.insert(ins.result.clone(), out);
    }

    let outs = lo
        .outputs
        .iter()
        .map(|n| {
            vm.get(n)
                .copied()
                .ok_or_else(|| anyhow!("unresolved graph output {n:?}"))
        })
        .collect::<Result<Vec<_>>>()?;
    b.set_outputs(outs);
    Ok(hir)
}
