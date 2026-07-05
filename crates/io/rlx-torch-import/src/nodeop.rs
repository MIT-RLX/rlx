// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Macro-driven op table for the "direct" ops — those that lower to a single
//! `add_node(Op::X, inputs, out_shape)`.
//!
//! These ops were previously written in **three** places (a `Call` variant, a
//! `hir_build` match arm building the live `Op`, and an `emit` match arm
//! printing the *same* `Op` as source text). That is pure duplication: the
//! build arm and the emit arm are the same construction, once as a value and
//! once as a string. [`define_node_ops!`] collapses it to **one line per op** —
//! the macro generates the `NodeOp` enum plus its [`NodeOp::build`] (value) and
//! [`NodeOp::emit`] (source) methods, so `hir_build`/`emit` each need a single
//! generic arm and adding a direct op is a one-line table entry.

use crate::call::{Value, dtype_token, shape_of};
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::op::{BinaryOp, CmpOp, ReduceOp};
use rlx_ir::{DType, HirGraphExt};

fn shape_source(dims: &[usize], dt: DType) -> String {
    let ds: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
    format!(
        "rlx_ir::Shape::new(&[{}], rlx_ir::DType::{})",
        ds.join(", "),
        dtype_token(dt)
    )
}

fn cmp_variant(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "Eq",
        CmpOp::Ne => "Ne",
        CmpOp::Lt => "Lt",
        CmpOp::Le => "Le",
        CmpOp::Gt => "Gt",
        CmpOp::Ge => "Ge",
    }
}

fn binop_variant(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "Add",
        BinaryOp::Sub => "Sub",
        BinaryOp::Mul => "Mul",
        BinaryOp::Div => "Div",
        BinaryOp::Max => "Max",
        BinaryOp::Min => "Min",
        BinaryOp::Pow => "Pow",
    }
}

fn reduce_variant(op: ReduceOp) -> &'static str {
    match op {
        ReduceOp::Sum => "Sum",
        ReduceOp::Mean => "Mean",
        ReduceOp::Max => "Max",
        ReduceOp::Min => "Min",
        ReduceOp::Prod => "Prod",
    }
}

fn usize_vec_source(v: &[usize]) -> String {
    format!(
        "vec![{}]",
        v.iter()
            .map(|d| format!("{d}usize"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn i64_vec_source(v: &[i64]) -> String {
    format!(
        "vec![{}]",
        v.iter()
            .map(|d| format!("{d}i64"))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Declare a family of single-`Op` nodes. Each entry lists its input value
/// fields `[a, b, …]`, its attribute fields `{ name: Type, … }`, how to build
/// the live `rlx_ir::Op` (`build =`), and how to print that same `Op` as Rust
/// source for the generated crate (`src =`). Both arms see the fields in scope.
macro_rules! define_node_ops {
    ( $(
        $(#[$doc:meta])*
        $variant:ident [ $($inp:ident),* $(,)? ] { $( $fname:ident : $fty:ty ),* $(,)? }
        build = $opbody:expr,
        src   = $srcbody:expr
      );+ $(;)? ) => {
        /// A node lowering to exactly one `add_node(Op, inputs, shape)`.
        #[derive(Debug, Clone)]
        pub enum NodeOp {
            $(
                $(#[$doc])*
                $variant {
                    $( $inp: Value, )*
                    $( $fname: $fty, )*
                    out: Vec<usize>,
                    out_dtype: DType,
                },
            )+
        }

        impl NodeOp {
            /// Build the live HIR node, resolving input names via `resolve`.
            pub fn build(
                &self,
                b: &mut HirMut,
                resolve: &mut dyn FnMut(&str) -> anyhow::Result<HirNodeId>,
            ) -> anyhow::Result<HirNodeId> {
                match self {
                    $(
                        NodeOp::$variant { $($inp,)* $($fname,)* out, out_dtype } => {
                            let inputs = vec![ $( resolve($inp)?, )* ];
                            let op: rlx_ir::Op = $opbody;
                            let _ = ($($fname,)*);
                            Ok(b.add_node(op, inputs, shape_of(out, *out_dtype)))
                        }
                    )+
                }
            }

            /// Print the equivalent generated-crate source line.
            pub fn emit(&self, res: &str, r: &dyn Fn(&str) -> String) -> String {
                match self {
                    $(
                        NodeOp::$variant { $($inp,)* $($fname,)* out, out_dtype } => {
                            let inputs = vec![ $( r($inp), )* ].join(", ");
                            let op_src: String = $srcbody;
                            let _ = ($($fname,)*);
                            format!(
                                "let {res} = b.add_node({op_src}, vec![{inputs}], {});",
                                shape_source(out, *out_dtype)
                            )
                        }
                    )+
                }
            }
        }
    };
}

define_node_ops! {
    /// Elementwise comparison → bool.
    Compare [a, b] { op: CmpOp }
        build = rlx_ir::Op::Compare(*op),
        src   = format!("rlx_ir::Op::Compare(rlx_ir::op::CmpOp::{})", cmp_variant(*op));

    /// Elementwise select `cond ? a : b`.
    Where [cond, a, b] { }
        build = rlx_ir::Op::Where,
        src   = "rlx_ir::Op::Where".to_string();

    /// Broadcast to a target shape.
    Expand [x] { target: Vec<i64> }
        build = rlx_ir::Op::Expand { target_shape: target.clone() },
        src   = format!("rlx_ir::Op::Expand {{ target_shape: {} }}", i64_vec_source(target));

    /// Binary op carrying an explicit output shape (Pow/Max/Min/…).
    BinaryShaped [a, b] { op: BinaryOp }
        build = rlx_ir::Op::Binary(*op),
        src   = format!("rlx_ir::Op::Binary(rlx_ir::op::BinaryOp::{})", binop_variant(*op));

    /// 2-D pooling (max / avg).
    Pool [x] { kind: ReduceOp, kernel: Vec<usize>, stride: Vec<usize>, padding: Vec<usize> }
        build = rlx_ir::Op::Pool {
            kind: *kind,
            kernel_size: kernel.clone(),
            stride: stride.clone(),
            padding: padding.clone(),
        },
        src = format!(
            "rlx_ir::Op::Pool {{ kind: rlx_ir::op::ReduceOp::{}, kernel_size: {}, stride: {}, padding: {} }}",
            reduce_variant(*kind), usize_vec_source(kernel), usize_vec_source(stride), usize_vec_source(padding)
        );

    /// Top-k indices along the last axis.
    TopK [x] { k: usize }
        build = rlx_ir::Op::TopK { k: *k },
        src   = format!("rlx_ir::Op::TopK {{ k: {k} }}");

    /// Group normalization on NCHW `[N,C,H,W]` — `x`, `gamma[C]`, `beta[C]`.
    /// Normalizes over `(C/num_groups) × H × W` per group (torch semantics).
    GroupNorm [x, gamma, beta] { num_groups: usize, eps: f32 }
        build = rlx_ir::Op::GroupNorm { num_groups: *num_groups, eps: *eps },
        src   = format!("rlx_ir::Op::GroupNorm {{ num_groups: {num_groups}, eps: {eps}f32 }}");

    /// Nearest-neighbor 2× upsample on NCHW (doubles spatial dims 2 and 3).
    ResizeNearest2x [x] { }
        build = rlx_ir::Op::ResizeNearest2x,
        src   = "rlx_ir::Op::ResizeNearest2x".to_string();
}
