// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! One typed function per op-table row: `ops::matmul(graph, lhs, rhs)`.
//!
//! These are what every binding calls. Each takes the graph and the row's
//! arguments as wire values (options flattened after the positional ones, in
//! table order), decodes them — node ids checked, enums parsed, pairs
//! validated — runs the body under the panic guard, and returns the wire
//! result. A binding's job shrinks to converting its own values to wire
//! values and back.

use rlx_ir::infer::GraphExt;
use rlx_ir::op::CmpOp;
use rlx_ir::{Graph, NodeId, Op};

use crate::{ApiError, ApiResult};

/// `Compare(op)` with the broadcast output shape.
pub fn compare_nodes(g: &mut Graph, op: CmpOp, a: NodeId, b: NodeId) -> ApiResult<NodeId> {
    let s = rlx_ir::shape::compare_shape(g.shape(a), g.shape(b))
        .map_err(|e| ApiError::type_(format!("compare shape: {e}")))?;
    Ok(g.add_node(Op::Compare(op), vec![a, b], s))
}

/// The core emitter: rows → `pub fn`s.
macro_rules! core_ops {
    ( [$($extra:tt)*]
      $(
        $(#[doc = $doc:literal])*
        $name:ident $( [ $( $rk:ident = $rv:literal ),* $(,)? ] )?
        ( $( $arg:ident : $kind:ident $( = $def:expr )? ),* $(,)?
          $( ; { $( $oarg:ident $( | $oal:ident )* : $okind:ident $( = $odef:expr )? ),* $(,)? } )? )
        $( -> $ret:ident )?
        => | $g:ident | $body:block ;
      )*
    ) => {
        $(
            $(#[doc = $doc])*
            #[allow(clippy::too_many_arguments, unused_variables, clippy::let_unit_value)]
            pub fn $name(
                graph: &mut Graph,
                $( $arg: $crate::wire_ty!($kind), )*
                $( $( $oarg: $crate::wire_ty!($okind), )* )?
            ) -> ApiResult<$crate::ret_ty!($($ret)?)> {
                let op = stringify!($name);
                let (out,) = {
                    let g = &*graph;
                    $( let $arg = $crate::decode!(g, $kind, $arg, $crate::kinds::Arg(op, stringify!($arg))); )*
                    $( $( let $oarg = $crate::decode!(g, $okind, $oarg, $crate::kinds::Arg(op, stringify!($oarg))); )* )?
                    let out = $crate::panics::catch(op, move || -> ApiResult<_> {
                        let $g = &mut *graph;
                        Ok($body)
                    })??;
                    (out,)
                };
                Ok($crate::encode!(($($ret)?), out))
            }
        )*
    };
}

crate::for_each_graph_op!(core_ops);
