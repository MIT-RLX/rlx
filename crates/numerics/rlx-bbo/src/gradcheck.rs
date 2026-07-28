// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Finite-difference gradient check for compiled RLX loss graphs.

use std::collections::HashMap;

use rlx_ir::{Graph, find_param_nodes};
use rlx_opt::rlx_autodiff::grad_with_loss;
use rlx_runtime::{Device, Session};

use crate::graph_opt::{GraphOptError, apply_all_params};

/// Central-difference gradcheck for `fwd` w.r.t. `optimize` params.
///
/// Compares reverse-mode grads from `grad_with_loss` against symmetric FD
/// on the forward loss (no `d_output` input).
pub fn gradcheck_graph(
    fwd: &Graph,
    optimize: &[&str],
    values: &HashMap<String, f32>,
    inputs: &[(&str, &[f32])],
    eps: f32,
    rtol: f32,
    atol: f32,
    device: Device,
) -> Result<(), GraphOptError> {
    let param_ids = find_param_nodes(fwd, optimize).map_err(GraphOptError::ParamNotFound)?;
    let bwd = grad_with_loss(fwd, &param_ids);
    let session = Session::new(device);
    let mut bwd_compiled = session.compile(bwd);
    let mut fwd_compiled = session.compile(fwd.clone());

    let opt_vals: Vec<f32> = optimize
        .iter()
        .map(|n| {
            values
                .get(*n)
                .copied()
                .ok_or_else(|| GraphOptError::ParamNotFound((*n).into()))
        })
        .collect::<Result<_, _>>()?;

    apply_all_params(&mut bwd_compiled, values, optimize, &opt_vals);
    let mut run_in: Vec<(&str, &[f32])> = inputs.to_vec();
    run_in.push(("d_output", &[1.0]));
    let outs = bwd_compiled.run(&run_in);
    let ad_grads: Vec<f32> = outs[1..].iter().map(|g| g[0]).collect();

    for (i, name) in optimize.iter().enumerate() {
        let xi = opt_vals[i];
        let h = eps.max(eps * xi.abs());

        let mut plus = opt_vals.clone();
        plus[i] = xi + h;
        let fp = eval_forward_loss(&mut fwd_compiled, values, optimize, &plus, inputs);

        let mut minus = opt_vals.clone();
        minus[i] = xi - h;
        let fm = eval_forward_loss(&mut fwd_compiled, values, optimize, &minus, inputs);

        let fd = (fp - fm) / (2.0 * h);
        let ad = ad_grads[i];
        if !is_close(ad, fd, rtol, atol) {
            return Err(GraphOptError::GradcheckMismatch {
                param: (*name).to_string(),
                ad,
                fd,
            });
        }
    }
    Ok(())
}

fn eval_forward_loss(
    compiled: &mut rlx_runtime::CompiledGraph,
    all: &HashMap<String, f32>,
    optimize: &[&str],
    opt_values: &[f32],
    inputs: &[(&str, &[f32])],
) -> f32 {
    apply_all_params(compiled, all, optimize, opt_values);
    compiled.run(inputs)[0][0]
}

#[inline]
fn is_close(a: f32, b: f32, rtol: f32, atol: f32) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rlx_ir::{DType, Graph, Op, Shape, op::BinaryOp};

    use super::*;

    fn quadratic_fwd() -> Graph {
        let mut g = Graph::new("quad_gc");
        let s = Shape::new(&[1], DType::F32);
        let x = g.param("x", s.clone());
        let target = g.add_node(
            Op::Constant {
                data: 2.0f32.to_le_bytes().to_vec(),
            },
            vec![],
            s.clone(),
        );
        let err = g.binary(BinaryOp::Sub, x, target, s.clone());
        let loss = g.binary(BinaryOp::Mul, err, err, s);
        g.set_outputs(vec![loss]);
        g
    }

    #[test]
    fn parabolic_gradcheck_passes() {
        let fwd = quadratic_fwd();
        let values = HashMap::from([("x".to_string(), 0.5f32)]);
        gradcheck_graph(&fwd, &["x"], &values, &[], 1e-3, 1e-2, 1e-4, Device::Cpu).unwrap();
    }
}
