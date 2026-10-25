// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A sparse autoencoder on an activation of a package's function: the
//! graph of one ([`sae_apply`]), and an SAE package (what
//! `trainer::PackageTrainer::save_sae` writes) spliced into the package a
//! model opens — open option `sae: "<path>"` or `{path, mode}`:
//!
//! - `mode: "replace"` (the default): each function with the SAE's `tap`
//!   reads its reconstruction there, so the model runs on what the SAE
//!   keeps of that activation;
//! - `mode: "features"`: the functions are unchanged but for one more
//!   output, `sae.features`, the SAE's features at the tap.

use anyhow::{Context, Result, bail};
use rlx_ir::{DType, Graph, GraphExt, NodeId, Shape};
use serde_json::Value;

use super::rlxp::Opened;

/// The SAE's params: `[d, width]`, `[1, width]`, `[width, d]`, `[1, d]`.
pub(super) const SAE_PARAMS: [&str; 4] = ["sae.w_enc", "sae.b_enc", "sae.w_dec", "sae.b_dec"];

/// `h` `[.., d]` through the SAE: `(features, reconstruction)` — `z =
/// relu((h − b_dec)·W_enc + b_enc)`, `ĥ = z·W_dec + b_dec`.
pub(super) fn sae_apply(g: &mut Graph, h: NodeId, d: usize, width: usize) -> (NodeId, NodeId) {
    let f32s = |s: &[usize]| Shape::new(s, DType::F32);
    let we = g.param(SAE_PARAMS[0], f32s(&[d, width]));
    let be = g.param(SAE_PARAMS[1], f32s(&[1, width]));
    let wd = g.param(SAE_PARAMS[2], f32s(&[width, d]));
    let bd = g.param(SAE_PARAMS[3], f32s(&[1, d]));
    let c = g.sub(h, bd);
    let z = g.mm(c, we);
    let z = g.add(z, be);
    let z = g.relu(z);
    let r = g.mm(z, wd);
    let r = g.add(r, bd);
    (z, r)
}

/// The node `tap` names in `g` (its `name`, or `#id`), if any.
pub(super) fn find_tap(g: &Graph, tap: &str) -> Result<Option<NodeId>> {
    Ok(match tap.strip_prefix('#') {
        Some(n) => Some(NodeId(n.parse().context("`#id` taps a node by its id")?)),
        None => g
            .nodes()
            .iter()
            .find(|n| n.name.as_deref() == Some(tap))
            .map(|n| n.id),
    })
}

/// Splice the SAE package `option` names (`"<path>"` or `{path, mode}`)
/// into `pkg`'s functions (see the module docs).
pub(super) fn splice(pkg: &mut Opened, option: &Value) -> Result<()> {
    let (path, mode) = match option {
        Value::String(p) => (p.as_str(), "replace"),
        Value::Object(o) => (
            o.get("path")
                .and_then(Value::as_str)
                .context("the `sae` option names its `path`")?,
            o.get("mode").and_then(Value::as_str).unwrap_or("replace"),
        ),
        _ => bail!("the `sae` option is a path or {{path, mode}}"),
    };
    if mode != "replace" && mode != "features" {
        bail!("the SAE's `mode` is \"replace\" or \"features\", not \"{mode}\"");
    }
    let sae = Opened::open(&super::rlxp::beside(&pkg.path, path))?;
    let spec = sae
        .meta
        .get("sae")
        .context("not an SAE package (no `sae` block)")?;
    let tap = spec
        .get("tap")
        .and_then(Value::as_str)
        .context("the SAE names its `tap`")?
        .to_string();
    let width = spec
        .get("width")
        .and_then(Value::as_u64)
        .context("the SAE gives its `width`")? as usize;
    let mut spliced = 0;
    for g in &mut pkg.graphs {
        let Some(id) = find_tap(g, &tap)? else {
            continue;
        };
        let d = match g.node(id).shape.dims().last() {
            Some(rlx_ir::Dim::Static(d)) => *d,
            _ => bail!("the tap '{tap}' has no static feature axis"),
        };
        // The tap, then the SAE on it: its readers read the reconstruction
        // (`replace`), or the features are one more output (`features`).
        let mut features = None;
        let mut out = rlx_ir::stateful::rebuild(g, |out, _, node, inputs| {
            if node.id != id {
                return Ok(None);
            }
            let tapped = out.add_node(node.op.clone(), inputs.to_vec(), node.shape.clone());
            out.node_mut(tapped).name = node.name.clone();
            let (z, r) = sae_apply(out, tapped, d, width);
            if mode == "features" {
                out.node_mut(z).name = Some("sae.features".into());
                features = Some(z);
                Ok(Some(tapped))
            } else {
                Ok(Some(r))
            }
        })
        .map_err(anyhow::Error::msg)?;
        if let Some(z) = features {
            let mut outs = out.outputs.clone();
            outs.push(z);
            out.set_outputs(outs);
        }
        out.name = g.name.clone();
        *g = out;
        spliced += 1;
    }
    if spliced == 0 {
        bail!(
            "no function of {} has the SAE's tap '{tap}'",
            pkg.path.display()
        );
    }
    for name in SAE_PARAMS {
        let (bytes, dtype) = sae.weights.held_bytes(name)?;
        pkg.weights.hold(name, bytes, dtype);
    }
    Ok(())
}
