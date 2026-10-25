// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pipeline stages addressed by host: which server runs which layer range.
//!
//! [`crate::LayerSplit`] decides how many layers each rank gets; a
//! [`StageSpec`] binds a contiguous range to a server address in pipeline order
//! (stage 0 first), the shape a coordinator that relays state through
//! addressable workers needs. [`plan_stages`] turns a split description into
//! stages, [`validate_stages`] refuses stages that do not tile the model —
//! a gap or overlap runs some layers twice and others never, with every shape
//! still agreeing, so it must be caught before any work.

use crate::LayerSplit;
use anyhow::{Result, ensure};
use std::ops::Range;

/// One pipeline stage: a server and the layers it runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageSpec {
    pub addr: String,
    pub layers: Range<usize>,
}

/// Parse `host:port:start:end,…`.
pub fn parse_stages(s: &str) -> Result<Vec<StageSpec>> {
    s.split(',')
        .filter(|w| !w.trim().is_empty())
        .map(|w| {
            let p: Vec<&str> = w.trim().rsplitn(3, ':').collect();
            ensure!(p.len() == 3, "{w:?} is not host:port:start:end");
            Ok(StageSpec {
                addr: p[2].to_string(),
                layers: p[1].parse()?..p[0].parse()?,
            })
        })
        .collect()
}

/// Spread `n_layers` over `addrs` in pipeline order. `split` is explicit counts
/// (`"30,33,30"`, [`LayerSplit::parse`]), relative weights prefixed `w:`
/// (`"w:64,24,24"` — e.g. each node's RAM; [`LayerSplit::weighted`]), or `None`
/// for an even split ([`LayerSplit::even`], the remainder to the last stages).
/// Every stage gets at least one layer.
pub fn plan_stages(
    addrs: &[String],
    n_layers: usize,
    split: Option<&str>,
) -> Result<Vec<StageSpec>> {
    let k = addrs.len();
    ensure!(k >= 1 && k <= n_layers, "{k} stages for {n_layers} layers");
    let ls = match split {
        Some(s) if s.starts_with("w:") => {
            let w: Vec<f64> = s[2..]
                .split(',')
                .map(|x| {
                    x.trim()
                        .parse::<f64>()
                        .map_err(|e| anyhow::anyhow!("{x:?}: {e}"))
                })
                .collect::<Result<_>>()?;
            ensure!(w.len() == k, "{} weights for {k} stages", w.len());
            ensure!(w.iter().all(|x| *x > 0.0), "weights must be positive");
            // LayerSplit weighs integers; scale so ratios keep 6 digits.
            let max = w.iter().cloned().fold(0.0, f64::max);
            let ints: Vec<u64> = w
                .iter()
                .map(|x| (x / max * 1e6).round().max(1.0) as u64)
                .collect();
            LayerSplit::weighted(n_layers, &ints)
        }
        Some(s) => LayerSplit::parse(s).map_err(anyhow::Error::msg)?,
        None => LayerSplit::even(n_layers, k as u32),
    };
    ls.validate(n_layers, k as u32)
        .map_err(anyhow::Error::msg)?;
    let mut start = 0;
    Ok(addrs
        .iter()
        .zip(&ls.counts)
        .map(|(a, &c)| {
            let s = StageSpec {
                addr: a.clone(),
                layers: start..start + c as usize,
            };
            start += c as usize;
            s
        })
        .collect())
}

/// Check stages tile `0..n_layers` in order with no gap or overlap.
pub fn validate_stages(stages: &[StageSpec], n_layers: usize) -> Result<()> {
    ensure!(!stages.is_empty(), "no stages");
    let mut next = 0;
    for s in stages {
        ensure!(
            s.layers.start == next && s.layers.end > s.layers.start,
            "stage {} runs layers {:?}, expected a non-empty range starting at {next}",
            s.addr,
            s.layers
        );
        next = s.layers.end;
    }
    ensure!(
        next == n_layers,
        "stages cover layers 0..{next} of {n_layers}: some layers would never run"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stages_plan_and_validate() {
        let addrs: Vec<String> = ["a:1", "b:2", "c:3"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let even = plan_stages(&addrs, 10, None).unwrap();
        assert_eq!(
            even.iter().map(|s| s.layers.clone()).collect::<Vec<_>>(),
            vec![0..3, 3..6, 6..10]
        );
        validate_stages(&even, 10).unwrap();
        let w = plan_stages(&addrs, 10, Some("w:2,1,1")).unwrap();
        assert_eq!(w.iter().map(|s| s.layers.len()).sum::<usize>(), 10);
        assert!(w[0].layers.len() > w[1].layers.len());
        let c = plan_stages(&addrs, 10, Some("2,3,5")).unwrap();
        assert_eq!(c[2].layers, 5..10);
        assert!(plan_stages(&addrs, 10, Some("2,3,4")).is_err());
        let gap = vec![
            StageSpec {
                addr: "a".into(),
                layers: 0..3,
            },
            StageSpec {
                addr: "b".into(),
                layers: 4..10,
            },
        ];
        assert!(validate_stages(&gap, 10).is_err());
        assert!(validate_stages(&even[..2], 10).is_err());
        let parsed = parse_stages("10.0.0.1:7000:0:46,host:7001:46:93").unwrap();
        assert_eq!(parsed[0].addr, "10.0.0.1:7000");
        assert_eq!(parsed[1].layers, 46..93);
    }
}
