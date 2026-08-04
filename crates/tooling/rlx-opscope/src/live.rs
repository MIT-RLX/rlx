// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Live **sampled** recorder — the production path. Instead of writing every run
//! to a CSV, it rides a real forward pass, samples 1-in-N runs, and folds each
//! op-site's element-wise fingerprint into bounded [`crate::online`] sketches:
//! a HyperLogLog estimates how many **distinct inputs** a site has seen (→
//! temporal recurrence / memoize opportunity) and a reservoir keeps a sample —
//! all at near-zero, constant memory regardless of stream length.

use crate::StatSpec;
use crate::guard::hash_f32;
use crate::online::{Hll, Reservoir};
use std::collections::HashMap;

struct SiteSketch {
    hll: Hll, // distinct input fingerprints seen
    reservoir: Reservoir,
    samples: u64, // sampled runs touching this site
}

impl SiteSketch {
    fn new() -> Self {
        Self {
            hll: Hll::new(12),
            reservoir: Reservoir::new(32),
            samples: 0,
        }
    }
}

/// Streaming per-site sketch aggregator over a sampled run stream.
pub struct LiveSampler {
    sample_every: u64,
    seen: u64,
    pub sampled: u64,
    sites: HashMap<String, SiteSketch>,
}

impl LiveSampler {
    /// Sample 1 in `sample_every` runs (1 = every run).
    pub fn new(sample_every: u64) -> Self {
        Self {
            sample_every: sample_every.max(1),
            seen: 0,
            sampled: 0,
            sites: HashMap::new(),
        }
    }

    /// Fold one run's tapped outputs into the sketches (sampled). Uses each
    /// site's `elemsig` (element-wise fingerprint) as the input identity.
    pub fn record(&mut self, specs: &[StatSpec], outs: &[Vec<f32>]) {
        self.seen += 1;
        if !self.seen.is_multiple_of(self.sample_every) {
            return;
        }
        self.sampled += 1;
        for spec in specs
            .iter()
            .filter(|s| s.stat == "elemsig" && s.role == "lhs")
        {
            let data = &outs[spec.out_idx];
            let s = self
                .sites
                .entry(spec.site.clone())
                .or_insert_with(SiteSketch::new);
            s.hll.add_bits(hash_f32(data)); // one fingerprint per run → distinct-input count
            s.reservoir.extend(data);
            s.samples += 1;
        }
    }

    /// Per-site streaming report: estimated distinct inputs and the implied
    /// temporal recurrence (`1 − distinct/samples`) → memoize candidates.
    pub fn report(&self) {
        println!(
            "Live sampler: {} runs seen, {} sampled (1-in-{}), {} sites, bounded memory",
            self.seen,
            self.sampled,
            self.sample_every,
            self.sites.len()
        );
        println!(
            "{:<14} {:>8} {:>10} {:>8}   opportunity",
            "site", "samples", "~distinct", "recur"
        );
        println!("{}", "-".repeat(72));
        let mut rows: Vec<(&String, f64, f32)> = self
            .sites
            .iter()
            .map(|(site, s)| {
                let distinct = s.hll.estimate();
                let recur = (1.0 - distinct as f32 / s.samples.max(1) as f32).clamp(0.0, 1.0);
                (site, distinct, recur)
            })
            .collect();
        rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        for (site, distinct, recur) in rows.iter().take(12) {
            let opp = if *recur > 0.5 {
                format!("memoize — inputs repeat {:.0}%", recur * 100.0)
            } else {
                "(inputs mostly distinct)".into()
            };
            let s = &self.sites[*site];
            println!(
                "{:<14} {:>8} {:>10.0} {:>7.0}%   {opp}",
                site.chars().take(14).collect::<String>(),
                s.samples,
                distinct,
                recur * 100.0
            );
        }
    }
}
