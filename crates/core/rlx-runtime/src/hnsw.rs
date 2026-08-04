// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compact **HNSW** (Hierarchical Navigable Small World) index for approximate
//! top-k relevance search — the sub-linear "smart arrangement" that lets KV
//! retrieval navigate hundreds of thousands of blocks (→ million-token context)
//! in `O(log N)` instead of a linear key scan.
//!
//! Purpose-built for the KV-retention block store: each block contributes one
//! **key** (mean K, width `dim`); a query (the newest token's K) searches for
//! the most relevant blocks by **inner product** (MIPS — the same `dot` score
//! the exact retrieval uses). It is *append-only* (blocks are written once as
//! context grows and never removed from the index — retrieval *copies* the
//! selected blocks' data), which is HNSW's sweet spot: no delete churn, stable
//! sub-linear search.
//!
//! Deterministic: node levels are derived from the insertion index via a
//! splitmix64 hash (no RNG), so an index rebuilt from the same inserts is
//! bit-identical — important for reproducible decode.
//!
//! This is a from-scratch, dependency-free implementation (rlx has no ANN crate;
//! rlx-umap's KNN is full-pairwise graph ops, not a navigable index).

/// Similarity metric. All are expressed as a **score where higher = more
/// relevant**, so navigation/search is a uniform max-search (L2 uses negative
/// squared distance).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Inner product (maximum-inner-product search). Matches the exact
    /// retrieval's `dot(query, block_key)` relevance.
    Dot,
    /// Cosine similarity (dot of L2-normalized vectors). Magnitude-invariant.
    Cosine,
    /// **Euclidean distance** (score = `−‖a−b‖²`). A true metric (triangle
    /// inequality holds), so the greedy small-world navigation is theoretically
    /// sound — often better-conditioned than MIPS for retrieval.
    L2,
}

/// HNSW build/search parameters. Defaults suit block counts up to ~1e6.
#[derive(Clone, Copy, Debug)]
pub struct HnswConfig {
    /// Neighbors per node at levels > 0.
    pub m: usize,
    /// Neighbors per node at level 0 (typically `2·m`).
    pub m0: usize,
    /// Candidate-list width during insertion (quality vs build cost).
    pub ef_construction: usize,
    /// Metric.
    pub metric: Metric,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 100,
            metric: Metric::Dot,
        }
    }
}

/// Deterministic uniform in `(0, 1]` from a 64-bit index (splitmix64).
fn hash_unit(mut z: u64) -> f64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Map to (0, 1]; +1 avoids 0 so ln() is finite.
    ((z >> 11) as f64 + 1.0) / ((1u64 << 53) as f64)
}

/// One indexed node.
struct Node {
    vector: Vec<f32>,
    /// `links[l]` = neighbor node indices at level `l` (0..=top level).
    links: Vec<Vec<u32>>,
    /// Precomputed L2 norm (for cosine).
    norm: f32,
}

/// A compact HNSW index over `f32` vectors of fixed `dim`.
pub struct Hnsw {
    #[allow(dead_code)] // fixed vector width the index was built for; kept for introspection
    dim: usize,
    cfg: HnswConfig,
    ml: f64,
    nodes: Vec<Node>,
    entry: Option<u32>,
    max_level: usize,
}

impl Hnsw {
    /// New empty index for `dim`-wide vectors.
    pub fn new(dim: usize, cfg: HnswConfig) -> Self {
        Hnsw {
            dim,
            cfg,
            ml: 1.0 / (cfg.m.max(2) as f64).ln(),
            nodes: Vec::new(),
            entry: None,
            max_level: 0,
        }
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Similarity score between two vectors — **higher = closer** for every
    /// metric (L2 returns negative squared distance).
    fn sim(&self, a: &[f32], a_norm: f32, b: &[f32], b_norm: f32) -> f32 {
        let n = a.len().min(b.len());
        match self.cfg.metric {
            Metric::L2 => {
                let mut d = 0.0f32;
                for i in 0..n {
                    let x = a[i] - b[i];
                    d += x * x;
                }
                -d
            }
            Metric::Dot => {
                let mut d = 0.0f32;
                for i in 0..n {
                    d += a[i] * b[i];
                }
                d
            }
            Metric::Cosine => {
                let mut d = 0.0f32;
                for i in 0..n {
                    d += a[i] * b[i];
                }
                let den = a_norm * b_norm;
                if den > 1e-12 { d / den } else { 0.0 }
            }
        }
    }

    fn node_sim(&self, q: &[f32], q_norm: f32, id: u32) -> f32 {
        let n = &self.nodes[id as usize];
        self.sim(q, q_norm, &n.vector, n.norm)
    }

    /// Insert `vector`, returning its node id (= insertion order). The id is the
    /// caller's block handle; store the mapping id→block yourself.
    pub fn insert(&mut self, vector: &[f32]) -> u32 {
        let id = self.nodes.len() as u32;
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Deterministic level from the id.
        let level = (-hash_unit(id as u64 + 1).ln() * self.ml).floor() as usize;
        self.nodes.push(Node {
            vector: vector.to_vec(),
            links: vec![Vec::new(); level + 1],
            norm,
        });

        let entry = match self.entry {
            None => {
                self.entry = Some(id);
                self.max_level = level;
                return id;
            }
            Some(e) => e,
        };

        let q = vector;
        let q_norm = norm;
        let old_max = self.max_level;
        // Phase 1: greedy descend from the entry down to `level+1`.
        let mut cur = entry;
        let mut cur_sim = self.node_sim(q, q_norm, cur);
        let mut lvl = self.max_level;
        while lvl > level {
            let mut changed = true;
            while changed {
                changed = false;
                let neighbors = self.nodes[cur as usize]
                    .links
                    .get(lvl)
                    .cloned()
                    .unwrap_or_default();
                for nb in neighbors {
                    let s = self.node_sim(q, q_norm, nb);
                    if s > cur_sim {
                        cur_sim = s;
                        cur = nb;
                        changed = true;
                    }
                }
            }
            lvl -= 1;
        }

        // Phase 2: connect only at levels that already exist in the graph
        // (`≤ old_max`). If the new node's level exceeds the old max, its higher
        // levels stay empty (it becomes the new entry) — connecting there would
        // index into entry nodes that have no slot at that level.
        let mut ep = vec![cur];
        let top = level.min(old_max);
        for l in (0..=top).rev() {
            let mut cands = self.search_layer(q, q_norm, &ep, self.cfg.ef_construction, l);
            // cands sorted best-first.
            let m = if l == 0 { self.cfg.m0 } else { self.cfg.m };
            let selected = self.select_neighbors(q, q_norm, &cands, m);
            // Link id ↔ selected.
            for &nb in &selected {
                self.nodes[id as usize].links[l].push(nb);
                self.nodes[nb as usize].links[l].push(id);
                // Prune nb's neighbor list if it now exceeds the budget.
                let nb_m = if l == 0 { self.cfg.m0 } else { self.cfg.m };
                if self.nodes[nb as usize].links[l].len() > nb_m {
                    let nbrs = self.nodes[nb as usize].links[l].clone();
                    let nb_vec = self.nodes[nb as usize].vector.clone();
                    let nb_norm = self.nodes[nb as usize].norm;
                    let scored: Vec<(u32, f32)> = nbrs
                        .iter()
                        .map(|&x| (x, self.node_sim(&nb_vec, nb_norm, x)))
                        .collect();
                    let pruned = self.select_neighbors_scored(&scored, nb_m);
                    self.nodes[nb as usize].links[l] = pruned;
                }
            }
            // Descend: next level's entry points = this level's candidates.
            ep = cands.drain(..).map(|(idx, _)| idx).collect();
            if ep.is_empty() {
                ep = vec![cur];
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry = Some(id);
        }
        id
    }

    /// Greedy best-first search at a single `level`. Returns candidates sorted
    /// best-first (highest similarity), up to `ef`.
    fn search_layer(
        &self,
        q: &[f32],
        q_norm: f32,
        entry_points: &[u32],
        ef: usize,
        level: usize,
    ) -> Vec<(u32, f32)> {
        use std::collections::HashSet;
        let mut visited: HashSet<u32> = HashSet::new();
        // `cand` = frontier (explore best-first). `result` = best-so-far (worst-first drop).
        let mut cand: Vec<(u32, f32)> = Vec::new();
        let mut result: Vec<(u32, f32)> = Vec::new();
        for &e in entry_points {
            if visited.insert(e) {
                let s = self.node_sim(q, q_norm, e);
                cand.push((e, s));
                result.push((e, s));
            }
        }
        // Explore.
        while !cand.is_empty() {
            // Pop best candidate.
            let (bi, _) = cand
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    a.1.1
                        .partial_cmp(&b.1.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, &(id, s))| (i, (id, s)))
                .unwrap();
            let (c, c_sim) = cand.swap_remove(bi);
            // Worst in result — if the frontier best can't beat it and result is full, stop.
            let worst = result.iter().map(|&(_, s)| s).fold(f32::INFINITY, f32::min);
            if result.len() >= ef && c_sim < worst {
                break;
            }
            let neighbors = self.nodes[c as usize]
                .links
                .get(level)
                .cloned()
                .unwrap_or_default();
            for nb in neighbors {
                if visited.insert(nb) {
                    let s = self.node_sim(q, q_norm, nb);
                    let worst = result.iter().map(|&(_, x)| x).fold(f32::INFINITY, f32::min);
                    if result.len() < ef || s > worst {
                        cand.push((nb, s));
                        result.push((nb, s));
                        if result.len() > ef {
                            // Drop the worst.
                            let (wi, _) = result
                                .iter()
                                .enumerate()
                                .min_by(|a, b| {
                                    a.1.1
                                        .partial_cmp(&b.1.1)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .map(|(i, &(id, sc))| (i, (id, sc)))
                                .unwrap();
                            result.swap_remove(wi);
                        }
                    }
                }
            }
        }
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Similarity between two existing nodes (node `a`'s vector vs node `b`).
    fn node_pair_sim(&self, a: u32, b: u32) -> f32 {
        let na = &self.nodes[a as usize];
        self.sim(
            &na.vector,
            na.norm,
            &self.nodes[b as usize].vector,
            self.nodes[b as usize].norm,
        )
    }

    /// Neighbor selection. `heuristic` picks the HNSW **diversity heuristic**
    /// (Malkov & Yashunin, Alg. 4); `false` is naive top-m by similarity.
    fn select_neighbors(
        &self,
        _q: &[f32],
        _q_norm: f32,
        cands: &[(u32, f32)],
        m: usize,
    ) -> Vec<u32> {
        self.select_neighbors_scored(cands, m)
    }

    /// **HNSW heuristic neighbor selection** (Malkov & Yashunin, Algorithm 4).
    /// Walk candidates nearest-first and keep `e` only if it is MORE similar to the
    /// base than to any neighbor already kept — so selected links point in *diverse*
    /// directions and long-range "bridge" links (e.g. to an outlier) survive,
    /// instead of collapsing onto one dense cluster. Naive top-m pruning deletes an
    /// outlier's back-links (it's dissimilar to the cluster) and orphans it → 0%
    /// recall; the heuristic preserves them. Backfills to `m` from leftovers so node
    /// degree isn't wasted (HNSW `keepPrunedConnections`).
    fn select_neighbors_heuristic(&self, cands: &[(u32, f32)], m: usize) -> Vec<u32> {
        let mut c: Vec<(u32, f32)> = cands.to_vec();
        c.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut kept: Vec<(u32, f32)> = Vec::with_capacity(m);
        for &(e, sim_eq) in &c {
            if kept.len() >= m {
                break;
            }
            // Keep e iff it's closer to the base than to every already-kept r.
            if kept
                .iter()
                .all(|&(r, _)| sim_eq >= self.node_pair_sim(e, r))
            {
                kept.push((e, sim_eq));
            }
        }
        if kept.len() < m {
            for &(e, sim_eq) in &c {
                if kept.len() >= m {
                    break;
                }
                if !kept.iter().any(|&(r, _)| r == e) {
                    kept.push((e, sim_eq));
                }
            }
        }
        kept.into_iter().map(|(id, _)| id).collect()
    }

    fn select_neighbors_scored(&self, cands: &[(u32, f32)], m: usize) -> Vec<u32> {
        self.select_neighbors_heuristic(cands, m)
    }

    /// Level-0 graph neighbors of `id` — the blocks HNSW considers most similar
    /// to it. Used to *expand* a retrieval hit with its semantic neighborhood for
    /// richer context (the small-world graph already encodes this adjacency, so
    /// expansion is free — no extra search).
    pub fn neighbors(&self, id: u32) -> &[u32] {
        self.nodes
            .get(id as usize)
            .and_then(|n| n.links.first())
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Approximate top-`k` most relevant node ids for `query`, best-first.
    /// `ef` (≥ k) trades recall for cost; `ef = max(k, 64)` is a good default.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        let entry = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };
        let q_norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Descend upper levels greedily to a single entry point.
        let mut cur = entry;
        let mut cur_sim = self.node_sim(query, q_norm, cur);
        let mut lvl = self.max_level;
        while lvl > 0 {
            let mut changed = true;
            while changed {
                changed = false;
                let neighbors = self.nodes[cur as usize]
                    .links
                    .get(lvl)
                    .cloned()
                    .unwrap_or_default();
                for nb in neighbors {
                    let s = self.node_sim(query, q_norm, nb);
                    if s > cur_sim {
                        cur_sim = s;
                        cur = nb;
                        changed = true;
                    }
                }
            }
            lvl -= 1;
        }
        let mut res = self.search_layer(query, q_norm, &[cur], ef.max(k), 0);
        res.truncate(k);
        res
    }

    /// **Fuzzy** top-k: like [`search`](Self::search) but drops matches whose
    /// score is below `min_score`, so weak/irrelevant hits aren't forced in when
    /// there are fewer than `k` confident matches. Returns 0..=k results.
    pub fn search_fuzzy(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        min_score: f32,
    ) -> Vec<(u32, f32)> {
        self.search(query, k, ef)
            .into_iter()
            .filter(|&(_, s)| s >= min_score)
            .collect()
    }

    /// **Radius / range** query: every node within a similarity `threshold`
    /// (score ≥ threshold) of `query`, up to `max` results, best-first. Tolerant
    /// (fuzzy) recall — returns *all* sufficiently-relevant blocks rather than a
    /// fixed count, so context that clears the relevance bar isn't cut off by k.
    /// For `L2`, `threshold` is a negative squared distance (e.g. `-r²`).
    pub fn search_radius(
        &self,
        query: &[f32],
        threshold: f32,
        max: usize,
        ef: usize,
    ) -> Vec<(u32, f32)> {
        let entry = match self.entry {
            Some(e) => e,
            None => return Vec::new(),
        };
        let q_norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut cur = entry;
        let mut cur_sim = self.node_sim(query, q_norm, cur);
        let mut lvl = self.max_level;
        while lvl > 0 {
            let mut changed = true;
            while changed {
                changed = false;
                let neighbors = self.nodes[cur as usize]
                    .links
                    .get(lvl)
                    .cloned()
                    .unwrap_or_default();
                for nb in neighbors {
                    let s = self.node_sim(query, q_norm, nb);
                    if s > cur_sim {
                        cur_sim = s;
                        cur = nb;
                        changed = true;
                    }
                }
            }
            lvl -= 1;
        }
        // Widen the level-0 exploration (ef ≥ max) then keep everything in range.
        let mut res = self.search_layer(query, q_norm, &[cur], ef.max(max), 0);
        res.retain(|&(_, s)| s >= threshold);
        res.truncate(max);
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_topk(vecs: &[Vec<f32>], q: &[f32], k: usize) -> Vec<u32> {
        let mut s: Vec<(u32, f32)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, v.iter().zip(q).map(|(a, b)| a * b).sum::<f32>()))
            .collect();
        s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        s.truncate(k);
        s.into_iter().map(|(i, _)| i).collect()
    }

    // Deterministic pseudo-random vectors (no RNG dep).
    fn gen_vecs(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| {
                        let h = super::hash_unit((i as u64) << 20 ^ (j as u64 + 1));
                        (h as f32) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn empty_search_is_empty() {
        let h = Hnsw::new(8, HnswConfig::default());
        assert!(h.search(&[0.0; 8], 5, 32).is_empty());
    }

    #[test]
    fn recall_matches_brute_force_top1() {
        let dim = 32;
        let vecs = gen_vecs(2000, dim);
        let mut h = Hnsw::new(dim, HnswConfig::default());
        for v in &vecs {
            h.insert(v);
        }
        assert_eq!(h.len(), 2000);
        // top-1 recall over many queries should be high.
        let mut hits = 0;
        let queries = gen_vecs(100, dim);
        for (qi, q) in queries.iter().enumerate() {
            let _ = qi;
            let got = h.search(q, 1, 64);
            let want = brute_topk(&vecs, q, 1);
            if !got.is_empty() && got[0].0 == want[0] {
                hits += 1;
            }
        }
        assert!(hits >= 90, "top-1 recall too low: {hits}/100");
    }

    // REGRESSION GUARD for the "HNSW recall 0% at scale" bug (1M telemetry).
    // A dense CLUSTER of near-identical vectors + a few OUTLIER "needles" (distinct
    // directions). Naive top-m selection ORPHANED the outliers (0/8 — their back-
    // links from cluster nodes got pruned as dissimilar, so greedy nav never entered
    // them). The HNSW diversity heuristic (Alg. 4) preserves those bridge links →
    // outliers stay reachable. Brute force is the 100% oracle.
    #[test]
    fn outlier_needles_expose_naive_neighbor_selection() {
        let dim = 32;
        let n_cluster = 3000usize;
        let n_needle = 8usize;
        // Cluster: all near a base direction (small per-dim jitter) → one dense blob.
        let base: Vec<f32> = (0..dim).map(|j| if j == 0 { 1.0 } else { 0.0 }).collect();
        let mut vecs: Vec<Vec<f32>> = (0..n_cluster)
            .map(|i| {
                let mut v = base.clone();
                for (j, x) in v.iter_mut().enumerate() {
                    *x += 0.05
                        * (super::hash_unit((i as u64) << 20 ^ (j as u64 + 1)) as f32 * 2.0 - 1.0);
                }
                v
            })
            .collect();
        // Needles: distinct one-hot-ish outlier directions (dims 5..5+n_needle).
        let needle_ids: Vec<u32> = (0..n_needle)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[5 + i] = 1.0;
                vecs.push(v);
                (n_cluster + i) as u32
            })
            .collect();
        let mut h = Hnsw::new(dim, HnswConfig::default());
        for v in &vecs {
            h.insert(v);
        }
        let mut hnsw_hits = 0usize;
        let mut brute_hits = 0usize;
        for (i, &nid) in needle_ids.iter().enumerate() {
            let mut q = vec![0.0f32; dim];
            q[5 + i] = 1.0; // query == needle direction
            if h.search(&q, 4, 400).iter().any(|&(id, _)| id == nid) {
                hnsw_hits += 1;
            }
            if brute_topk(&vecs, &q, 4).contains(&nid) {
                brute_hits += 1;
            }
        }
        eprintln!(
            "[hnsw-diag] outlier needles: HNSW {hnsw_hits}/{n_needle}, brute {brute_hits}/{n_needle}"
        );
        assert_eq!(
            brute_hits, n_needle,
            "brute force must find every outlier needle"
        );
        // With the diversity heuristic the outliers keep their in-links and stay
        // reachable — HNSW must now find (nearly) all of them (was 0/8 before).
        assert!(
            hnsw_hits >= n_needle - 1,
            "HNSW orphaned outliers ({hnsw_hits}/{n_needle}) — diversity heuristic regressed"
        );
    }

    #[test]
    fn recall_at_10_is_high() {
        let dim = 24;
        let vecs = gen_vecs(3000, dim);
        let mut h = Hnsw::new(dim, HnswConfig::default());
        for v in &vecs {
            h.insert(v);
        }
        let queries = gen_vecs(50, dim);
        let mut overlap = 0usize;
        for q in &queries {
            let got: std::collections::HashSet<u32> =
                h.search(q, 10, 100).into_iter().map(|(i, _)| i).collect();
            let want: std::collections::HashSet<u32> =
                brute_topk(&vecs, q, 10).into_iter().collect();
            overlap += got.intersection(&want).count();
        }
        // ≥85% of the true top-10 recovered on average.
        assert!(
            overlap >= 50 * 10 * 85 / 100,
            "recall@10 too low: {overlap}/500"
        );
    }

    #[test]
    fn l2_metric_recall() {
        let dim = 32;
        let vecs = gen_vecs(1500, dim);
        let mut h = Hnsw::new(
            dim,
            HnswConfig {
                metric: Metric::L2,
                ..Default::default()
            },
        );
        for v in &vecs {
            h.insert(v);
        }
        let queries = gen_vecs(50, dim);
        let mut hits = 0;
        for q in &queries {
            let mut s: Vec<(u32, f32)> = vecs
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    (
                        i as u32,
                        -v.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(),
                    )
                })
                .collect();
            s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let got = h.search(q, 1, 64);
            if !got.is_empty() && got[0].0 == s[0].0 {
                hits += 1;
            }
        }
        assert!(hits >= 44, "L2 top-1 recall too low: {hits}/50");
    }

    #[test]
    fn fuzzy_and_radius_filter_by_score() {
        let dim = 24;
        let vecs = gen_vecs(1000, dim);
        let mut h = Hnsw::new(dim, HnswConfig::default()); // Dot
        for v in &vecs {
            h.insert(v);
        }
        // Query = a stored vector → its self-dot (‖v‖²) is a high score.
        let q = &vecs[42];
        let self_score = q.iter().map(|x| x * x).sum::<f32>();
        // Radius at 90% of the self score returns a bounded, in-range set incl. #42.
        let r = h.search_radius(q, self_score * 0.9, 50, 128);
        assert!(
            r.iter().all(|&(_, s)| s >= self_score * 0.9),
            "radius kept out-of-range"
        );
        assert!(
            r.iter().any(|&(id, _)| id == 42),
            "radius missed the exact match"
        );
        assert!(r.len() < vecs.len(), "radius should not return everything");
        // Fuzzy top-k with a floor above most matches → few strong hits.
        let f = h.search_fuzzy(q, 20, 128, self_score * 0.9);
        assert!(f.len() <= 20 && f.iter().all(|&(_, s)| s >= self_score * 0.9));
    }

    #[test]
    fn deterministic_rebuild() {
        let dim = 16;
        let vecs = gen_vecs(500, dim);
        let build = || {
            let mut h = Hnsw::new(dim, HnswConfig::default());
            for v in &vecs {
                h.insert(v);
            }
            h.search(&vecs[0], 10, 64)
        };
        assert_eq!(build(), build());
    }
}
