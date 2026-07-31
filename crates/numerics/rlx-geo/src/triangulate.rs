// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serial Guibas-Stolfi divide-and-conquer Delaunay triangulation over signed
//! integer sites, using a compact two-dart primal edge ring (`sym(e) = e ^ 1`).
//! Exact integer predicates make the result robust and deterministic.

use crate::predicates::{Pred, PredFast, PredWide, PredicateWidth, predicate_width};
use std::marker::PhantomData;

const DELETED: u32 = u32::MAX;

/// Prepared sites: unique coordinates, their original indices, and the x/y spans.
type Prepared = (Vec<[i32; 2]>, Vec<u32>, i64, i64);

/// Triangulate `points`, returning counterclockwise triangles as indices into
/// the original `points` slice. Coincident points collapse (the lowest original
/// index is kept); collinear input yields no triangles.
///
/// # Panics
/// If the coordinate span exceeds [`crate::predicates::MAX_COORDINATE_SPAN`].
pub fn triangulate(points: &[[i32; 2]]) -> Vec<[u32; 3]> {
    let prof = std::env::var_os("GEO_PROF").is_some();
    let t0 = std::time::Instant::now();
    let Some((coord, orig, sx, sy)) = prepare(points) else {
        return Vec::new();
    };
    if prof {
        eprintln!(
            "  prepare(sort+dedup): {:.2} ms",
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    let mut out = Vec::new();
    match predicate_width(sx, sy) {
        PredicateWidth::Int64 => run_dwyer::<PredFast>(&coord, &orig, &mut out),
        PredicateWidth::Int128 => run::<PredWide>(&coord, &orig, &mut out),
        PredicateWidth::Unsupported => {
            panic!("rlx-geo: coordinate span exceeds MAX_COORDINATE_SPAN")
        }
    }
    out
}

/// Parallel triangulation: split sorted sites into contiguous x-ranges, build
/// each on its own thread (`std::thread::scope`), then zip the pieces
/// left-to-right with the same Guibas-Stolfi merge. `threads`: 0 = auto.
/// Below [`PARALLEL_MIN`] sites it runs serially (thread overhead isn't worth it).
pub fn triangulate_par(points: &[[i32; 2]], threads: usize) -> Vec<[u32; 3]> {
    let t = if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    };
    let tp = std::time::Instant::now();
    let Some((coord, orig, sx, sy)) = prepare_par(points, t) else {
        return Vec::new();
    };
    if std::env::var_os("GEO_PROF").is_some() {
        eprintln!(
            "  par prepare (sort):     {:.2} ms",
            tp.elapsed().as_secs_f64() * 1e3
        );
    }
    let mut out = Vec::new();
    match predicate_width(sx, sy) {
        PredicateWidth::Int64 => run_par::<PredFast>(&coord, &orig, t, &mut out),
        PredicateWidth::Int128 => run_par::<PredWide>(&coord, &orig, t, &mut out),
        PredicateWidth::Unsupported => {
            panic!("rlx-geo: coordinate span exceeds MAX_COORDINATE_SPAN")
        }
    }
    out
}

/// Parallel `prepare`: bucket-sort sites by x (buckets are x-ordered, so no
/// merge), sort each contiguous bucket-group in parallel, then dedup.
fn prepare_par(points: &[[i32; 2]], threads: usize) -> Option<Prepared> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    if threads <= 1 || n < PARALLEL_MIN {
        return prepare(points);
    }
    let mut sites: Vec<(i32, i32, u32)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| (p[0], p[1], i as u32))
        .collect();
    let (mut mnx, mut mxx) = (i32::MAX, i32::MIN);
    for s in &sites {
        mnx = mnx.min(s.0);
        mxx = mxx.max(s.0);
    }
    bucket_sort_by_x(&mut sites, threads, mnx, mxx);
    finish_sites(&sites)
}

/// Sort `sites` by (x, y, original) via an x-bucket sort. Buckets partition the
/// x-range so they're globally ordered; contiguous groups of whole buckets are
/// sorted in parallel with `split_at_mut` (safe, no cross-thread aliasing).
fn bucket_sort_by_x(sites: &mut Vec<(i32, i32, u32)>, threads: usize, mnx: i32, mxx: i32) {
    let n = sites.len();
    let b = (threads * 16).clamp(1, 8192);
    let span = (mxx as i64 - mnx as i64 + 1).max(1);
    let bucket = move |x: i32| -> usize {
        ((((x as i64 - mnx as i64) * b as i64) / span) as usize).min(b - 1)
    };

    let t = threads.max(1);
    let per = n.div_ceil(t.max(1)).max(1);
    let nt = n.div_ceil(per); // actual number of input chunks

    // Phase 1: per-chunk local histograms (parallel).
    let locals: Vec<Vec<u32>> = std::thread::scope(|s| {
        let handles: Vec<_> = sites
            .chunks(per)
            .map(|chunk| {
                s.spawn(move || {
                    let mut h = vec![0u32; b];
                    for &(x, _, _) in chunk {
                        h[bucket(x)] += 1;
                    }
                    h
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Phase 2: global bucket starts (prefix sum) + per-chunk scatter offsets.
    let mut start = vec![0u32; b + 1];
    for bk in 0..b {
        let mut tot = 0u32;
        for local in &locals {
            tot += local[bk];
        }
        start[bk + 1] = tot;
    }
    for i in 0..b {
        start[i + 1] += start[i];
    }
    // off[c][bk] = where chunk c's bucket-bk elements begin in the output.
    let mut off = vec![vec![0u32; b]; nt];
    for bk in 0..b {
        let mut acc = start[bk];
        for c in 0..nt {
            off[c][bk] = acc;
            acc += locals[c][bk];
        }
    }

    // Phase 3: parallel scatter to disjoint positions (raw-pointer, sound because
    // each (chunk, bucket) writes a distinct, non-overlapping output range).
    let mut out = vec![(0i32, 0i32, 0u32); n];
    let base_addr = out.as_mut_ptr() as usize; // usize is Send+Copy (avoids raw-ptr capture)
    std::thread::scope(|s| {
        for (c, chunk) in sites.chunks(per).enumerate() {
            let mut o = off[c].clone();
            s.spawn(move || {
                let ptr = base_addr as *mut (i32, i32, u32);
                for &st in chunk {
                    let bk = bucket(st.0);
                    unsafe {
                        *ptr.add(o[bk] as usize) = st;
                    }
                    o[bk] += 1;
                }
            });
        }
    });

    // Split at bucket boundaries into ~threads balanced groups.
    let target = n / threads;
    let mut splits: Vec<usize> = Vec::with_capacity(threads);
    let mut next = target;
    for &s in start.iter().take(b).skip(1) {
        let off = s as usize;
        if off >= next && splits.len() + 1 < threads {
            splits.push(off);
            next += target;
        }
    }
    // Carve `out` into contiguous slices at those offsets and sort each in parallel.
    let mut slices: Vec<&mut [(i32, i32, u32)]> = Vec::with_capacity(splits.len() + 1);
    let mut rem = &mut out[..];
    let mut prev = 0usize;
    for &sp in &splits {
        let (head, tail) = rem.split_at_mut(sp - prev);
        slices.push(head);
        rem = tail;
        prev = sp;
    }
    slices.push(rem);
    std::thread::scope(|s| {
        for sl in slices {
            s.spawn(move || sl.sort_unstable());
        }
    });

    *sites = out;
}

/// Sites below this count use the serial path.
pub const PARALLEL_MIN: usize = 50_000;

/// Sort by (x,y), deduplicate (lowest original index kept), return the unique
/// coordinates, their original indices, and the x/y spans. `None` if < 3 unique.
fn prepare(points: &[[i32; 2]]) -> Option<Prepared> {
    if points.len() < 3 {
        return None;
    }
    let mut sites: Vec<(i32, i32, u32)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| (p[0], p[1], i as u32))
        .collect();
    sites.sort_unstable();
    finish_sites(&sites)
}

fn finish_sites(sites: &[(i32, i32, u32)]) -> Option<Prepared> {
    let mut coord: Vec<[i32; 2]> = Vec::with_capacity(sites.len());
    let mut orig: Vec<u32> = Vec::with_capacity(sites.len());
    let mut last: Option<(i32, i32)> = None;
    for &(x, y, o) in sites {
        if last != Some((x, y)) {
            coord.push([x, y]);
            orig.push(o);
            last = Some((x, y));
        }
    }
    if coord.len() < 3 {
        return None;
    }
    let (mut mnx, mut mny, mut mxx, mut mxy) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for p in &coord {
        mnx = mnx.min(p[0]);
        mny = mny.min(p[1]);
        mxx = mxx.max(p[0]);
        mxy = mxy.max(p[1]);
    }
    Some((
        coord,
        orig,
        mxx as i64 - mnx as i64,
        mxy as i64 - mny as i64,
    ))
}

fn run<P: Pred>(coord: &[[i32; 2]], orig: &[u32], out: &mut Vec<[u32; 3]>) {
    let prof = std::env::var_os("GEO_PROF").is_some();
    let m = coord.len();
    let mut arena = Arena::<P>::with_capacity(coord, m.saturating_mul(8));
    let t = std::time::Instant::now();
    arena.delaunay(0, m as u32);
    if prof {
        eprintln!(
            "  build(delaunay):     {:.2} ms  ({} darts)",
            t.elapsed().as_secs_f64() * 1e3,
            arena.darts.len()
        );
    }
    let t = std::time::Instant::now();
    arena.export(orig, out);
    if prof {
        eprintln!(
            "  export:              {:.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
}

fn run_par<P: Pred>(coord: &[[i32; 2]], orig: &[u32], threads: usize, out: &mut Vec<[u32; 3]>) {
    let m = coord.len();
    let nchunks = if threads <= 1 || m < PARALLEL_MIN {
        1
    } else {
        threads.min(m / 2000).max(1)
    };
    if nchunks == 1 {
        return run::<P>(coord, orig, out);
    }
    let bounds: Vec<(u32, u32)> = (0..nchunks)
        .map(|i| ((i * m / nchunks) as u32, ((i + 1) * m / nchunks) as u32))
        .collect();
    let prof = std::env::var_os("GEO_PROF").is_some();
    let tb = std::time::Instant::now();
    let mut pieces: Vec<Piece> = std::thread::scope(|s| {
        let handles: Vec<_> = bounds
            .iter()
            .map(|&(lo, hi)| s.spawn(move || build_piece::<P>(coord, lo, hi)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    if prof {
        eprintln!(
            "  par build ({nchunks} chunks): {:.2} ms",
            tb.elapsed().as_secs_f64() * 1e3
        );
    }
    let tm = std::time::Instant::now();
    let total: usize = pieces.iter().map(|p| p.darts.len()).sum::<usize>() + m * 2;
    let mut acc = Arena::<P>::with_capacity(coord, total);
    let mut drain = pieces.drain(..);
    let first = drain.next().unwrap();
    let (mut le, mut re) = (first.le, first.re);
    acc.append_piece(first);
    for p in drain {
        let base = acc.len();
        let (ple, pre) = (p.le + base, p.re + base);
        acc.append_piece(p);
        let (nle, nre) = acc.merge(le, re, ple, pre);
        le = nle;
        re = nre;
    }
    if prof {
        eprintln!(
            "  concat+trunk merge:     {:.2} ms",
            tm.elapsed().as_secs_f64() * 1e3
        );
    }
    let te = std::time::Instant::now();
    acc.export_par(orig, threads, out);
    if prof {
        eprintln!(
            "  par export:             {:.2} ms",
            te.elapsed().as_secs_f64() * 1e3
        );
    }
}

fn build_piece<P: Pred>(coord: &[[i32; 2]], lo: u32, hi: u32) -> Piece {
    // Dwyer (compact pieces) on the fast path; x-cut on the wide path (coords
    // don't fit 16-bit Morton).
    if P::USE_MORTON {
        return build_piece_dwyer::<P>(coord, lo, hi);
    }
    let n = (hi - lo) as usize;
    let mut arena = Arena::<P>::with_capacity(coord, n.saturating_mul(8));
    let (le, re) = arena.delaunay(lo, hi);
    arena.into_piece(le, re)
}

/// A finished sub-triangulation with its two outer hull darts.
/// Each dart stores `[org, next, prev]` (Array-of-Structs: one cache line per
/// dart, so multi-field ops touch 1 line instead of 3 — a win on the
/// latency-bound build).
struct Piece {
    darts: Vec<[u32; 3]>,
    le: u32,
    re: u32,
}

/// Dart arena: two consecutive darts per undirected edge; `darts[e] = [org,
/// next, prev]`.
struct Arena<'c, P: Pred> {
    coord: &'c [[i32; 2]],
    darts: Vec<[u32; 3]>,
    /// Even bases of deleted edges, reused by `make_edge` so the arena stays at
    /// peak-alive size (~6 darts/point) instead of growing append-only (~18/pt).
    free: Vec<u32>,
    _p: PhantomData<P>,
}

impl<'c, P: Pred> Arena<'c, P> {
    fn with_capacity(coord: &'c [[i32; 2]], cap: usize) -> Self {
        Arena {
            coord,
            darts: Vec::with_capacity(cap),
            free: Vec::new(),
            _p: PhantomData,
        }
    }

    #[inline(always)]
    fn len(&self) -> u32 {
        self.darts.len() as u32
    }

    /// Append a finished piece; dart links are shifted by the current base.
    fn append_piece(&mut self, p: Piece) {
        let base = self.darts.len() as u32;
        // org (index 0) is a site id — unchanged; next/prev (1,2) shift by base.
        self.darts
            .extend(p.darts.iter().map(|&[o, n, pr]| [o, n + base, pr + base]));
    }

    fn into_piece(self, le: u32, re: u32) -> Piece {
        Piece {
            darts: self.darts,
            le,
            re,
        }
    }

    // --- navigation (indices are valid by construction) ---
    #[inline(always)]
    fn sym(e: u32) -> u32 {
        e ^ 1
    }
    #[inline(always)]
    fn org(&self, e: u32) -> u32 {
        unsafe { self.darts.get_unchecked(e as usize)[0] }
    }
    #[inline(always)]
    fn dest(&self, e: u32) -> u32 {
        unsafe { self.darts.get_unchecked((e ^ 1) as usize)[0] }
    }
    #[inline(always)]
    fn onext(&self, e: u32) -> u32 {
        unsafe { self.darts.get_unchecked(e as usize)[1] }
    }
    #[inline(always)]
    fn oprev(&self, e: u32) -> u32 {
        unsafe { self.darts.get_unchecked(e as usize)[2] }
    }
    #[inline(always)]
    fn lnext(&self, e: u32) -> u32 {
        unsafe { self.darts.get_unchecked((e ^ 1) as usize)[2] }
    }
    #[inline(always)]
    fn rprev(&self, e: u32) -> u32 {
        unsafe { self.darts.get_unchecked((e ^ 1) as usize)[1] }
    }

    // --- mutation ---
    fn make_edge(&mut self, a: u32, b: u32) -> u32 {
        if let Some(e) = self.free.pop() {
            // Reuse a deleted slot (e is an even base < len; e^1 is its pair).
            let i = e as usize;
            unsafe {
                *self.darts.get_unchecked_mut(i) = [a, e, e];
                *self.darts.get_unchecked_mut(i + 1) = [b, e + 1, e + 1];
            }
            return e;
        }
        let e = self.darts.len() as u32;
        self.darts.push([a, e, e]);
        self.darts.push([b, e + 1, e + 1]);
        e
    }
    fn splice(&mut self, a: u32, b: u32) {
        let (a, b) = (a as usize, b as usize);
        unsafe {
            let an = self.darts.get_unchecked(a)[1];
            let bn = self.darts.get_unchecked(b)[1];
            self.darts.get_unchecked_mut(a)[1] = bn;
            self.darts.get_unchecked_mut(bn as usize)[2] = a as u32;
            self.darts.get_unchecked_mut(b)[1] = an;
            self.darts.get_unchecked_mut(an as usize)[2] = b as u32;
        }
    }
    fn connect(&mut self, a: u32, b: u32) -> u32 {
        let e = self.make_edge(self.dest(a), self.org(b));
        let la = self.lnext(a);
        self.splice(e, la);
        let s = Self::sym(e);
        self.splice(s, b);
        e
    }
    fn delete_edge(&mut self, e: u32) {
        let op = self.oprev(e);
        self.splice(e, op);
        let se = Self::sym(e);
        let ops = self.oprev(se);
        self.splice(se, ops);
        let base = e & !1; // even base of the (e, e^1) pair
        unsafe {
            self.darts.get_unchecked_mut(base as usize)[0] = DELETED;
            self.darts.get_unchecked_mut((base | 1) as usize)[0] = DELETED;
        }
        self.free.push(base);
    }

    // --- predicates on site ids ---
    #[inline(always)]
    fn pt(&self, s: u32) -> [i32; 2] {
        unsafe { *self.coord.get_unchecked(s as usize) }
    }
    #[inline(always)]
    fn orient3(&self, a: u32, b: u32, c: u32) -> i32 {
        P::orient(self.pt(a), self.pt(b), self.pt(c))
    }
    #[inline(always)]
    fn in_circle4(&self, a: u32, b: u32, c: u32, d: u32) -> bool {
        P::in_circle(self.pt(a), self.pt(b), self.pt(c), self.pt(d)) > 0
    }
    #[inline(always)]
    fn left_of(&self, x: u32, e: u32) -> bool {
        self.orient3(x, self.org(e), self.dest(e)) > 0
    }
    #[inline(always)]
    fn right_of(&self, x: u32, e: u32) -> bool {
        self.orient3(x, self.dest(e), self.org(e)) > 0
    }

    // --- divide & conquer over site-id range [lo, hi) ---
    fn delaunay(&mut self, lo: u32, hi: u32) -> (u32, u32) {
        let n = hi - lo;
        if n == 2 {
            let a = self.make_edge(lo, lo + 1);
            return (a, Self::sym(a));
        }
        if n == 3 {
            let (s1, s2, s3) = (lo, lo + 1, lo + 2);
            let a = self.make_edge(s1, s2);
            let b = self.make_edge(s2, s3);
            let sa = Self::sym(a);
            self.splice(sa, b);
            let o = self.orient3(s1, s2, s3);
            if o > 0 {
                self.connect(b, a);
                return (a, Self::sym(b));
            } else if o < 0 {
                let c = self.connect(b, a);
                return (Self::sym(c), c);
            } else {
                return (a, Self::sym(b));
            }
        }
        let mid = lo + n / 2;
        let (ldo, ldi) = self.delaunay(lo, mid);
        let (rdi, rdo) = self.delaunay(mid, hi);
        self.merge(ldo, ldi, rdi, rdo)
    }

    fn merge(&mut self, mut ldo: u32, mut ldi: u32, mut rdi: u32, mut rdo: u32) -> (u32, u32) {
        loop {
            if self.left_of(self.org(rdi), ldi) {
                ldi = self.lnext(ldi);
            } else if self.right_of(self.org(ldi), rdi) {
                rdi = self.rprev(rdi);
            } else {
                break;
            }
        }
        let mut basel = self.connect(Self::sym(rdi), ldi);
        if self.org(ldi) == self.org(ldo) {
            ldo = Self::sym(basel);
        }
        if self.org(rdi) == self.org(rdo) {
            rdo = basel;
        }
        loop {
            // basel's endpoints are loop-invariant across the two while-loops;
            // hoist them and inline valid(e) := orient(dest(e), db, ob) > 0.
            let db = self.dest(basel);
            let ob = self.org(basel);
            let sb = Self::sym(basel);

            let mut lcand = self.onext(sb);
            let mut l_valid = self.orient3(self.dest(lcand), db, ob) > 0;
            if l_valid {
                loop {
                    let ln = self.onext(lcand); // computed once per step
                    if self.in_circle4(db, ob, self.dest(lcand), self.dest(ln)) {
                        self.delete_edge(lcand);
                        lcand = ln;
                    } else {
                        break;
                    }
                }
                l_valid = self.orient3(self.dest(lcand), db, ob) > 0;
            }

            let mut rcand = self.oprev(basel);
            let mut r_valid = self.orient3(self.dest(rcand), db, ob) > 0;
            if r_valid {
                loop {
                    let rp = self.oprev(rcand);
                    if self.in_circle4(db, ob, self.dest(rcand), self.dest(rp)) {
                        self.delete_edge(rcand);
                        rcand = rp;
                    } else {
                        break;
                    }
                }
                r_valid = self.orient3(self.dest(rcand), db, ob) > 0;
            }

            if !l_valid && !r_valid {
                break;
            }
            if !l_valid
                || (r_valid
                    && self.in_circle4(
                        self.dest(lcand),
                        self.org(lcand),
                        self.org(rcand),
                        self.dest(rcand),
                    ))
            {
                basel = self.connect(rcand, sb);
            } else {
                basel = self.connect(sb, Self::sym(lcand));
            }
        }
        (ldo, rdo)
    }

    /// Emit the interior CCW triangle whose minimum dart is `e` (if any).
    #[inline(always)]
    fn emit_face(&self, e: u32, orig: &[u32], out: &mut Vec<[u32; 3]>) {
        if self.org(e) == DELETED {
            return;
        }
        let e1 = self.lnext(e);
        let e2 = self.lnext(e1);
        if self.lnext(e2) == e && e <= e1 && e <= e2 {
            let a = self.org(e);
            let b = self.org(e1);
            let c = self.org(e2);
            if self.orient3(a, b, c) > 0 {
                out.push([orig[a as usize], orig[b as usize], orig[c as usize]]);
            }
        }
    }

    fn export(&self, orig: &[u32], out: &mut Vec<[u32; 3]>) {
        let dart_count = self.darts.len() as u32;
        for e in 0..dart_count {
            self.emit_face(e, orig, out);
        }
    }

    /// Parallel export: each thread scans a disjoint dart range. A triangle is
    /// emitted only from its minimum dart, which lives in exactly one range, so
    /// there are no duplicates and no coordination.
    fn export_par(&self, orig: &[u32], threads: usize, out: &mut Vec<[u32; 3]>) {
        let dc = self.darts.len() as u32;
        let nth = threads.max(1) as u32;
        let per = dc.div_ceil(nth);
        let parts: Vec<Vec<[u32; 3]>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..nth)
                .map(|i| {
                    let lo = i * per;
                    let hi = ((i + 1) * per).min(dc);
                    s.spawn(move || {
                        let mut v = Vec::new();
                        for e in lo..hi {
                            self.emit_face(e, orig, &mut v);
                        }
                        v
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for p in &parts {
            out.extend_from_slice(p);
        }
    }
}

// ============================================================================
// Dwyer alternating-cut build (compact pieces -> fewer cache misses)
// ============================================================================

/// Outer darts at the x- and y-extremes of a piece.
#[derive(Clone, Copy)]
struct DHulls {
    xl: u32, // leftmost  (min x)
    xr: u32, // rightmost (max x)
    yl: u32, // bottommost (min y)
    yr: u32, // topmost    (max y)
}

#[inline]
fn morton_spread(mut v: u32) -> u32 {
    v &= 0xffff;
    v = (v | (v << 8)) & 0x00ff_00ff;
    v = (v | (v << 4)) & 0x0f0f_0f0f;
    v = (v | (v << 2)) & 0x3333_3333;
    (v | (v << 1)) & 0x5555_5555
}
#[inline]
fn morton_code(x: u32, y: u32) -> u32 {
    morton_spread(x) | (morton_spread(y) << 1)
}

impl<'c, P: Pred> Arena<'c, P> {
    /// Delaunay of an explicit x-sorted list of site ids (leaf builder).
    fn delaunay_slice(&mut self, ids: &[u32]) -> (u32, u32) {
        let n = ids.len();
        if n == 2 {
            let a = self.make_edge(ids[0], ids[1]);
            return (a, Self::sym(a));
        }
        if n == 3 {
            let (s1, s2, s3) = (ids[0], ids[1], ids[2]);
            let a = self.make_edge(s1, s2);
            let b = self.make_edge(s2, s3);
            let sa = Self::sym(a);
            self.splice(sa, b);
            let o = self.orient3(s1, s2, s3);
            if o > 0 {
                self.connect(b, a);
                return (a, Self::sym(b));
            } else if o < 0 {
                let c = self.connect(b, a);
                return (Self::sym(c), c);
            } else {
                return (a, Self::sym(b));
            }
        }
        let mid = n / 2;
        let (ldo, ldi) = self.delaunay_slice(&ids[..mid]);
        let (rdi, rdo) = self.delaunay_slice(&ids[mid..]);
        self.merge(ldo, ldi, rdi, rdo)
    }

    /// Walk the outer hull (`lnext`) from `seed` (left face = outer face) and
    /// return all four directional extreme darts.
    fn scan_dhulls(&self, seed: u32) -> DHulls {
        let (mut xl, mut xr, mut yl, mut yr) = (seed, seed, seed, seed);
        let (mut xlx, mut xly) = (i32::MAX, i32::MAX);
        let (mut xrx, mut xry) = (i32::MIN, i32::MAX);
        let (mut yly, mut ylx) = (i32::MAX, i32::MIN);
        let (mut yry, mut yrx) = (i32::MIN, i32::MIN);
        let mut o = seed;
        loop {
            let dp = self.pt(self.dest(o));
            let op = self.pt(self.org(o));
            if dp[0] < xlx || (dp[0] == xlx && dp[1] < xly) {
                xlx = dp[0];
                xly = dp[1];
                xl = Self::sym(o);
            }
            if op[0] > xrx || (op[0] == xrx && op[1] < xry) {
                xrx = op[0];
                xry = op[1];
                xr = o;
            }
            if dp[1] < yly || (dp[1] == yly && dp[0] > ylx) {
                yly = dp[1];
                ylx = dp[0];
                yl = Self::sym(o);
            }
            if op[1] > yry || (op[1] == yry && op[0] > yrx) {
                yry = op[1];
                yrx = op[0];
                yr = o;
            }
            o = self.lnext(o);
            if o == seed {
                break;
            }
        }
        DHulls { xl, xr, yl, yr }
    }

    /// Morton alternating-cut build. `pos[i]` is the global site id at
    /// Morton-index `i`; `keys[i]` its Morton code (both sorted by key).
    fn build_dwyer(&mut self, pos: &[u32], keys: &[u32], lo: u32, hi: u32) -> DHulls {
        let diff = keys[lo as usize] ^ keys[(hi - 1) as usize];
        let split = if diff == 0 {
            None
        } else {
            let bit = 31 - diff.leading_zeros();
            let mask = 1u32 << bit;
            let (mut low, mut high) = (lo, hi);
            while low < high {
                let m = low + (high - low) / 2;
                if keys[m as usize] & mask == 0 {
                    low = m + 1;
                } else {
                    high = m;
                }
            }
            if low - lo < 2 || hi - low < 2 {
                None
            } else {
                Some((low, bit))
            }
        };
        match split {
            Some((mid, bit)) => {
                let left = self.build_dwyer(pos, keys, lo, mid);
                let right = self.build_dwyer(pos, keys, mid, hi);
                let horizontal = bit & 1 == 1;
                let (ll, lr, rl, rr) = if horizontal {
                    (left.yl, left.yr, right.yl, right.yr)
                } else {
                    (left.xl, left.xr, right.xl, right.xr)
                };
                let (mle, mre) = self.merge(ll, lr, rl, rr);
                let mut d = self.scan_dhulls(Self::sym(mle));
                if horizontal {
                    d.yl = mle;
                    d.yr = mre;
                } else {
                    d.xl = mle;
                    d.xr = mre;
                }
                d
            }
            None => {
                let mut ids: Vec<u32> = pos[lo as usize..hi as usize].to_vec();
                ids.sort_unstable_by_key(|&a| self.pt(a));
                let (le, _re) = self.delaunay_slice(&ids);
                self.scan_dhulls(Self::sym(le))
            }
        }
    }
}

/// Morton-sorted positions of `coord[lo..hi]` and their keys (rebased so coords
/// fit 16-bit; valid for the i64 fast-path span).
fn morton_positions(coord: &[[i32; 2]], lo: u32, hi: u32) -> (Vec<u32>, Vec<u32>) {
    let (mut mnx, mut mny) = (i32::MAX, i32::MAX);
    for p in &coord[lo as usize..hi as usize] {
        mnx = mnx.min(p[0]);
        mny = mny.min(p[1]);
    }
    let key = |i: u32| {
        let p = coord[i as usize];
        morton_code((p[0] - mnx) as u32, (p[1] - mny) as u32)
    };
    let mut pos: Vec<u32> = (lo..hi).collect();
    let mut keys: Vec<u32> = pos.iter().map(|&i| key(i)).collect();
    // LSD radix sort pos by the u32 Morton keys (O(n)).
    let n = pos.len();
    let mut pos2 = vec![0u32; n];
    let mut key2 = vec![0u32; n];
    for shift in [0u32, 8, 16, 24] {
        let mut cnt = [0u32; 257];
        for &k in keys.iter() {
            cnt[((k >> shift) & 0xff) as usize + 1] += 1;
        }
        for i in 0..256 {
            cnt[i + 1] += cnt[i];
        }
        for i in 0..n {
            let d = ((keys[i] >> shift) & 0xff) as usize;
            let p = cnt[d] as usize;
            cnt[d] += 1;
            pos2[p] = pos[i];
            key2[p] = keys[i];
        }
        std::mem::swap(&mut pos, &mut pos2);
        std::mem::swap(&mut keys, &mut key2);
    }
    (pos, keys)
}

/// Serial build via Morton alternating cuts (fast path only; span ≤ 29609).
fn run_dwyer<P: Pred>(coord: &[[i32; 2]], orig: &[u32], out: &mut Vec<[u32; 3]>) {
    let m = coord.len();
    let (pos, keys) = morton_positions(coord, 0, m as u32);
    let mut arena = Arena::<P>::with_capacity(coord, m.saturating_mul(8));
    arena.build_dwyer(&pos, &keys, 0, m as u32);
    arena.export(orig, out);
}

/// Build one parallel chunk (coord[lo..hi]) via Dwyer; returns a Piece whose
/// (le, re) are its x-extremes for the left-to-right trunk merge.
fn build_piece_dwyer<P: Pred>(coord: &[[i32; 2]], lo: u32, hi: u32) -> Piece {
    let n = (hi - lo) as usize;
    let (pos, keys) = morton_positions(coord, lo, hi);
    let mut arena = Arena::<P>::with_capacity(coord, n.saturating_mul(8));
    let d = arena.build_dwyer(&pos, &keys, 0, n as u32);
    arena.into_piece(d.xl, d.xr)
}

/// Triangulate using Dwyer alternating cuts (Morton order). Falls back to the
/// x-cut path for the wide predicate range (coords don't fit 16-bit Morton).
pub fn triangulate_dwyer(points: &[[i32; 2]]) -> Vec<[u32; 3]> {
    let Some((coord, orig, sx, sy)) = prepare(points) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    match predicate_width(sx, sy) {
        PredicateWidth::Int64 => run_dwyer::<PredFast>(&coord, &orig, &mut out),
        PredicateWidth::Int128 => run::<PredWide>(&coord, &orig, &mut out),
        PredicateWidth::Unsupported => {
            panic!("rlx-geo: coordinate span exceeds MAX_COORDINATE_SPAN")
        }
    }
    out
}
