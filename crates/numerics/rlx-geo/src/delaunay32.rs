// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Guibas-Stolfi divide-and-conquer exact-integer 2D Delaunay (`Triangulator`), ported
//! from the standalone `delaunay32` crate. viz/export (threers) intentionally omitted.
#![allow(clippy::all)]

// Exact integer 2D Delaunay triangulation via Guibas-Stolfi divide-and-conquer.
//
// Mirrors the design of the C++ `delaunay32` library: signed 32-bit integer
// sites, exact integer predicates (i64 fast path / i128 wide path chosen once),
// and a compact two-dart primal edge ring (`sym(e) = e ^ 1`).
//
// Two execution paths:
//   * serial   — one arena, textbook recursive D&C.
//   * parallel — sites split into T contiguous x-ranges, each triangulated on
//                its own thread (std::thread::scope), then the pieces are
//                concatenated into one arena and zipped left-to-right with the
//                same GS merge. No external dependencies.

use std::marker::PhantomData;

/// A signed integer site.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub fn new(x: i32, y: i32) -> Self {
        Point { x, y }
    }
}

/// Counterclockwise indices into the caller's original point array.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Triangle {
    pub i0: u32,
    pub i1: u32,
    pub i2: u32,
}

/// Largest equal x/y span certified for the i64 fast path.
pub const FAST_COORDINATE_SPAN: i64 = 29_609;
/// Largest equal x/y span certified for the i128 wide path.
pub const MAX_COORDINATE_SPAN: i64 = 1_940_470_527;

/// Below this many sites the parallel path is never used.
const PARALLEL_MIN_POINTS: usize = 50_000;

// ----------------------------------------------------------------------------
// Exact predicates
// ----------------------------------------------------------------------------

trait Pred {
    fn orient(a: Point, b: Point, c: Point) -> i32;
    fn in_circle(a: Point, b: Point, c: Point, d: Point) -> i32;
}

macro_rules! orient_body {
    ($t:ty, $a:expr, $b:expr, $c:expr) => {{
        let v = ($b.x as $t - $a.x as $t) * ($c.y as $t - $a.y as $t)
            - ($b.y as $t - $a.y as $t) * ($c.x as $t - $a.x as $t);
        (v > 0) as i32 - (v < 0) as i32
    }};
}

macro_rules! in_circle_body {
    ($t:ty, $a:expr, $b:expr, $c:expr, $d:expr) => {{
        let ax = $a.x as $t - $d.x as $t;
        let ay = $a.y as $t - $d.y as $t;
        let bx = $b.x as $t - $d.x as $t;
        let by = $b.y as $t - $d.y as $t;
        let cx = $c.x as $t - $d.x as $t;
        let cy = $c.y as $t - $d.y as $t;
        let det = (ax * ax + ay * ay) * (bx * cy - cx * by)
            - (bx * bx + by * by) * (ax * cy - cx * ay)
            + (cx * cx + cy * cy) * (ax * by - bx * ay);
        (det > 0) as i32 - (det < 0) as i32
    }};
}

/// Fast path: both predicates in i64 (span <= FAST_COORDINATE_SPAN).
struct PredFast;
impl Pred for PredFast {
    #[inline(always)]
    fn orient(a: Point, b: Point, c: Point) -> i32 {
        orient_body!(i64, a, b, c)
    }
    #[inline(always)]
    fn in_circle(a: Point, b: Point, c: Point, d: Point) -> i32 {
        in_circle_body!(i64, a, b, c, d)
    }
}

/// Wide path: orientation in i64 (safe to ~3e9 span), in-circle in i128.
struct PredWide;
impl Pred for PredWide {
    #[inline(always)]
    fn orient(a: Point, b: Point, c: Point) -> i32 {
        orient_body!(i64, a, b, c)
    }
    #[inline(always)]
    fn in_circle(a: Point, b: Point, c: Point, d: Point) -> i32 {
        in_circle_body!(i128, a, b, c, d)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PredicateWidth {
    Int64,
    Int128,
    Unsupported,
}

fn predicate_width(sx: i64, sy: i64) -> PredicateWidth {
    let span = sx.max(sy);
    if span <= FAST_COORDINATE_SPAN {
        PredicateWidth::Int64
    } else if span <= MAX_COORDINATE_SPAN {
        PredicateWidth::Int128
    } else {
        PredicateWidth::Unsupported
    }
}

// ----------------------------------------------------------------------------
// Dart arena (two consecutive darts per undirected edge)
// ----------------------------------------------------------------------------

const DELETED: u32 = u32::MAX;

/// A completed sub-triangulation with its two outer hull darts.
struct Piece {
    org: Vec<u32>,
    next: Vec<u32>,
    prev: Vec<u32>,
    le: u32, // outer dart out of the leftmost site
    re: u32, // outer dart into the rightmost site
}

/// Owns topology arrays, borrows the shared read-only coordinate table.
struct Arena<'c, P: Pred> {
    coord: &'c [Point],
    org: Vec<u32>,
    next: Vec<u32>,
    prev: Vec<u32>,
    _p: PhantomData<P>,
}

impl<'c, P: Pred> Arena<'c, P> {
    fn with_capacity(coord: &'c [Point], cap: usize) -> Self {
        Arena {
            coord,
            org: Vec::with_capacity(cap),
            next: Vec::with_capacity(cap),
            prev: Vec::with_capacity(cap),
            _p: PhantomData,
        }
    }

    /// Build over caller-owned buffers (reused across calls). Buffers are
    /// cleared and grown to `cap`.
    fn from_buffers(
        coord: &'c [Point],
        mut org: Vec<u32>,
        mut next: Vec<u32>,
        mut prev: Vec<u32>,
        cap: usize,
    ) -> Self {
        org.clear();
        next.clear();
        prev.clear();
        org.reserve(cap);
        next.reserve(cap);
        prev.reserve(cap);
        Arena {
            coord,
            org,
            next,
            prev,
            _p: PhantomData,
        }
    }

    fn into_buffers(self) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        (self.org, self.next, self.prev)
    }

    #[inline(always)]
    fn len(&self) -> u32 {
        self.org.len() as u32
    }

    /// Append another finished piece; dart links are shifted by the current base.
    fn append_piece(&mut self, p: Piece) {
        let base = self.org.len() as u32;
        self.org.extend_from_slice(&p.org);
        self.next.extend(p.next.iter().map(|&x| x + base));
        self.prev.extend(p.prev.iter().map(|&x| x + base));
    }

    fn into_piece(self, le: u32, re: u32) -> Piece {
        Piece {
            org: self.org,
            next: self.next,
            prev: self.prev,
            le,
            re,
        }
    }

    // --- dart navigation ---
    //
    // SAFETY for all `get_unchecked` calls: every dart index came from
    // `make_edge` (which pushes `e` and `e^1`), so it is always < arena length;
    // every origin value is a site id < `coord.len()`. The construction
    // preserves both invariants, so the indexing is in-bounds by construction.
    #[inline(always)]
    fn sym(e: u32) -> u32 {
        e ^ 1
    }
    #[inline(always)]
    fn org(&self, e: u32) -> u32 {
        unsafe { *self.org.get_unchecked(e as usize) }
    }
    #[inline(always)]
    fn dest(&self, e: u32) -> u32 {
        unsafe { *self.org.get_unchecked((e ^ 1) as usize) }
    }
    #[inline(always)]
    fn onext(&self, e: u32) -> u32 {
        unsafe { *self.next.get_unchecked(e as usize) }
    }
    #[inline(always)]
    fn oprev(&self, e: u32) -> u32 {
        unsafe { *self.prev.get_unchecked(e as usize) }
    }
    #[inline(always)]
    fn lnext(&self, e: u32) -> u32 {
        unsafe { *self.prev.get_unchecked((e ^ 1) as usize) }
    }
    #[inline(always)]
    fn rprev(&self, e: u32) -> u32 {
        unsafe { *self.next.get_unchecked((e ^ 1) as usize) }
    }

    // --- topology mutation ---
    fn make_edge(&mut self, a: u32, b: u32) -> u32 {
        let e = self.org.len() as u32;
        self.org.push(a);
        self.org.push(b);
        self.next.push(e);
        self.next.push(e + 1);
        self.prev.push(e);
        self.prev.push(e + 1);
        e
    }

    fn splice(&mut self, a: u32, b: u32) {
        let (a, b) = (a as usize, b as usize);
        unsafe {
            let an = *self.next.get_unchecked(a);
            let bn = *self.next.get_unchecked(b);
            *self.next.get_unchecked_mut(a) = bn;
            *self.prev.get_unchecked_mut(bn as usize) = a as u32;
            *self.next.get_unchecked_mut(b) = an;
            *self.prev.get_unchecked_mut(an as usize) = b as u32;
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
        unsafe {
            *self.org.get_unchecked_mut(e as usize) = DELETED;
            *self.org.get_unchecked_mut(se as usize) = DELETED;
        }
    }

    // --- predicates on site ids ---
    #[inline(always)]
    fn coord(&self, s: u32) -> Point {
        unsafe { *self.coord.get_unchecked(s as usize) }
    }
    #[inline(always)]
    fn orient3(&self, a: u32, b: u32, c: u32) -> i32 {
        P::orient(self.coord(a), self.coord(b), self.coord(c))
    }
    #[inline(always)]
    fn in_circle4(&self, a: u32, b: u32, c: u32, d: u32) -> bool {
        P::in_circle(self.coord(a), self.coord(b), self.coord(c), self.coord(d)) > 0
    }
    #[inline(always)]
    fn left_of(&self, x: u32, e: u32) -> bool {
        self.orient3(x, self.org(e), self.dest(e)) > 0
    }
    #[inline(always)]
    fn right_of(&self, x: u32, e: u32) -> bool {
        self.orient3(x, self.dest(e), self.org(e)) > 0
    }
    #[inline(always)]
    fn valid(&self, e: u32, basel: u32) -> bool {
        self.right_of(self.dest(e), basel)
    }

    // --- divide & conquer over the site-id range [lo, hi) ---
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
        // Lower common tangent.
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

        // Rising bubble.
        loop {
            let mut lcand = self.onext(Self::sym(basel));
            if self.valid(lcand, basel) {
                while self.in_circle4(
                    self.dest(basel),
                    self.org(basel),
                    self.dest(lcand),
                    self.dest(self.onext(lcand)),
                ) {
                    let t = self.onext(lcand);
                    self.delete_edge(lcand);
                    lcand = t;
                }
            }
            let mut rcand = self.oprev(basel);
            if self.valid(rcand, basel) {
                while self.in_circle4(
                    self.dest(basel),
                    self.org(basel),
                    self.dest(rcand),
                    self.dest(self.oprev(rcand)),
                ) {
                    let t = self.oprev(rcand);
                    self.delete_edge(rcand);
                    rcand = t;
                }
            }

            let l_valid = self.valid(lcand, basel);
            let r_valid = self.valid(rcand, basel);
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
                basel = self.connect(rcand, Self::sym(basel));
            } else {
                basel = self.connect(Self::sym(basel), Self::sym(lcand));
            }
        }
        (ldo, rdo)
    }

    // --- export: emit each interior (CCW) triangular face once ---
    fn export(&self, orig: &[u32], out: &mut Vec<Triangle>) {
        let dart_count = self.org.len() as u32;
        let mut e = 0u32;
        while e < dart_count {
            if self.org(e) != DELETED {
                let e1 = self.lnext(e);
                let e2 = self.lnext(e1);
                if self.lnext(e2) == e && e <= e1 && e <= e2 {
                    let a = self.org(e);
                    let b = self.org(e1);
                    let c = self.org(e2);
                    if self.orient3(a, b, c) > 0 {
                        out.push(Triangle {
                            i0: orig[a as usize],
                            i1: orig[b as usize],
                            i2: orig[c as usize],
                        });
                    }
                }
            }
            e += 1;
        }
    }
}

// ----------------------------------------------------------------------------
// Triangulator (public API)
// ----------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Site {
    x: i32,
    y: i32,
    original: u32,
}

/// Integer divide-and-conquer triangulator with reusable working storage.
pub struct Triangulator {
    threads: usize, // 1 = serial, 0 = auto, N = fixed
    coord: Vec<Point>,
    orig: Vec<u32>,
    sites: Vec<Site>,
    scratch: Vec<Site>,
    out: Vec<Triangle>,
    // Reused arena buffers (serial path, and the accumulator on the parallel path).
    abuf_org: Vec<u32>,
    abuf_next: Vec<u32>,
    abuf_prev: Vec<u32>,
}

impl Default for Triangulator {
    fn default() -> Self {
        Self::with_threads(1)
    }
}

impl Triangulator {
    pub fn new() -> Self {
        Self::with_threads(1)
    }

    /// `threads`: 1 = serial, 0 = auto (hardware concurrency), N = fixed.
    pub fn with_threads(threads: usize) -> Self {
        Triangulator {
            threads,
            coord: Vec::new(),
            orig: Vec::new(),
            sites: Vec::new(),
            scratch: Vec::new(),
            out: Vec::new(),
            abuf_org: Vec::new(),
            abuf_next: Vec::new(),
            abuf_prev: Vec::new(),
        }
    }

    pub fn predicate_width_for_spans(sx: i64, sy: i64) -> PredicateWidth {
        predicate_width(sx, sy)
    }

    fn resolved_threads(&self, m: usize) -> usize {
        if m < PARALLEL_MIN_POINTS {
            return 1;
        }
        match self.threads {
            0 => std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            n => n,
        }
    }

    pub fn triangulate(&mut self, points: &[Point]) -> Vec<Triangle> {
        let (min_x, min_y, max_x, max_y) = match self.prepare_sites(points) {
            Some(bbox) => bbox,
            None => return Vec::new(),
        };
        let m = self.coord.len();
        if m < 3 {
            return Vec::new();
        }
        let sx = max_x as i64 - min_x as i64;
        let sy = max_y as i64 - min_y as i64;
        let threads = self.resolved_threads(m);

        // Take the shared coord/orig/out out of self to sidestep borrow issues,
        // then put them back for reuse across calls.
        let coord = std::mem::take(&mut self.coord);
        let orig = std::mem::take(&mut self.orig);
        let mut out = std::mem::take(&mut self.out);
        out.clear();
        let mut buffers = Buffers {
            org: std::mem::take(&mut self.abuf_org),
            next: std::mem::take(&mut self.abuf_next),
            prev: std::mem::take(&mut self.abuf_prev),
        };

        match predicate_width(sx, sy) {
            PredicateWidth::Int64 => {
                run::<PredFast>(&coord, &orig, threads, &mut out, &mut buffers)
            }
            PredicateWidth::Int128 => {
                run::<PredWide>(&coord, &orig, threads, &mut out, &mut buffers)
            }
            PredicateWidth::Unsupported => {
                panic!("coordinate span exceeds MAX_COORDINATE_SPAN")
            }
        }

        self.coord = coord;
        self.orig = orig;
        self.abuf_org = buffers.org;
        self.abuf_next = buffers.next;
        self.abuf_prev = buffers.prev;
        let result = std::mem::take(&mut out);
        self.out = out; // keep the (now-empty) allocation for reuse
        result
    }

    /// Radix-sort sites by (x, y) and deduplicate, keeping the lowest original
    /// index per coordinate. Returns the bounding box, or None if < 1 unique.
    fn prepare_sites(&mut self, points: &[Point]) -> Option<(i32, i32, i32, i32)> {
        self.coord.clear();
        self.orig.clear();
        if points.is_empty() {
            return None;
        }

        let (mut min_x, mut min_y, mut max_x, mut max_y) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for p in points {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }

        self.sites.clear();
        self.sites.reserve(points.len());
        for (i, p) in points.iter().enumerate() {
            self.sites.push(Site {
                x: p.x,
                y: p.y,
                original: i as u32,
            });
        }

        // LSD radix sort by (dx, dy) with dy least significant -> ordered by x
        // then y. Stable + input-in-original-order => equal coords keep the
        // lowest original index first.
        radix_sort_sites(&mut self.sites, &mut self.scratch, min_x, min_y);

        let mut last: Option<(i32, i32)> = None;
        for s in &self.sites {
            if last != Some((s.x, s.y)) {
                self.coord.push(Point { x: s.x, y: s.y });
                self.orig.push(s.original);
                last = Some((s.x, s.y));
            }
        }
        Some((min_x, min_y, max_x, max_y))
    }
}

/// Stable LSD radix sort of sites by (x, y), rebased to unsigned.
fn radix_sort_sites(sites: &mut Vec<Site>, scratch: &mut Vec<Site>, min_x: i32, min_y: i32) {
    let n = sites.len();
    scratch.clear();
    scratch.resize(
        n,
        Site {
            x: 0,
            y: 0,
            original: 0,
        },
    );

    // key digits from least to most significant: dy.lo, dy.hi, dx.lo, dx.hi
    let digit = |s: &Site, pass: usize| -> usize {
        let dx = (s.x as i64 - min_x as i64) as u64;
        let dy = (s.y as i64 - min_y as i64) as u64;
        let v = match pass {
            0 => dy & 0xffff,
            1 => (dy >> 16) & 0xffff,
            2 => dx & 0xffff,
            _ => (dx >> 16) & 0xffff,
        };
        v as usize
    };

    let mut counts = vec![0u32; 0x1_0000 + 1];
    for pass in 0..4 {
        for c in counts.iter_mut() {
            *c = 0;
        }
        for s in sites.iter() {
            counts[digit(s, pass) + 1] += 1;
        }
        for i in 0..0x1_0000 {
            counts[i + 1] += counts[i];
        }
        for s in sites.iter() {
            let d = digit(s, pass);
            let pos = counts[d];
            counts[d] += 1;
            scratch[pos as usize] = *s;
        }
        std::mem::swap(sites, scratch);
    }
    // 4 passes (even) => result is back in `sites`.
}

/// Reusable arena buffers carried across triangulate() calls.
struct Buffers {
    org: Vec<u32>,
    next: Vec<u32>,
    prev: Vec<u32>,
}

fn run<P: Pred>(
    coord: &[Point],
    orig: &[u32],
    threads: usize,
    out: &mut Vec<Triangle>,
    buffers: &mut Buffers,
) {
    let m = coord.len();
    // Number of parallel chunks (1 => serial).
    let nchunks = if threads <= 1 {
        1
    } else {
        threads.min(m / 2000).max(1)
    };

    if nchunks == 1 {
        let mut arena = Arena::<P>::from_buffers(
            coord,
            std::mem::take(&mut buffers.org),
            std::mem::take(&mut buffers.next),
            std::mem::take(&mut buffers.prev),
            m.saturating_mul(10),
        );
        arena.delaunay(0, m as u32);
        arena.export(orig, out);
        let (o, n, p) = arena.into_buffers();
        buffers.org = o;
        buffers.next = n;
        buffers.prev = p;
        return;
    }

    // Parallel: split the sorted sites into contiguous x-ranges, one per chunk.
    let bounds: Vec<(u32, u32)> = (0..nchunks)
        .map(|i| {
            let lo = i * m / nchunks;
            let hi = (i + 1) * m / nchunks;
            (lo as u32, hi as u32)
        })
        .collect();

    let mut pieces: Vec<Piece> = std::thread::scope(|s| {
        let handles: Vec<_> = bounds
            .iter()
            .map(|&(lo, hi)| s.spawn(move || build_piece::<P>(coord, lo, hi)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Concatenate pieces into one (reused) accumulator arena, then zip L→R.
    let total: usize = pieces.iter().map(|p| p.org.len()).sum::<usize>() + m * 2;
    let mut acc = Arena::<P>::from_buffers(
        coord,
        std::mem::take(&mut buffers.org),
        std::mem::take(&mut buffers.next),
        std::mem::take(&mut buffers.prev),
        total,
    );
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
    acc.export(orig, out);
    let (o, n, p) = acc.into_buffers();
    buffers.org = o;
    buffers.next = n;
    buffers.prev = p;
}

fn build_piece<P: Pred>(coord: &[Point], lo: u32, hi: u32) -> Piece {
    let n = (hi - lo) as usize;
    let mut arena = Arena::<P>::with_capacity(coord, n.saturating_mul(10));
    let (le, re) = arena.delaunay(lo, hi);
    arena.into_piece(le, re)
}
