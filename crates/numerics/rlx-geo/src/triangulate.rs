// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serial Guibas-Stolfi divide-and-conquer Delaunay triangulation over signed
//! integer sites, using a compact two-dart primal edge ring (`sym(e) = e ^ 1`).
//! Exact integer predicates make the result robust and deterministic.

use crate::predicates::{
    MAX_COORDINATE_SPAN, Pred, PredFast, PredWide, PredicateWidth, predicate_width,
};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Error returned by the triangulation entry points.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GeoError {
    /// The input's coordinate span exceeds the range over which the exact
    /// integer predicates are certified ([`MAX_COORDINATE_SPAN`]). `span` is the
    /// larger of the x and y spans. Rescale the coordinates to recover.
    CoordinateSpanExceeded { span: i64, max: i64 },
}

impl std::fmt::Display for GeoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeoError::CoordinateSpanExceeded { span, max } => write!(
                f,
                "coordinate span {span} exceeds the exact-predicate limit {max}"
            ),
        }
    }
}

impl std::error::Error for GeoError {}

#[inline]
fn span_error(sx: i64, sy: i64) -> GeoError {
    GeoError::CoordinateSpanExceeded {
        span: sx.max(sy),
        max: MAX_COORDINATE_SPAN,
    }
}

const DELETED: u32 = u32::MAX;
/// High bit of a dart's `org`, set on the outer (unbounded) face's darts before
/// export so the scan skips them (and deleted darts, whose `org == u32::MAX`,
/// also have it set) with one test — no per-dart triangle check needed, since
/// every remaining bounded face is a triangle. Site ids are `< 2^31`, so it's free.
const VISITED: u32 = 0x8000_0000;

/// A raw pointer that is `Send` so it can cross a `thread::scope` boundary while
/// **preserving provenance** (unlike a `ptr as usize` round-trip, which relies on
/// exposed provenance). Soundness is the caller's: every thread that copies this
/// must only dereference disjoint, non-overlapping slots.
struct SendPtr<T>(*mut T);
// A raw pointer is `Copy` for any `T`; the derive would wrongly add `T: Copy`.
impl<T> Clone for SendPtr<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendPtr<T> {}
// SAFETY: callers guarantee each thread writes/reads a distinct address range.
unsafe impl<T> Send for SendPtr<T> {}
// SAFETY: same disjointness guarantee — sharing `&SendPtr` across pool workers is
// sound because each only dereferences its own non-overlapping index.
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline(always)]
    fn ptr(self) -> *mut T {
        self.0
    }
}

// ============================================================================
// Persistent worker pool
// ============================================================================

/// A `dyn Fn(usize)` fat pointer asserted `Send`/`Sync`. Sound because `for_each`
/// blocks until every worker has finished calling it, so the referent outlives
/// the pointer, and the referent is `Sync`.
#[derive(Clone, Copy)]
struct FnRef(*const (dyn Fn(usize) + Sync));
unsafe impl Send for FnRef {}
unsafe impl Sync for FnRef {}

struct PoolInner {
    // Job dispatch state (mutex-guarded); `next`/`active` are the hot path.
    state: std::sync::Mutex<PoolState>,
    work: std::sync::Condvar, // workers wait here for a new job
    done: std::sync::Condvar, // the submitter waits here for completion
    next: AtomicUsize,        // work-stealing cursor into 0..n
    active: AtomicUsize,      // workers still processing the current job
}
struct PoolState {
    generation: u64,
    n: usize,
    f: FnRef,
    shutdown: bool,
}

/// A persistent work-stealing pool: `threads` workers spawned **once**, reused
/// across every phase of a triangulation via [`Pool::for_each`]. This replaces
/// the ~11 `thread::scope`s (≈168 fresh OS-thread spawns, ~1.8 ms) that a single
/// call would otherwise pay — the big fixed cost at small/medium n.
struct Pool {
    inner: std::sync::Arc<PoolInner>,
    workers: Vec<std::thread::JoinHandle<()>>,
    n_workers: usize,
}

impl Pool {
    fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let inner = std::sync::Arc::new(PoolInner {
            state: std::sync::Mutex::new(PoolState {
                generation: 0,
                n: 0,
                f: FnRef(std::ptr::null::<fn(usize)>() as *const (dyn Fn(usize) + Sync)),
                shutdown: false,
            }),
            work: std::sync::Condvar::new(),
            done: std::sync::Condvar::new(),
            next: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
        });
        let workers = (0..threads)
            .map(|_| {
                let inner = std::sync::Arc::clone(&inner);
                std::thread::spawn(move || Self::worker(inner))
            })
            .collect();
        Pool {
            inner,
            workers,
            n_workers: threads,
        }
    }

    fn worker(inner: std::sync::Arc<PoolInner>) {
        let mut seen = 0u64;
        loop {
            let (n, f) = {
                let mut st = inner.state.lock().unwrap();
                while st.generation == seen && !st.shutdown {
                    st = inner.work.wait(st).unwrap();
                }
                if st.shutdown {
                    return;
                }
                seen = st.generation;
                (st.n, st.f)
            };
            loop {
                let i = inner.next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                // SAFETY: `for_each` keeps `f`'s referent alive and `Sync` until
                // `active` reaches 0, which only happens after this call returns.
                unsafe { (*f.0)(i) };
            }
            if inner.active.fetch_sub(1, Ordering::AcqRel) == 1 {
                let _guard = inner.state.lock().unwrap();
                inner.done.notify_one();
            }
        }
    }

    /// Run `f(0), …, f(n-1)` across the workers (work-stealing) and block until
    /// all complete. `f` may borrow the caller's stack, like `thread::scope`.
    fn for_each(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        if n == 0 {
            return;
        }
        // Erase `f`'s lifetime: `FnRef` holds a `'static`-pointee raw pointer, but
        // `f` only lives for this call. Sound because we block below until every
        // worker has stopped dereferencing it (`active` reaches 0).
        let fp: *const (dyn Fn(usize) + Sync + '_) = f;
        let fr = FnRef(unsafe {
            std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), *const (dyn Fn(usize) + Sync)>(
                fp,
            )
        });
        {
            let mut st = self.inner.state.lock().unwrap();
            st.n = n;
            st.f = fr;
            st.generation += 1;
            self.inner.next.store(0, Ordering::Relaxed);
            self.inner.active.store(self.n_workers, Ordering::Release);
        }
        self.inner.work.notify_all();
        let mut st = self.inner.state.lock().unwrap();
        while self.inner.active.load(Ordering::Acquire) != 0 {
            st = self.inner.done.wait(st).unwrap();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut st = self.inner.state.lock().unwrap();
            st.shutdown = true;
        }
        self.inner.work.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

thread_local! {
    /// One persistent pool per calling thread, created on first use and reused
    /// across every `triangulate_par` call from that thread — the exact analogue
    /// of C++ `delaunay32`'s persistent `WorkerTeam`/`Triangulator`. Idle workers
    /// park on a condvar (no CPU), so this only spawns OS threads once instead of
    /// ~14 per call. Sized to `available_parallelism`; the requested `threads`
    /// only tunes chunk count, and a pool with more workers than chunks is fine
    /// (surplus workers see `i >= n` and return immediately). Thread-local, so
    /// concurrent callers never share a pool — no cross-triangulation races.
    static POOL: std::cell::OnceCell<std::rc::Rc<Pool>> = const { std::cell::OnceCell::new() };
}

/// The calling thread's persistent [`Pool`], lazily spawned on first use.
fn thread_pool() -> std::rc::Rc<Pool> {
    // Ablation: `GEO_ABL_NOPOOL=1` spawns a fresh pool per call (the pre-persistent
    // behaviour) so the persistent pool's contribution can be measured.
    if std::env::var_os("GEO_ABL_NOPOOL").is_some() {
        let n = std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1);
        return std::rc::Rc::new(Pool::new(n));
    }
    POOL.with(|c| {
        c.get_or_init(|| {
            let n = std::thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(1);
            std::rc::Rc::new(Pool::new(n))
        })
        .clone()
    })
}

/// Ablation: `GEO_ABL_NODWYER=1` routes the wide (i128) span back to the old plain
/// x-cut Guibas-Stolfi build instead of the 64-bit-Morton/Dwyer fast build.
#[inline]
fn abl_no_wide_dwyer() -> bool {
    std::env::var_os("GEO_ABL_NODWYER").is_some()
}

/// Reusable per-thread backing stores, handed back after each triangulation
/// (capacity retained, contents logically dropped). These are the large
/// allocations — at 2M the concat arena is ~144 MB, and `coord`/`orig` another
/// ~32 MB — and macOS munmaps them on free, so fresh `Vec`s re-fault tens of MB
/// of zeroed pages every call (measured ~50–105 MB). Reusing them turns that into
/// warm writes, mirroring C++ `delaunay32` reusing its `Triangulator`'s buffers.
#[derive(Default)]
struct Scratch {
    darts: Vec<[u32; 3]>,
    coord: Vec<[i32; 3]>,
    orig: Vec<u32>,
    /// Free-list of per-chunk build buffers (one becomes each `Piece`'s dart
    /// arena). Popped on the calling thread and distributed to pool workers by
    /// index, reclaimed after the concat copies them. Capped so a run of tiny
    /// triangulations can't pin unbounded memory.
    piece_bufs: Vec<Vec<[u32; 3]>>,
}

/// Cap on pooled per-piece buffers (chunk count is `threads*mul`, ≤ ~56 by
/// default; 96 leaves headroom without hoarding on size regressions).
const PIECE_POOL_CAP: usize = 96;

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = const {
        std::cell::RefCell::new(Scratch {
            darts: Vec::new(),
            coord: Vec::new(),
            orig: Vec::new(),
            piece_bufs: Vec::new(),
        })
    };
}

/// `GEO_NO_REUSE=1` disables the scratch pool (fresh allocation every call) — used
/// only to A/B the reuse win in one process, immune to background-load drift.
fn reuse_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GEO_NO_REUSE").is_none())
}

/// Keep whichever buffer has the larger capacity (never shrink the pooled one on
/// a small triangulation sandwiched between two large ones).
#[inline]
fn keep_larger<T>(slot: &mut Vec<T>, v: Vec<T>) {
    if reuse_enabled() && v.capacity() >= slot.capacity() {
        *slot = v;
    }
}

#[inline]
fn take_darts_scratch() -> Vec<[u32; 3]> {
    SCRATCH.with(|s| {
        let mut v = std::mem::take(&mut s.borrow_mut().darts);
        v.clear();
        v
    })
}
#[inline]
fn put_darts_scratch(v: Vec<[u32; 3]>) {
    SCRATCH.with(|s| keep_larger(&mut s.borrow_mut().darts, v));
}
#[inline]
fn take_coord_scratch() -> Vec<[i32; 3]> {
    SCRATCH.with(|s| {
        let mut v = std::mem::take(&mut s.borrow_mut().coord);
        v.clear();
        v
    })
}
#[inline]
fn put_coord_scratch(v: Vec<[i32; 3]>) {
    SCRATCH.with(|s| keep_larger(&mut s.borrow_mut().coord, v));
}
#[inline]
fn take_orig_scratch() -> Vec<u32> {
    SCRATCH.with(|s| {
        let mut v = std::mem::take(&mut s.borrow_mut().orig);
        v.clear();
        v
    })
}
#[inline]
fn put_orig_scratch(v: Vec<u32>) {
    SCRATCH.with(|s| keep_larger(&mut s.borrow_mut().orig, v));
}

/// Take `n` recycled per-piece build buffers (each emptied), padding with fresh
/// empty `Vec`s when the pool holds fewer. Called once on the calling thread
/// before the parallel build, then handed to workers by index.
fn take_piece_bufs(n: usize) -> Vec<Vec<[u32; 3]>> {
    SCRATCH.with(|s| {
        let pool = &mut s.borrow_mut().piece_bufs;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            match pool.pop() {
                Some(mut v) => {
                    v.clear();
                    out.push(v);
                }
                None => out.push(Vec::new()),
            }
        }
        out
    })
}

/// Return a per-piece buffer to the pool (capped) for the next call to reuse.
#[inline]
fn put_piece_buf(v: Vec<[u32; 3]>) {
    if !reuse_enabled() || v.capacity() == 0 {
        return;
    }
    SCRATCH.with(|s| {
        let pool = &mut s.borrow_mut().piece_bufs;
        if pool.len() < PIECE_POOL_CAP {
            pool.push(v);
        }
    });
}

/// Release this thread's reusable scratch buffers (the dart arena, `coord`/`orig`
/// scatter buffers, and per-piece pool) back to the allocator. The buffers exist
/// only to avoid re-allocating/re-faulting across repeated triangulations on the
/// same thread — worth ~8–11% on Linux — but at large `n` they retain hundreds of
/// MB. Call this after a big one-off triangulation to reclaim that memory; the
/// next call simply re-allocates. Only affects the calling thread.
pub fn release_scratch() {
    SCRATCH.with(|s| {
        let mut sc = s.borrow_mut();
        *sc = Scratch::default();
    });
}

thread_local! {
    /// Radix ping-pong scratch for `morton_positions`, one pair per thread. Since
    /// the pool workers are persistent, each reuses its own pair across every chunk
    /// and every call — the last per-chunk allocation left after the arena/piece
    /// pools. Self-contained (taken and returned inside `morton_positions`).
    static MORTON_SCRATCH: std::cell::RefCell<(Vec<u32>, Vec<u64>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Take the radix scratch pair sized to `n` (contents overwritten by the sort).
#[inline]
fn take_morton_scratch(n: usize) -> (Vec<u32>, Vec<u64>) {
    MORTON_SCRATCH.with(|s| {
        let (mut p, mut k) = std::mem::take(&mut *s.borrow_mut());
        p.clear();
        p.reserve(n);
        k.clear();
        k.reserve(n);
        // Fully rewritten each radix pass before being read (the scatter is a
        // permutation onto [0, n)); uninit `set_len` avoids a zero-fill.
        unsafe {
            p.set_len(n);
            k.set_len(n);
        }
        (p, k)
    })
}

/// Return the radix scratch pair for the next `morton_positions` to reuse.
#[inline]
fn put_morton_scratch(p: Vec<u32>, k: Vec<u64>) {
    if !reuse_enabled() {
        return;
    }
    MORTON_SCRATCH.with(|s| {
        let mut slot = s.borrow_mut();
        if p.capacity() >= slot.0.capacity() {
            slot.0 = p;
        }
        if k.capacity() >= slot.1.capacity() {
            slot.1 = k;
        }
    });
}

/// Prepared sites: unique coordinates, their original indices, and the x/y spans.
type Prepared = (Vec<[i32; 3]>, Vec<u32>, i64, i64);

/// Triangulate `points`, returning counterclockwise triangles as indices into
/// the original `points` slice. Coincident points collapse (the lowest original
/// index is kept); collinear input yields no triangles.
///
/// # Errors
/// [`GeoError::CoordinateSpanExceeded`] if the coordinate span exceeds
/// [`MAX_COORDINATE_SPAN`] (rescale the coordinates to recover).
pub fn triangulate(points: &[[i32; 2]]) -> Result<Vec<[u32; 3]>, GeoError> {
    let prof = std::env::var_os("GEO_PROF").is_some();
    let t0 = std::time::Instant::now();
    let Some((coord, orig, sx, sy)) = prepare(points) else {
        return Ok(Vec::new());
    };
    if prof {
        eprintln!(
            "  prepare(sort+dedup): {:.2} ms",
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    let mut out = Vec::new();
    match predicate_width(sx, sy) {
        // Both widths use the Morton/Dwyer build; only the predicate arithmetic
        // differs (i64 lifted vs i128). The wide path used to fall back to the
        // plain x-cut Guibas-Stolfi `run`, which was ~4× slower — the 64-bit
        // Morton makes the fast build valid for the full coordinate range.
        PredicateWidth::Int64 => run_dwyer::<PredFast>(&coord, &orig, &mut out),
        PredicateWidth::Int128 if abl_no_wide_dwyer() => run::<PredWide>(&coord, &orig, &mut out),
        PredicateWidth::Int128 => run_dwyer::<PredWide>(&coord, &orig, &mut out),
        PredicateWidth::Unsupported => return Err(span_error(sx, sy)),
    }
    Ok(out)
}

/// Parallel triangulation: split sorted sites into contiguous x-ranges, build
/// each on its own thread (`std::thread::scope`), then zip the pieces
/// left-to-right with the same Guibas-Stolfi merge. `threads`: 0 = auto.
/// Below [`PARALLEL_MIN`] sites it runs serially (thread overhead isn't worth it).
///
/// # Errors
/// [`GeoError::CoordinateSpanExceeded`] if the coordinate span exceeds
/// [`MAX_COORDINATE_SPAN`].
pub fn triangulate_par(points: &[[i32; 2]], threads: usize) -> Result<Vec<[u32; 3]>, GeoError> {
    if points.len() < 3 {
        return Ok(Vec::new());
    }
    let t = if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    };
    let mut out = Vec::new();

    // Spans straight from the raw points (cheap O(n) pass) — needed to pick the
    // predicate width *before* preparing, so the fast path can skip the dedup sort.
    let (mut mnx, mut mny, mut mxx, mut mxy) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for p in points {
        mnx = mnx.min(p[0]);
        mny = mny.min(p[1]);
        mxx = mxx.max(p[0]);
        mxy = mxy.max(p[1]);
    }
    let (sx, sy) = (mxx as i64 - mnx as i64, mxy as i64 - mny as i64);

    // Fast + parallel: partition without a full sort and dedup per chunk during
    // the build's Morton sort (eliminates prepare's redundant second sort). The
    // 64-bit Morton makes this valid for both predicate widths, so the wide range
    // gets the fast path too (i128 predicates over the same Morton/Dwyer build).
    if t > 1 && points.len() >= parallel_min() {
        let done = match predicate_width(sx, sy) {
            PredicateWidth::Int64 => run_par_fast::<PredFast>(points, t, mnx, mny, mxx, &mut out),
            // Ablation: skip the wide fast path so it falls through to the robust
            // (plain x-cut Guibas-Stolfi) `run_par` below.
            PredicateWidth::Int128 if abl_no_wide_dwyer() => false,
            PredicateWidth::Int128 => run_par_fast::<PredWide>(points, t, mnx, mny, mxx, &mut out),
            PredicateWidth::Unsupported => return Err(span_error(sx, sy)),
        };
        if done {
            return Ok(out);
        }
    }
    out.clear(); // fast path bailed on degenerate input — use the robust path

    // Robust path (also the wide/i128 path): full sort + global dedup, then build.
    let tp = std::time::Instant::now();
    let Some((coord, orig, sx, sy)) = prepare_par(points, t) else {
        return Ok(Vec::new());
    };
    if std::env::var_os("GEO_PROF").is_some() {
        eprintln!(
            "  par prepare (sort):     {:.2} ms",
            tp.elapsed().as_secs_f64() * 1e3
        );
    }
    match predicate_width(sx, sy) {
        PredicateWidth::Int64 => run_par::<PredFast>(&coord, &orig, t, &mut out),
        PredicateWidth::Int128 => run_par::<PredWide>(&coord, &orig, t, &mut out),
        PredicateWidth::Unsupported => return Err(span_error(sx, sy)),
    }
    Ok(out)
}

/// Parallel `prepare`: bucket-sort sites by x (buckets are x-ordered, so no
/// merge), sort each contiguous bucket-group in parallel, then dedup.
fn prepare_par(points: &[[i32; 2]], threads: usize) -> Option<Prepared> {
    let n = points.len();
    if n < 3 {
        return None;
    }
    if threads <= 1 || n < parallel_min() {
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
    let prof = std::env::var_os("GEO_PROF").is_some();
    let t = std::time::Instant::now();
    bucket_sort_by_x(&mut sites, threads, mnx, mxx);
    if prof {
        eprintln!(
            "  [prep] bucket_sort:   {:.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let t = std::time::Instant::now();
    let r = finish_sites(&sites);
    if prof {
        eprintln!(
            "  [prep] finish/dedup:  {:.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    r
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
    let base = SendPtr(out.as_mut_ptr());
    std::thread::scope(|s| {
        for (c, chunk) in sites.chunks(per).enumerate() {
            let mut o = off[c].clone();
            s.spawn(move || {
                let ptr = base.ptr(); // captures `base` whole (Copy) → Send
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

/// Sites below this count use the serial path. Measured crossover: the parallel
/// path (persistent pool → no per-call spawn cost) overtakes the serial Dwyer at
/// ~6.8k sites; 8k is the robust threshold (clear win with margin, and safe under
/// background load, which penalises small-n parallelism most). Parallel is up to
/// ~3× faster than serial across the old [8k, 50k] gap this closes.
pub const PARALLEL_MIN: usize = 8_000;

/// Effective serial→parallel crossover. Defaults to [`PARALLEL_MIN`]; overridable
/// via `GEO_PARALLEL_MIN` for crossover tuning (cached — read once).
pub(crate) fn parallel_min() -> usize {
    use std::sync::OnceLock;
    static PM: OnceLock<usize> = OnceLock::new();
    *PM.get_or_init(|| {
        std::env::var("GEO_PARALLEL_MIN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(PARALLEL_MIN)
    })
}

/// Minimum sites per chunk — the floor on chunk count (`nchunks ≤ n / chunk_min`).
/// Smaller ⇒ more chunks ⇒ more cores used at small n, at the cost of more trunk
/// merges. Overridable via `GEO_CHUNK_MIN` (cached).
pub(crate) fn chunk_min() -> usize {
    use std::sync::OnceLock;
    static CM: OnceLock<usize> = OnceLock::new();
    *CM.get_or_init(|| {
        std::env::var("GEO_CHUNK_MIN")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&x| x >= 1)
            .unwrap_or(2000)
    })
}

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
    let mut coord: Vec<[i32; 3]> = Vec::with_capacity(sites.len());
    let mut orig: Vec<u32> = Vec::with_capacity(sites.len());
    let mut last: Option<(i32, i32)> = None;
    for &(x, y, o) in sites {
        if last != Some((x, y)) {
            coord.push([x, y, 0]); // lift filled below for the fast span
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
    let (sx, sy) = (mxx as i64 - mnx as i64, mxy as i64 - mny as i64);
    pack_lift(&mut coord, sx.max(sy), mnx, mny);
    Some((coord, orig, sx, sy))
}

/// Pack the paraboloid lift `(x-mnx)² + (y-mny)²` into `coord[..][2]` when the
/// span fits the i64 fast path (it then fits i32); otherwise leave it `0` (the
/// wide `in_circle` doesn't read it). Same origin for every point so lift
/// *differences* are meaningful in the lifted determinant.
#[inline]
fn pack_lift(coord: &mut [[i32; 3]], span: i64, mnx: i32, mny: i32) {
    if span > crate::predicates::FAST_COORDINATE_SPAN {
        return;
    }
    for p in coord.iter_mut() {
        let dx = (p[0] - mnx) as i64;
        let dy = (p[1] - mny) as i64;
        p[2] = (dx * dx + dy * dy) as i32;
    }
}

fn run<P: Pred>(coord: &[[i32; 3]], orig: &[u32], out: &mut Vec<[u32; 3]>) {
    let prof = std::env::var_os("GEO_PROF").is_some();
    let m = coord.len();
    let mut arena = Arena::<P>::with_capacity(coord, m.saturating_mul(8));
    let t = std::time::Instant::now();
    let (le, _re) = arena.delaunay(0, m as u32);
    if prof {
        eprintln!(
            "  build(delaunay):     {:.2} ms  ({} darts)",
            t.elapsed().as_secs_f64() * 1e3,
            arena.darts.len()
        );
    }
    let t = std::time::Instant::now();
    arena.export(le, orig, out);
    if prof {
        eprintln!(
            "  export:              {:.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
}

fn run_par<P: Pred>(coord: &[[i32; 3]], orig: &[u32], threads: usize, out: &mut Vec<[u32; 3]>) {
    let m = coord.len();
    if threads <= 1 || m < parallel_min() {
        return run::<P>(coord, orig, out);
    }
    // Oversubscribe: cut more chunks than threads so a work-stealing queue keeps
    // fast P-cores busy while slow E-cores take fewer (M-series is heterogeneous).
    // `GEO_CHUNK_MUL` overrides the multiplier for tuning.
    let mul = std::env::var("GEO_CHUNK_MUL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&x| x >= 1)
        .unwrap_or(4);
    let nchunks = (threads * mul).min(m / chunk_min()).max(1);
    if nchunks == 1 {
        return run::<P>(coord, orig, out);
    }
    let bounds: Vec<(u32, u32)> = (0..nchunks)
        .map(|i| ((i * m / nchunks) as u32, ((i + 1) * m / nchunks) as u32))
        .collect();
    let prof = std::env::var_os("GEO_PROF").is_some();
    let tb = std::time::Instant::now();
    let pool = thread_pool();
    // Work-stealing over oversubscribed chunks: the pool hands out chunk indices
    // and each build result lands directly in its own slot (x order preserved).
    let mut slots: Vec<Option<Piece>> = (0..nchunks).map(|_| None).collect();
    {
        let sp = SendPtr(slots.as_mut_ptr());
        let bref = &bounds;
        pool.for_each(bounds.len(), &|i| {
            let (lo, hi) = bref[i];
            let piece = build_piece::<P>(coord, lo, hi);
            unsafe { *sp.ptr().add(i) = Some(piece) };
        });
    }
    let pieces: Vec<Piece> = slots.into_iter().map(|p| p.unwrap()).collect();
    if prof {
        eprintln!(
            "  par build ({nchunks} chunks): {:.2} ms",
            tb.elapsed().as_secs_f64() * 1e3
        );
    }
    assemble_and_export::<P>(&pool, coord, orig, pieces, m, threads, out);
}

/// Stitch built pieces into one triangulation and export the faces: parallel
/// shared-arena concat (disjoint windows) → parallel tree trunk merge → parallel
/// export. Pieces must be in left-to-right x order with darts referencing global
/// `coord`/`orig` indices. Shared by both the deduped-input and raw-input paths.
fn assemble_and_export<P: Pred>(
    pool: &Pool,
    coord: &[[i32; 3]],
    orig: &[u32],
    mut pieces: Vec<Piece>,
    m: usize,
    threads: usize,
    out: &mut Vec<[u32; 3]>,
) {
    let prof = std::env::var_os("GEO_PROF").is_some();
    // Concat: assign each piece a disjoint window in one buffer and scatter-copy
    // in parallel (the raw-pointer/disjoint-range pattern from `bucket_sort_by_x`).
    let tm = std::time::Instant::now();
    let np = pieces.len();
    let mut base = vec![0u32; np + 1]; // window starts (dart units), prefix sum
    for (i, p) in pieces.iter().enumerate() {
        base[i + 1] = base[i] + p.darts.len() as u32;
    }
    let total = base[np] as usize;
    // Headroom for seam edges the trunk merge bump-allocates (tiny in practice;
    // overflow would panic, not corrupt — see MergeCtx).
    let cap = total + m * 2;
    // Reuse the thread-local arena (see `Scratch`): a warm buffer instead of a
    // fresh ~144 MB (2M) allocation that Linux would re-fault from zeroed pages.
    let mut darts = take_darts_scratch();
    darts.reserve(cap);
    let dst = SendPtr(darts.as_mut_ptr());
    {
        let (pcs, bs) = (&pieces, &base);
        pool.for_each(np, &|i| {
            let b = bs[i];
            let ptr = dst.ptr();
            // org (0) is a global site id; next/prev (1,2) shift by the base.
            for (k, &[o, nx, pv]) in pcs[i].darts.iter().enumerate() {
                unsafe { ptr.add(b as usize + k).write([o, nx + b, pv + b]) };
            }
        });
    }
    // Every slot in [0, total) was written exactly once (windows tile the range).
    unsafe { darts.set_len(total) };
    if prof {
        eprintln!(
            "  par concat ({np} pieces):  {:.2} ms",
            tm.elapsed().as_secs_f64() * 1e3
        );
    }
    let tmm = std::time::Instant::now();
    let hulls: Vec<(u32, u32)> = (0..np)
        .map(|i| (pieces[i].le + base[i], pieces[i].re + base[i]))
        .collect();
    // The piece darts are now copied into the shared arena; recycle their buffers
    // (the ~144 MB of per-chunk allocations at 2M) for the next call to reuse.
    for p in pieces.iter_mut() {
        put_piece_buf(std::mem::take(&mut p.darts));
    }
    let le = merge_tree_par::<P>(pool, coord, &mut darts, hulls, total as u32, cap as u32);
    if prof {
        eprintln!(
            "  par trunk merge:        {:.2} ms  (darts {})",
            tmm.elapsed().as_secs_f64() * 1e3,
            darts.len()
        );
    }
    let mut acc = Arena::<P> {
        coord,
        darts,
        free: Vec::new(),
        _p: PhantomData,
    };
    let te = std::time::Instant::now();
    acc.export_par(pool, le, orig, threads, out);
    if prof {
        eprintln!(
            "  par export:             {:.2} ms",
            te.elapsed().as_secs_f64() * 1e3
        );
    }
    // Hand the arena back for the next triangulation on this thread to reuse.
    put_darts_scratch(acc.darts);
}

/// Fast parallel path: partition points into x-separable whole-bucket chunks
/// WITHOUT a within-bucket sort or a global dedup, then dedup per chunk during
/// the Morton sort the build needs anyway (identical points share a Morton key).
/// This removes the redundant second sort — the old path sorted by (x,y) for
/// dedup and the build then re-sorted by Morton. Generic over the predicate width
/// `P`: the Morton build structure is the same whether predicates are i64
/// (`PredFast`, span ≤ 29609) or i128 (`PredWide`, larger spans). Returns `false`
/// on pathological input (a chunk collapsing to <2 distinct points) so the caller
/// can fall back to the robust dedup-first path.
fn run_par_fast<P: Pred>(
    points: &[[i32; 2]],
    threads: usize,
    mnx: i32,
    mny: i32,
    mxx: i32,
    out: &mut Vec<[u32; 3]>,
) -> bool {
    let n = points.len();
    let mul = std::env::var("GEO_CHUNK_MUL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&x| x >= 1)
        .unwrap_or(4);
    let nchunks = (threads * mul).min(n / chunk_min()).max(1);
    if nchunks <= 1 {
        return false;
    }
    let prof = std::env::var_os("GEO_PROF").is_some();
    // Persistent per-thread pool: spawned once, reused across every call and phase.
    let pool = thread_pool();
    let tp = std::time::Instant::now();
    let (coord_raw, orig_raw, bounds) =
        bucket_partition_by_x(&pool, points, threads, nchunks, mnx, mny, mxx);
    if prof {
        eprintln!(
            "  par partition (no sort): {:.2} ms  ({} chunks)",
            tp.elapsed().as_secs_f64() * 1e3,
            bounds.len()
        );
    }
    let tb = std::time::Instant::now();
    // Build every chunk via the pool; results land in disjoint slots by index.
    // Each chunk builds into a recycled buffer pulled from the per-thread pool
    // (distributed here, on the calling thread — workers can't touch it directly).
    let nb = bounds.len();
    let mut slots: Vec<Option<Piece>> = (0..nb).map(|_| None).collect();
    let mut bufs = take_piece_bufs(nb);
    {
        let sp = SendPtr(slots.as_mut_ptr());
        let bp = SendPtr(bufs.as_mut_ptr());
        let (cr, or, bnds) = (&coord_raw, &orig_raw, &bounds);
        pool.for_each(nb, &|i| {
            let (lo, hi) = bnds[i];
            // Move chunk i's buffer out of the shared vec (leaves an empty Vec at
            // that slot); the Piece takes ownership. Disjoint i → no aliasing.
            let buf = unsafe { std::mem::take(&mut *bp.ptr().add(i)) };
            let piece = build_piece_dwyer_dedup::<P>(cr, or, lo, hi, buf);
            unsafe { *sp.ptr().add(i) = piece };
        });
    }
    drop(bufs); // every entry was taken → all empty Vecs now, free to drop
    // A `None` slot means a chunk had <2 distinct points — bail to the robust path.
    let mut pieces: Vec<Piece> = Vec::with_capacity(bounds.len());
    let mut bail = false;
    for s in slots {
        match s {
            Some(p) => pieces.push(p),
            None => {
                bail = true;
                break;
            }
        }
    }
    if prof {
        eprintln!(
            "  par build+dedup:        {:.2} ms",
            tb.elapsed().as_secs_f64() * 1e3
        );
    }
    // A single piece with too few unique points can't form a triangulation.
    let engaged =
        !bail && !(pieces.len() < 2 && pieces.first().map(|p| p.darts.len()).unwrap_or(0) < 6);
    if engaged {
        assemble_and_export::<P>(
            &pool,
            &coord_raw,
            &orig_raw,
            pieces,
            coord_raw.len(),
            threads,
            out,
        );
    }
    // Hand the big buffers back for the next call to reuse (even when bailing).
    put_coord_scratch(coord_raw);
    put_orig_scratch(orig_raw);
    engaged
}

/// Scatter `points` into x-buckets (no within-bucket sort, no dedup) and group
/// whole buckets into ~`nchunks` contiguous, x-separable chunks. Returns the
/// scattered coordinates, their original indices, and the chunk `[lo, hi)`
/// ranges. Because a given x maps to exactly one bucket, every copy of a point
/// lands in the same chunk (so per-chunk dedup is complete) and chunk i's max x
/// is strictly below chunk i+1's min x (so the trunk merge stays valid).
fn bucket_partition_by_x(
    pool: &Pool,
    points: &[[i32; 2]],
    threads: usize,
    nchunks: usize,
    mnx: i32,
    mny: i32,
    mxx: i32,
) -> (Vec<[i32; 3]>, Vec<u32>, Vec<(u32, u32)>) {
    let n = points.len();
    let b = (threads * 16).clamp(1, 8192);
    let span = (mxx as i64 - mnx as i64 + 1).max(1);
    let bucket = move |x: i32| -> usize {
        ((((x as i64 - mnx as i64) * b as i64) / span) as usize).min(b - 1)
    };
    let per = n.div_ceil(threads.max(1)).max(1);
    let nt = n.div_ceil(per);

    // Phase 1: per-chunk histograms (pool).
    let mut locals: Vec<Vec<u32>> = (0..nt).map(|_| Vec::new()).collect();
    {
        let lp = SendPtr(locals.as_mut_ptr());
        pool.for_each(nt, &|c| {
            let chunk = &points[c * per..((c + 1) * per).min(n)];
            let mut h = vec![0u32; b];
            for p in chunk {
                h[bucket(p[0])] += 1;
            }
            unsafe { *lp.ptr().add(c) = h };
        });
    }

    // Phase 2: bucket starts + per-chunk scatter offsets.
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
    let mut off = vec![vec![0u32; b]; nt];
    for bk in 0..b {
        let mut acc = start[bk];
        for c in 0..nt {
            off[c][bk] = acc;
            acc += locals[c][bk];
        }
    }

    // Phase 3: parallel scatter into disjoint positions, packing `[x, y, lift]`
    // (the paraboloid lift is computed here for free, riding the scatter). The
    // scatter is a bijection onto [0, n), so every slot is written exactly once —
    // reuse warm thread-local buffers with an uninit `set_len` (no zero-fill).
    let mut coord_raw = take_coord_scratch();
    let mut orig_raw = take_orig_scratch();
    coord_raw.reserve(n);
    orig_raw.reserve(n);
    unsafe {
        coord_raw.set_len(n);
        orig_raw.set_len(n);
    }
    let cbaseptr = SendPtr(coord_raw.as_mut_ptr());
    let obaseptr = SendPtr(orig_raw.as_mut_ptr());
    let offr = &off;
    pool.for_each(nt, &|c| {
        let mut o = offr[c].clone();
        let cbase = c * per;
        let chunk = &points[c * per..((c + 1) * per).min(n)];
        let cp = cbaseptr.ptr();
        let op = obaseptr.ptr();
        for (li, &pt) in chunk.iter().enumerate() {
            let pos = o[bucket(pt[0])];
            let dx = (pt[0] - mnx) as i64;
            let dy = (pt[1] - mny) as i64;
            unsafe {
                *cp.add(pos as usize) = [pt[0], pt[1], (dx * dx + dy * dy) as i32];
                *op.add(pos as usize) = (cbase + li) as u32;
            }
            o[bucket(pt[0])] += 1;
        }
    });

    // Group whole buckets into ~nchunks contiguous chunks of ~n/nchunks points.
    let target = n.div_ceil(nchunks);
    let mut bounds: Vec<(u32, u32)> = Vec::with_capacity(nchunks);
    let mut prev = 0u32;
    for bk in 1..=b {
        let offb = start[bk];
        let big_enough = (offb - prev) as usize >= target && bounds.len() + 1 < nchunks;
        if (big_enough || bk == b) && offb > prev {
            bounds.push((prev, offb));
            prev = offb;
        }
    }
    (coord_raw, orig_raw, bounds)
}

fn build_piece<P: Pred>(coord: &[[i32; 3]], lo: u32, hi: u32) -> Piece {
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

/// Shared behaviour of a dart arena. The Guibas-Stolfi `merge` and all
/// navigation / predicates / topological mutation are written **once** here as
/// default methods over a few storage primitives, so the serial builder
/// ([`Arena`]) and the concurrent trunk-merge context ([`MergeCtx`]) cannot drift
/// apart. Darts are read and written by value (`[u32; 3]` is `Copy`), so no
/// `&mut`-from-`&` reference is ever formed into the (possibly shared) buffer.
trait DartStore {
    /// The exact-predicate width (fast i64 vs wide i128) this store builds with.
    type Pr: Pred;

    // --- storage primitives (implemented per backing store) ---
    /// Packed coordinates: `[x, y, lift]` where `lift = (x-mnx)² + (y-mny)²` is
    /// the precomputed paraboloid lift for the fast-path `in_circle` (`0` on the
    /// wide path, where the lifted branch is compiled out). Packing `lift` in the
    /// third lane keeps it on the same cache line as `x`/`y`.
    fn coord(&self) -> &[[i32; 3]];
    /// Site `s`'s precomputed lift (`coord[s][2]`, same cache line as `pt`).
    #[inline(always)]
    fn lift(&self, s: u32) -> i64 {
        unsafe { self.coord().get_unchecked(s as usize)[2] as i64 }
    }
    fn org(&self, e: u32) -> u32;
    fn next(&self, e: u32) -> u32;
    fn prev(&self, e: u32) -> u32;
    fn set_org(&mut self, e: u32, v: u32);
    fn set_next(&mut self, e: u32, v: u32);
    fn set_prev(&mut self, e: u32, v: u32);
    /// Allocate a fresh `(e, e^1)` dart pair as two isolated darts with origins
    /// `a`, `b`; returns the even base `e`.
    fn make_edge(&mut self, a: u32, b: u32) -> u32;
    /// Return a deleted edge's even base to the free list.
    fn recycle(&mut self, base: u32);

    // --- navigation (shared) ---
    #[inline(always)]
    fn sym(e: u32) -> u32 {
        e ^ 1
    }
    #[inline(always)]
    fn dest(&self, e: u32) -> u32 {
        self.org(e ^ 1)
    }
    #[inline(always)]
    fn onext(&self, e: u32) -> u32 {
        self.next(e)
    }
    #[inline(always)]
    fn oprev(&self, e: u32) -> u32 {
        self.prev(e)
    }
    #[inline(always)]
    fn lnext(&self, e: u32) -> u32 {
        self.prev(e ^ 1)
    }
    #[inline(always)]
    fn rprev(&self, e: u32) -> u32 {
        self.next(e ^ 1)
    }

    // --- predicates on site ids (shared) ---
    #[inline(always)]
    fn pt(&self, s: u32) -> [i32; 2] {
        let p = unsafe { self.coord().get_unchecked(s as usize) };
        [p[0], p[1]]
    }
    #[inline(always)]
    fn orient3(&self, a: u32, b: u32, c: u32) -> i32 {
        Self::Pr::orient(self.pt(a), self.pt(b), self.pt(c))
    }
    #[inline(always)]
    fn in_circle4(&self, a: u32, b: u32, c: u32, d: u32) -> bool {
        // Fast path: lifted determinant with precomputed lifts. `ap = lift[a] -
        // lift[d]` replaces recomputing `ax²+ay²` each call — 9 multiplies vs 15
        // (translation-invariant paraboloid form; magnitude bound is the same
        // 12·S⁴ ≤ 2⁶³, so the i64 fast-path span still holds). The `USE_MORTON`
        // const monomorphizes the dead branch away.
        if Self::Pr::USE_MORTON {
            let pa = self.pt(a);
            let pb = self.pt(b);
            let pc = self.pt(c);
            let pd = self.pt(d);
            let (dx, dy) = (pd[0] as i64, pd[1] as i64);
            let ax = pa[0] as i64 - dx;
            let ay = pa[1] as i64 - dy;
            let bx = pb[0] as i64 - dx;
            let by = pb[1] as i64 - dy;
            let cx = pc[0] as i64 - dx;
            let cy = pc[1] as i64 - dy;
            let dl = self.lift(d);
            let ap = self.lift(a) - dl;
            let bp = self.lift(b) - dl;
            let cp = self.lift(c) - dl;
            ax * (by * cp - bp * cy) - ay * (bx * cp - bp * cx) + ap * (bx * cy - by * cx) > 0
        } else {
            Self::Pr::in_circle(self.pt(a), self.pt(b), self.pt(c), self.pt(d)) > 0
        }
    }
    #[inline(always)]
    fn left_of(&self, x: u32, e: u32) -> bool {
        self.orient3(x, self.org(e), self.dest(e)) > 0
    }
    #[inline(always)]
    fn right_of(&self, x: u32, e: u32) -> bool {
        self.orient3(x, self.dest(e), self.org(e)) > 0
    }

    // --- topological mutation (shared) ---
    fn splice(&mut self, a: u32, b: u32) {
        let an = self.next(a);
        let bn = self.next(b);
        self.set_next(a, bn);
        self.set_prev(bn, a);
        self.set_next(b, an);
        self.set_prev(an, b);
    }
    fn connect(&mut self, a: u32, b: u32) -> u32 {
        let e = self.make_edge(self.dest(a), self.org(b));
        let la = self.lnext(a);
        self.splice(e, la);
        self.splice(Self::sym(e), b);
        e
    }
    fn delete_edge(&mut self, e: u32) {
        let op = self.oprev(e);
        self.splice(e, op);
        let se = Self::sym(e);
        let ops = self.oprev(se);
        self.splice(se, ops);
        let base = e & !1; // even base of the (e, e^1) pair
        self.set_org(base, DELETED);
        self.set_org(base | 1, DELETED);
        self.recycle(base);
    }

    /// Guibas-Stolfi merge of two sub-triangulations given their facing hull
    /// darts `(ldo, ldi)` and `(rdi, rdo)`; returns the combined outer `(l, r)`.
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
            // basel's endpoints are loop-invariant across the two inner loops.
            let db = self.dest(basel);
            let ob = self.org(basel);
            let sb = Self::sym(basel);

            let mut lcand = self.onext(sb);
            let mut l_valid = self.orient3(self.dest(lcand), db, ob) > 0;
            if l_valid {
                let mut deleted = false;
                loop {
                    let ln = self.onext(lcand);
                    if self.in_circle4(db, ob, self.dest(lcand), self.dest(ln)) {
                        self.delete_edge(lcand);
                        lcand = ln;
                        deleted = true;
                    } else {
                        break;
                    }
                }
                // Only re-test validity if the prune loop actually advanced
                // `lcand`; otherwise it's unchanged and still valid (saves an
                // orient3 on every non-pruning outer iteration — the common case).
                if deleted {
                    l_valid = self.orient3(self.dest(lcand), db, ob) > 0;
                }
            }

            let mut rcand = self.oprev(basel);
            let mut r_valid = self.orient3(self.dest(rcand), db, ob) > 0;
            if r_valid {
                let mut deleted = false;
                loop {
                    let rp = self.oprev(rcand);
                    if self.in_circle4(db, ob, self.dest(rcand), self.dest(rp)) {
                        self.delete_edge(rcand);
                        rcand = rp;
                        deleted = true;
                    } else {
                        break;
                    }
                }
                if deleted {
                    r_valid = self.orient3(self.dest(rcand), db, ob) > 0;
                }
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
}

/// Dart arena: two consecutive darts per undirected edge; `darts[e] = [org,
/// next, prev]`.
struct Arena<'c, P: Pred> {
    coord: &'c [[i32; 3]], // [x, y, lift]
    darts: Vec<[u32; 3]>,
    /// Even bases of deleted edges, reused by `make_edge` so the arena stays at
    /// peak-alive size (~6 darts/point) instead of growing append-only (~18/pt).
    free: Vec<u32>,
    _p: PhantomData<P>,
}

impl<'c, P: Pred> Arena<'c, P> {
    fn with_capacity(coord: &'c [[i32; 3]], cap: usize) -> Self {
        Arena {
            coord,
            darts: Vec::with_capacity(cap),
            free: Vec::new(),
            _p: PhantomData,
        }
    }

    /// Build into a recycled dart buffer (emptied, capacity retained) instead of a
    /// fresh allocation — see the per-piece buffer pool in `run_par_fast`.
    fn from_buf(coord: &'c [[i32; 3]], mut darts: Vec<[u32; 3]>, cap: usize) -> Self {
        darts.clear();
        darts.reserve(cap);
        Arena {
            coord,
            darts,
            free: Vec::new(),
            _p: PhantomData,
        }
    }

    fn into_piece(self, le: u32, re: u32) -> Piece {
        Piece {
            darts: self.darts,
            le,
            re,
        }
    }
}

impl<'c, P: Pred> DartStore for Arena<'c, P> {
    type Pr = P;
    #[inline(always)]
    fn coord(&self) -> &[[i32; 3]] {
        self.coord
    }
    #[inline(always)]
    fn org(&self, e: u32) -> u32 {
        unsafe { self.darts.get_unchecked(e as usize)[0] }
    }
    #[inline(always)]
    fn next(&self, e: u32) -> u32 {
        unsafe { self.darts.get_unchecked(e as usize)[1] }
    }
    #[inline(always)]
    fn prev(&self, e: u32) -> u32 {
        unsafe { self.darts.get_unchecked(e as usize)[2] }
    }
    #[inline(always)]
    fn set_org(&mut self, e: u32, v: u32) {
        unsafe { self.darts.get_unchecked_mut(e as usize)[0] = v };
    }
    #[inline(always)]
    fn set_next(&mut self, e: u32, v: u32) {
        unsafe { self.darts.get_unchecked_mut(e as usize)[1] = v };
    }
    #[inline(always)]
    fn set_prev(&mut self, e: u32, v: u32) {
        unsafe { self.darts.get_unchecked_mut(e as usize)[2] = v };
    }
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
    #[inline(always)]
    fn recycle(&mut self, base: u32) {
        self.free.push(base);
    }
}

impl<'c, P: Pred> Arena<'c, P> {
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

    /// Tag every dart of the outer (unbounded) face `visited` (high bit of `org`)
    /// so export skips them without a per-dart triangle check. `le` is any hull
    /// dart; the outer face is on the side of `le`/`sym(le)` whose `lnext`-ring is
    /// not a strictly-CCW triangle (h > 3 → not a 3-cycle; h == 3 → the CW one).
    /// O(hull). Mutates the arena (fine — it's consumed by export next).
    fn mark_outer_face(&mut self, le: u32) {
        if self.org(le) == DELETED {
            return;
        }
        let mut seed = le;
        for start in [le, Self::sym(le)] {
            let e1 = self.lnext(start);
            let e2 = self.lnext(e1);
            if self.lnext(e2) != start
                || self.orient3(self.org(start), self.org(e1), self.org(e2)) < 0
            {
                seed = start;
                break;
            }
        }
        let mut o = seed;
        loop {
            let v = self.org(o) | VISITED;
            self.set_org(o, v);
            o = self.lnext(o);
            if o == seed {
                break;
            }
        }
    }

    /// Emit the triangle whose minimum dart is `e`, unless `e` is deleted or on
    /// the outer face (both flagged by the `org` high bit). No `lnext(e2)==e`
    /// check: with the outer face pre-marked, every remaining face is a triangle
    /// (Delaunay), and it is CCW by the Guibas-Stolfi invariant.
    #[inline(always)]
    fn emit_face(&self, e: u32, orig: &[u32], out: &mut Vec<[u32; 3]>) {
        let o = self.org(e);
        if o & VISITED != 0 {
            return; // deleted (org == u32::MAX) or outer-face dart
        }
        let e1 = self.lnext(e);
        let e2 = self.lnext(e1);
        if e <= e1 && e <= e2 {
            out.push([
                orig[o as usize],
                orig[self.org(e1) as usize],
                orig[self.org(e2) as usize],
            ]);
        }
    }

    fn export(&mut self, le: u32, orig: &[u32], out: &mut Vec<[u32; 3]>) {
        self.mark_outer_face(le);
        let dart_count = self.darts.len() as u32;
        for e in 0..dart_count {
            self.emit_face(e, orig, out);
        }
    }

    /// Parallel export: darts are cut into more ranges than threads and pulled
    /// from a work-stealing cursor, so fast P-cores scan more ranges than slow
    /// E-cores (the same load balancing the build uses). A triangle is emitted
    /// only from its minimum dart, which lives in exactly one range, so there are
    /// no duplicates and no coordination. The outer face is pre-marked (serial).
    fn export_par(
        &mut self,
        pool: &Pool,
        le: u32,
        orig: &[u32],
        threads: usize,
        out: &mut Vec<[u32; 3]>,
    ) {
        self.mark_outer_face(le);
        let this = &*self; // outer face marked; the parallel scan only reads
        let dc = this.darts.len() as u32;
        let per = dc.div_ceil((threads * 4).max(1) as u32).max(1);
        let nranges = dc.div_ceil(per) as usize;
        let mut slots: Vec<Vec<[u32; 3]>> = (0..nranges).map(|_| Vec::new()).collect();
        {
            let sp = SendPtr(slots.as_mut_ptr());
            pool.for_each(nranges, &|r| {
                let lo = r as u32 * per;
                let hi = (lo + per).min(dc);
                let mut v = Vec::new();
                for e in lo..hi {
                    this.emit_face(e, orig, &mut v);
                }
                unsafe { *sp.ptr().add(r) = v };
            });
        }
        // Reassemble in range order (deterministic output).
        for v in &slots {
            out.extend_from_slice(v);
        }
    }
}

// ============================================================================
// Parallel trunk merge (tree reduction)
// ============================================================================

/// A single Guibas-Stolfi merge running against a *shared* dart buffer. New
/// edges are bump-allocated from one atomic cursor; deleted slots go on a
/// **private** free list. Within a merge-tree level, sibling merges touch
/// disjoint sub-triangulations and claim disjoint cursor slots, so the shared
/// raw pointer is only ever dereferenced at non-overlapping addresses — sound
/// without locking. A barrier between levels orders the dependent merges.
struct MergeCtx<'c, P: Pred> {
    coord: &'c [[i32; 3]], // [x, y, lift]
    base: *mut [u32; 3],   // whole shared buffer; indices are global
    cap: u32,              // buffer capacity (bump must stay below this)
    cursor: &'c AtomicU32,
    free: Vec<u32>, // even bases this merge deleted, reused before bumping
    _p: PhantomData<P>,
}

impl<'c, P: Pred> DartStore for MergeCtx<'c, P> {
    type Pr = P;
    #[inline(always)]
    fn coord(&self) -> &[[i32; 3]] {
        self.coord
    }
    // Reads and writes go straight through the shared raw pointer *by value*
    // (`[u32; 3]` is `Copy`), so no `&`/`&mut` reference into the shared buffer is
    // ever formed. Sibling merges only ever touch disjoint slots (see the type
    // doc), which is what makes concurrent access sound — Miri-checked by
    // `internal_tests::parallel_raw_pointer_path`.
    #[inline(always)]
    fn org(&self, e: u32) -> u32 {
        unsafe { (*self.base.add(e as usize))[0] }
    }
    #[inline(always)]
    fn next(&self, e: u32) -> u32 {
        unsafe { (*self.base.add(e as usize))[1] }
    }
    #[inline(always)]
    fn prev(&self, e: u32) -> u32 {
        unsafe { (*self.base.add(e as usize))[2] }
    }
    #[inline(always)]
    fn set_org(&mut self, e: u32, v: u32) {
        unsafe { (*self.base.add(e as usize))[0] = v };
    }
    #[inline(always)]
    fn set_next(&mut self, e: u32, v: u32) {
        unsafe { (*self.base.add(e as usize))[1] = v };
    }
    #[inline(always)]
    fn set_prev(&mut self, e: u32, v: u32) {
        unsafe { (*self.base.add(e as usize))[2] = v };
    }
    fn make_edge(&mut self, a: u32, b: u32) -> u32 {
        let e = if let Some(e) = self.free.pop() {
            e
        } else {
            let e = self.cursor.fetch_add(2, Ordering::Relaxed);
            // Fail loud (panic, not OOB write) if the slack is exhausted — the
            // caller sizes `cap` well above the seam edges a merge can create.
            assert!(e + 1 < self.cap, "merge_tree_par: dart buffer overflow");
            e
        };
        unsafe {
            *self.base.add(e as usize) = [a, e, e];
            *self.base.add((e + 1) as usize) = [b, e + 1, e + 1];
        }
        e
    }
    #[inline(always)]
    fn recycle(&mut self, base: u32) {
        self.free.push(base);
    }
}

/// Stitch x-adjacent pieces (each `(xl, xr)` in `nodes`, global dart indices)
/// into one Delaunay triangulation via a **balanced tree of merges**: each level
/// merges adjacent pairs concurrently (an odd tail carries up), halving the
/// serial merge chain. All merges share `darts` (pre-reserved to `cap`); the
/// atomic cursor hands out fresh dart slots. Sets `darts.len()` to the final
/// high-water mark. `total` is the count of already-placed piece darts.
fn merge_tree_par<P: Pred>(
    pool: &Pool,
    coord: &[[i32; 3]],
    darts: &mut Vec<[u32; 3]>,
    mut nodes: Vec<(u32, u32)>,
    total: u32,
    cap: u32,
) -> u32 {
    let cursor = AtomicU32::new(total);
    let dartptr = SendPtr(darts.as_mut_ptr());
    while nodes.len() > 1 {
        let m = nodes.len();
        let npairs = m / 2;
        // Each level's merges are independent (disjoint sub-triangulations); run
        // them on the pool. Results land in disjoint slots by pair index.
        let mut next: Vec<(u32, u32)> = vec![(0, 0); npairs];
        {
            let sp = SendPtr(next.as_mut_ptr());
            let cur = &cursor;
            let nd = &nodes;
            pool.for_each(npairs, &|k| {
                let (l, r) = (nd[2 * k], nd[2 * k + 1]);
                let mut ctx = MergeCtx::<P> {
                    coord,
                    base: dartptr.ptr(),
                    cap,
                    cursor: cur,
                    free: Vec::new(),
                    _p: PhantomData,
                };
                let res = ctx.merge(l.0, l.1, r.0, r.1);
                unsafe { *sp.ptr().add(k) = res };
            });
        }
        if m % 2 == 1 {
            next.push(nodes[m - 1]); // odd tail: carry the rightmost node up
        }
        nodes = next;
    }
    // Every slot in [0, hw) is initialized: [0, total) by the concat, and each
    // bump in [total, hw) is written inside make_edge before any navigation.
    let hw = cursor.load(Ordering::Relaxed) as usize;
    unsafe { darts.set_len(hw) };
    nodes[0].0 // final combined outer-left hull dart (for export's outer-face skip)
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
fn morton_spread(v: u32) -> u64 {
    // Interleave a full 32-bit coordinate into the even bits of a u64, so the
    // Morton code is injective over the entire i32 coordinate range (rebased to
    // u32). This keeps "equal key ⟺ equal point" — the invariant the dedup and
    // the alternating-cut split both rely on — valid at any span, not just ≤16 bits.
    let mut v = v as u64;
    v = (v | (v << 16)) & 0x0000_ffff_0000_ffff;
    v = (v | (v << 8)) & 0x00ff_00ff_00ff_00ff;
    v = (v | (v << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    v = (v | (v << 2)) & 0x3333_3333_3333_3333;
    (v | (v << 1)) & 0x5555_5555_5555_5555
}
#[inline]
fn morton_code(x: u32, y: u32) -> u64 {
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
    ///
    /// Below `MORTON_LEAF` points the Morton recursion stops and the whole range
    /// is triangulated by the x-cut D&C (`delaunay_slice`) instead. The Morton
    /// merge must `scan_dhulls` (walk the full outer hull) after *every* merge to
    /// recover the four directional extremes, whereas `delaunay_slice`'s merge
    /// tracks `(le, re)` for free — so pushing the many small bottom-level merges
    /// into `delaunay_slice` cuts the total hull-scan work by ~6× (this is why the
    /// reference C++ uses a 16-point Morton leaf).
    fn build_dwyer(&mut self, pos: &mut [u32], keys: &[u64], lo: u32, hi: u32) -> DHulls {
        const MORTON_LEAF: u32 = 16;
        let split = if hi - lo <= MORTON_LEAF {
            None
        } else {
            let diff = keys[lo as usize] ^ keys[(hi - 1) as usize];
            if diff == 0 {
                None
            } else {
                let bit = 63 - diff.leading_zeros();
                let mask = 1u64 << bit;
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
                // Sort this leaf's ids in place (leaves own disjoint `pos`
                // ranges, so no cross-leaf interference) — avoids a tiny Vec
                // allocation per leaf (~m/3 leaves at 1M).
                let ids = &mut pos[lo as usize..hi as usize];
                ids.sort_unstable_by_key(|&a| self.pt(a));
                let (le, _re) = self.delaunay_slice(ids);
                self.scan_dhulls(Self::sym(le))
            }
        }
    }
}

/// Morton-sorted positions of `coord[lo..hi]` and their 64-bit Morton keys
/// (rebased to the range's own origin). Injective over any i32 coordinate span.
fn morton_positions(coord: &[[i32; 3]], lo: u32, hi: u32) -> (Vec<u32>, Vec<u64>) {
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
    let mut keys: Vec<u64> = pos.iter().map(|&i| key(i)).collect();
    // LSD radix sort pos by the u64 Morton keys (O(n)), 11 bits per pass. Only as
    // many passes as the actual key magnitude needs: a 16-bit-span range (≤32-bit
    // Morton) still costs 3 passes as before, and only larger spans pay for more.
    const RADIX: usize = 1 << 11;
    const MASK: u64 = (RADIX as u64) - 1;
    let n = pos.len();
    let maxkey = keys.iter().copied().max().unwrap_or(0);
    let (mut pos2, mut key2) = take_morton_scratch(n);
    let mut shift = 0u32;
    // `shift < 64` guards the final iteration: a 64-bit Morton key needs 6 passes
    // of 11 bits, so `shift` reaches 66 after covering bits 55..=65 — evaluating
    // `maxkey >> 66` would panic (debug) or wrap the shift to `>> 2` (release,
    // → wrong passes / non-termination). Once `shift >= 64` every bit is covered.
    while shift < 64 && (maxkey >> shift) != 0 {
        let mut cnt = [0u32; RADIX + 1];
        for &k in keys.iter() {
            cnt[((k >> shift) & MASK) as usize + 1] += 1;
        }
        for i in 0..RADIX {
            cnt[i + 1] += cnt[i];
        }
        for i in 0..n {
            let d = ((keys[i] >> shift) & MASK) as usize;
            let p = cnt[d] as usize;
            cnt[d] += 1;
            pos2[p] = pos[i];
            key2[p] = keys[i];
        }
        std::mem::swap(&mut pos, &mut pos2);
        std::mem::swap(&mut keys, &mut key2);
        shift += 11;
    }
    // `pos2`/`key2` now hold the leftover (non-result) buffers — return them.
    put_morton_scratch(pos2, key2);
    (pos, keys)
}

/// Serial build via Morton alternating cuts (fast path only; span ≤ 29609).
fn run_dwyer<P: Pred>(coord: &[[i32; 3]], orig: &[u32], out: &mut Vec<[u32; 3]>) {
    let prof = std::env::var_os("GEO_PROF").is_some();
    let m = coord.len();
    let t = std::time::Instant::now();
    let (pos, keys) = morton_positions(coord, 0, m as u32);
    // Compact coord+orig into Morton order so the latency-bound build accesses
    // spatially-adjacent points sequentially instead of gathering from the
    // x-sorted array. Site ids become the identity (Morton index). The packed
    // lift `[2]` (global-origin) rides along, so no recompute is needed.
    let coord_c: Vec<[i32; 3]> = pos.iter().map(|&j| coord[j as usize]).collect();
    let orig_c: Vec<u32> = pos.iter().map(|&j| orig[j as usize]).collect();
    let mut ident: Vec<u32> = (0..m as u32).collect();
    if prof {
        eprintln!(
            "  [dwyer] morton+compact: {:.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let mut arena = Arena::<P>::with_capacity(&coord_c, m.saturating_mul(8));
    let t = std::time::Instant::now();
    let d = arena.build_dwyer(&mut ident, &keys, 0, m as u32);
    if prof {
        eprintln!(
            "  [dwyer] build_dwyer:  {:.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let t = std::time::Instant::now();
    arena.export(d.xl, &orig_c, out);
    if prof {
        eprintln!(
            "  [dwyer] export:       {:.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
}

/// Build one parallel chunk (coord[lo..hi]) via Dwyer; returns a Piece whose
/// (le, re) are its x-extremes for the left-to-right trunk merge.
fn build_piece_dwyer<P: Pred>(coord: &[[i32; 3]], lo: u32, hi: u32) -> Piece {
    // No Morton coord-compaction here: each chunk's coord range is already
    // L2-resident (~24k pts / 192 KB at the default 42-way split), so the
    // gather+remap costs more than the marginal L1 locality it would buy
    // (measured neutral-to-negative). Compaction is applied on the serial path
    // (`run_dwyer`), where the whole 8 MB coord array is not cache-resident.
    let n = (hi - lo) as usize;
    let (mut pos, keys) = morton_positions(coord, lo, hi);
    let mut arena = Arena::<P>::with_capacity(coord, n.saturating_mul(8));
    let d = arena.build_dwyer(&mut pos, &keys, 0, n as u32);
    arena.into_piece(d.xl, d.xr)
}

/// Build one fast-path chunk from *raw* (possibly duplicated, unsorted-within)
/// points `coord_raw[lo..hi]`, deduplicating during the Morton sort: after the
/// radix sort, identical points are adjacent (a Morton code is a bijection on the
/// rebased 16-bit coordinates, so equal key ⟺ equal point), and each run keeps
/// the lowest original index. Darts reference global `coord_raw` indices. Returns
/// `None` if fewer than 2 distinct points remain (caller falls back).
fn build_piece_dwyer_dedup<P: Pred>(
    coord_raw: &[[i32; 3]],
    orig_raw: &[u32],
    lo: u32,
    hi: u32,
    buf: Vec<[u32; 3]>,
) -> Option<Piece> {
    let (mut pos, mut keys) = morton_positions(coord_raw, lo, hi);
    let n = pos.len();
    // Compact runs of equal keys (== identical points) in place, keeping the
    // lowest original index. Duplicates are rare, so the singleton fast path
    // does no `orig_raw` lookup (the dominant case is a near no-op copy).
    let mut w = 0usize;
    let mut i = 0usize;
    while i < n {
        let k = keys[i];
        if i + 1 == n || keys[i + 1] != k {
            pos[w] = pos[i];
            keys[w] = k;
            w += 1;
            i += 1;
        } else {
            // Run of ≥2 identical points: pick the lowest original index.
            let mut best = pos[i];
            let mut best_orig = orig_raw[pos[i] as usize];
            let mut j = i + 1;
            while j < n && keys[j] == k {
                let o = orig_raw[pos[j] as usize];
                if o < best_orig {
                    best_orig = o;
                    best = pos[j];
                }
                j += 1;
            }
            pos[w] = best;
            keys[w] = k;
            w += 1;
            i = j;
        }
    }
    if w < 2 {
        return None;
    }
    pos.truncate(w);
    keys.truncate(w);
    let mut arena = Arena::<P>::from_buf(coord_raw, buf, w.saturating_mul(8));
    let d = arena.build_dwyer(&mut pos, &keys, 0, w as u32);
    Some(arena.into_piece(d.xl, d.xr))
}

/// Triangulate using Dwyer alternating cuts (Morton order). Falls back to the
/// x-cut path for the wide predicate range (coords don't fit 16-bit Morton).
pub fn triangulate_dwyer(points: &[[i32; 2]]) -> Result<Vec<[u32; 3]>, GeoError> {
    let Some((coord, orig, sx, sy)) = prepare(points) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    match predicate_width(sx, sy) {
        // Both widths use the Morton/Dwyer build; only the predicate arithmetic
        // differs (i64 lifted vs i128). The wide path used to fall back to the
        // plain x-cut Guibas-Stolfi `run`, which was ~4× slower — the 64-bit
        // Morton makes the fast build valid for the full coordinate range.
        PredicateWidth::Int64 => run_dwyer::<PredFast>(&coord, &orig, &mut out),
        PredicateWidth::Int128 if abl_no_wide_dwyer() => run::<PredWide>(&coord, &orig, &mut out),
        PredicateWidth::Int128 => run_dwyer::<PredWide>(&coord, &orig, &mut out),
        PredicateWidth::Unsupported => return Err(span_error(sx, sy)),
    }
    Ok(out)
}

#[cfg(test)]
mod internal_tests {
    use super::*;

    /// Deterministic distinct points inside the i64 fast-path span.
    fn distinct(n: usize) -> Vec<[i32; 2]> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        let mut s: u64 = 0x1234_5678_9abc_def1;
        while out.len() < n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let p = [((s >> 16) % 20_000) as i32, ((s >> 34) % 20_000) as i32];
            if seen.insert(p) {
                out.push(p);
            }
        }
        out
    }

    /// Exercises the whole parallel raw-pointer path (SendPtr scatter in
    /// `bucket_partition_by_x`, the shared-arena concat, and the `MergeCtx` tree
    /// merge) on a small input so it is tractable under Miri, which checks the
    /// `unsafe` for UB / aliasing / provenance:
    ///   cargo +nightly miri test -p rlx-geo --no-default-features parallel_raw_pointer_path
    #[test]
    fn parallel_raw_pointer_path() {
        // >= 2 chunks (the n/2000 floor) so `merge_tree_par` actually merges.
        let p = distinct(4200);
        let (mnx, mny, mxx) = p
            .iter()
            .fold((i32::MAX, i32::MAX, i32::MIN), |(ax, ay, bx), q| {
                (ax.min(q[0]), ay.min(q[1]), bx.max(q[0]))
            });
        let mut par = Vec::new();
        let engaged = run_par_fast::<PredFast>(&p, 2, mnx, mny, mxx, &mut par);
        assert!(
            engaged,
            "fast parallel path should engage for distinct points"
        );
        assert!(!par.is_empty());
        // Triangle count is invariant across any valid Delaunay of the same set,
        // so it must equal the (independently built) serial result.
        assert_eq!(
            par.len(),
            triangulate(&p).unwrap().len(),
            "parallel count != serial"
        );
    }

    /// `release_scratch` frees the thread-local pools; the next call must
    /// re-allocate and still produce a valid triangulation of the same size.
    /// (Triangle *count* is the invariant — the exact set can differ at cocircular
    /// ties, which the parallel path resolves in thread-race order regardless.)
    #[test]
    fn release_scratch_preserves_correctness() {
        let p = distinct(9000); // ≥ PARALLEL_MIN so the parallel/pool path runs
        let a = triangulate_par(&p, 0).unwrap();
        release_scratch();
        let b = triangulate_par(&p, 0).unwrap(); // re-allocates the freed pools
        release_scratch();
        assert!(!a.is_empty() && !b.is_empty());
        assert_eq!(a.len(), b.len(), "count changed across release_scratch");
        // Matches the serial reference count too (the size invariant).
        assert_eq!(a.len(), triangulate(&p).unwrap().len());
    }
}
