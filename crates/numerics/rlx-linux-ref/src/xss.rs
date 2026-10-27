// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `np.argsort` (default `kind='quicksort'`) as NumPy 2.3.5 runs it on an AVX2 x86-64
//! CPU (the Linux reference, msi): Intel's x86-simd-sort (BSD-3-Clause, the copy
//! NumPy vendors under `npysort/x86-simd-sort`), `avx2_argsort` for 64-bit keys
//! with 64-bit indices — four lanes per register. The order it gives equal keys is
//! a property of its networks, not of the keys, so it is reproduced lane for lane:
//!
//! * arrays already sorted are returned as the identity (`std::is_sorted`);
//! * up to 256 elements: `argsort_n` — one bitonic key/index network over
//!   `numVecs` registers (halved while `2N ≤ 4·numVecs`), padded with the key type's
//!   maximum and index `u64::MAX`;
//! * above: a quicksort whose pivot is the third of four sorted samples and whose
//!   partition writes each register through the AVX2 compress-store permutation
//!   (below-pivot lanes first in order, the others from the back in reverse) to
//!   both ends, as `avx2_double_compressstore64` does.
//!
//! On macOS NumPy uses its scalar introsort instead — this module is the Linux
//! x86-64 spelling only.

/// A 64-bit key with AVX2's comparison semantics.
pub trait Key: Copy {
    fn lt(self, o: Self) -> bool;
    fn eq_(self, o: Self) -> bool;
    /// `vtype::min(x, y)` / `vtype::max(x, y)` lane semantics.
    fn vmin(x: Self, y: Self) -> Self;
    fn vmax(x: Self, y: Self) -> Self;
    fn type_max() -> Self;
    fn type_min() -> Self;
}

impl Key for i64 {
    fn lt(self, o: Self) -> bool {
        self < o
    }
    fn eq_(self, o: Self) -> bool {
        self == o
    }
    // avx2_emu_min/max: blendv on cmpgt
    fn vmin(x: Self, y: Self) -> Self {
        if x > y { y } else { x }
    }
    fn vmax(x: Self, y: Self) -> Self {
        if x > y { x } else { y }
    }
    fn type_max() -> Self {
        i64::MAX
    }
    fn type_min() -> Self {
        i64::MIN
    }
}

impl Key for f64 {
    fn lt(self, o: Self) -> bool {
        self < o
    }
    fn eq_(self, o: Self) -> bool {
        self == o
    }
    // _mm256_min_pd / _mm256_max_pd: the second operand unless the first wins strictly
    fn vmin(x: Self, y: Self) -> Self {
        if x < y { x } else { y }
    }
    fn vmax(x: Self, y: Self) -> Self {
        if x > y { x } else { y }
    }
    fn type_max() -> Self {
        f64::INFINITY
    }
    fn type_min() -> Self {
        f64::NEG_INFINITY
    }
}

type Ix = [u64; 4];

#[derive(Clone, Copy)]
struct Reg<K> {
    k: [K; 4],
    i: Ix,
}

fn perm<T: Copy>(v: [T; 4], p: [usize; 4]) -> [T; 4] {
    [v[p[0]], v[p[1]], v[p[2]], v[p[3]]]
}

const SWAP2: [usize; 4] = [1, 0, 3, 2]; // permute4x64 0b10110001
const SWAP4: [usize; 4] = [2, 3, 0, 1]; // permute4x64 0b01001110
const REV: [usize; 4] = [3, 2, 1, 0]; // SHUFFLE_MASK(0, 1, 2, 3)

/// The key/index `cmp_merge(in1, in2, idx1, idx2, mask)`: masked lanes take the max,
/// the others the min; each lane keeps its own index iff its key did not change.
fn cmp_merge<K: Key>(r: Reg<K>, p: [usize; 4], mask: u32) -> Reg<K> {
    let (k2, i2) = (perm(r.k, p), perm(r.i, p));
    let mut out = r;
    for l in 0..4 {
        let mn = K::vmin(k2[l], r.k[l]);
        let mx = K::vmax(k2[l], r.k[l]);
        let t = if mask >> l & 1 != 0 { mx } else { mn };
        out.i[l] = if t.eq_(r.k[l]) { r.i[l] } else { i2[l] };
        out.k[l] = t;
    }
    out
}

/// `sort_reg_4lanes` (key/index).
fn sort_reg4<K: Key>(r: Reg<K>) -> Reg<K> {
    let r = cmp_merge(r, SWAP2, 0xA);
    let r = cmp_merge(r, REV, 0xC);
    cmp_merge(r, SWAP2, 0xA)
}

/// `bitonic_merge_reg_4lanes` (key/index).
fn merge_reg4<K: Key>(r: Reg<K>) -> Reg<K> {
    let r = cmp_merge(r, SWAP4, 0xC);
    cmp_merge(r, SWAP2, 0xA)
}

fn reverse<K: Key>(r: Reg<K>) -> Reg<K> {
    Reg {
        k: perm(r.k, REV),
        i: perm(r.i, REV),
    }
}

/// `COEX(key1, key2, index1, index2)` across two registers.
fn coex<K: Key>(a: &mut Reg<K>, b: &mut Reg<K>) {
    for l in 0..4 {
        let t1 = K::vmin(a.k[l], b.k[l]);
        let t2 = K::vmax(a.k[l], b.k[l]);
        let eq = t1.eq_(a.k[l]);
        let (i1, i2) = (a.i[l], b.i[l]);
        a.i[l] = if eq { i1 } else { i2 };
        b.i[l] = if eq { i2 } else { i1 };
        a.k[l] = t1;
        b.k[l] = t2;
    }
}

fn merge_n_vec<K: Key>(v: &mut [Reg<K>]) {
    let n = v.len();
    if n >= 2 {
        for i in 0..n / 2 {
            let j = n - i - 1;
            v[j] = reverse(v[j]);
            let (lo, hi) = v.split_at_mut(j);
            coex(&mut lo[i], &mut hi[0]);
            v[j] = reverse(v[j]);
        }
    }
    let mut num = n / 2;
    while num >= 2 {
        let mut j = 0;
        while j < n {
            for i in 0..num / 2 {
                let (a, b) = (i + j, i + j + num / 2);
                let (lo, hi) = v.split_at_mut(b);
                coex(&mut lo[a], &mut hi[0]);
            }
            j += num;
        }
        num /= 2;
    }
    for r in v.iter_mut() {
        *r = merge_reg4(*r);
    }
}

/// `argsort_n<vtype, argtype, 256>` on `arg` (`n ≤ 256` indices into `keys`).
fn argsort_n<K: Key>(keys: &[K], arg: &mut [u64]) {
    let n = arg.len();
    let mut nv = 64usize;
    while nv > 1 && n * 2 <= nv * 4 {
        nv /= 2;
    }
    let mut regs: Vec<Reg<K>> = (0..nv)
        .map(|v| {
            let mut r = Reg {
                k: [K::type_max(); 4],
                i: [u64::MAX; 4],
            };
            for l in 0..4 {
                let p = v * 4 + l;
                if p < n {
                    r.i[l] = arg[p];
                    r.k[l] = keys[arg[p] as usize];
                }
            }
            r
        })
        .collect();
    for r in regs.iter_mut() {
        *r = sort_reg4(*r);
    }
    let mut per = 2;
    while per <= nv {
        for g in 0..nv / per {
            merge_n_vec(&mut regs[g * per..(g + 1) * per]);
        }
        per *= 2;
    }
    for (p, a) in arg.iter_mut().enumerate() {
        *a = regs[p / 4].i[p % 4];
    }
}

/// `partition_vec_avx2`: write `idx` permuted (below-pivot lanes first, in order;
/// the rest from the back, reversed) to `arg[left..left+4]`, then to
/// `arg[right-4..right]`; returns how many lanes were `≥ pivot`.
#[allow(clippy::too_many_arguments)]
fn partition_vec<K: Key>(
    arg: &mut [u64],
    left: usize,
    right: usize,
    idx: Ix,
    cur: [K; 4],
    pivot: K,
    mn: &mut [K; 4],
    mx: &mut [K; 4],
) -> usize {
    let mut out = [0u64; 4];
    let (mut lo, mut hi) = (0usize, 4usize);
    let mut ge_count = 0;
    for l in 0..4 {
        let ge = !cur[l].lt(pivot);
        if ge {
            hi -= 1;
            out[hi] = idx[l];
            ge_count += 1;
        } else {
            out[lo] = idx[l];
            lo += 1;
        }
    }
    arg[left..left + 4].copy_from_slice(&out);
    arg[right - 4..right].copy_from_slice(&out);
    for l in 0..4 {
        mn[l] = K::vmin(cur[l], mn[l]);
        mx[l] = K::vmax(cur[l], mx[l]);
    }
    ge_count
}

fn load<K: Key>(keys: &[K], arg: &[u64], at: usize) -> (Ix, [K; 4]) {
    let i = [arg[at], arg[at + 1], arg[at + 2], arg[at + 3]];
    (
        i,
        [
            keys[i[0] as usize],
            keys[i[1] as usize],
            keys[i[2] as usize],
            keys[i[3] as usize],
        ],
    )
}

fn reduce_min<K: Key>(v: [K; 4]) -> K {
    let a = K::vmin(v[0], v[1]);
    let b = K::vmin(v[2], v[3]);
    K::vmin(a, b)
}

fn reduce_max<K: Key>(v: [K; 4]) -> K {
    let a = K::vmax(v[0], v[1]);
    let b = K::vmax(v[2], v[3]);
    K::vmax(a, b)
}

/// `std::min(a, b, comp)` / `std::max(a, b, comp)`.
fn smin<K: Key>(a: K, b: K) -> K {
    if b.lt(a) { b } else { a }
}
fn smax<K: Key>(a: K, b: K) -> K {
    if a.lt(b) { b } else { a }
}

/// `argpartition_unrolled<vtype, argtype, num_unroll>` over `[left, right)`.
#[allow(clippy::too_many_arguments)]
fn argpartition<K: Key>(
    keys: &[K],
    arg: &mut [u64],
    mut left: usize,
    mut right: usize,
    pivot: K,
    smallest: &mut K,
    biggest: &mut K,
    unroll: usize,
) -> usize {
    const L: usize = 4;
    if unroll > 1 && right - left <= 8 * unroll * L {
        return argpartition(keys, arg, left, right, pivot, smallest, biggest, 1);
    }
    let mut r = (right - left) % (unroll * L);
    while r > 0 {
        let v = keys[arg[left] as usize];
        *smallest = smin(*smallest, v);
        *biggest = smax(*biggest, v);
        if !v.lt(pivot) {
            right -= 1;
            arg.swap(left, right);
        } else {
            left += 1;
        }
        r -= 1;
    }
    if left == right {
        return left;
    }
    let mut mn = [*smallest; 4];
    let mut mx = [*biggest; 4];
    if unroll == 1 && right - left == L {
        let (iv, kv) = load(keys, arg, left);
        let amount = partition_vec(arg, left, left + L, iv, kv, pivot, &mut mn, &mut mx);
        *smallest = reduce_min(mn);
        *biggest = reduce_max(mx);
        return left + (L - amount);
    }
    let lefts: Vec<(Ix, [K; 4])> = (0..unroll)
        .map(|ii| load(keys, arg, left + L * ii))
        .collect();
    let rights: Vec<(Ix, [K; 4])> = (0..unroll)
        .map(|ii| load(keys, arg, right - L * (unroll - ii)))
        .collect();
    let mut r_store = right - L;
    let mut l_store = left;
    left += unroll * L;
    right -= unroll * L;
    while right - left != 0 {
        let vecs: Vec<(Ix, [K; 4])>;
        if (r_store + L) - right < left - l_store {
            right -= unroll * L;
            vecs = (0..unroll)
                .map(|ii| load(keys, arg, right + ii * L))
                .collect();
        } else {
            vecs = (0..unroll)
                .map(|ii| load(keys, arg, left + ii * L))
                .collect();
            left += unroll * L;
        }
        for (iv, kv) in vecs {
            let amount = partition_vec(arg, l_store, r_store + L, iv, kv, pivot, &mut mn, &mut mx);
            l_store += L - amount;
            r_store -= amount;
        }
    }
    if unroll == 1 {
        // `argpartition`: the left vector, then the right one stored at l_store only
        let (iv, kv) = lefts[0];
        let amount = partition_vec(arg, l_store, r_store + L, iv, kv, pivot, &mut mn, &mut mx);
        l_store += L - amount;
        let (iv, kv) = rights[0];
        let amount = partition_vec(arg, l_store, l_store + L, iv, kv, pivot, &mut mn, &mut mx);
        l_store += L - amount;
    } else {
        for (iv, kv) in lefts.into_iter().chain(rights) {
            let amount = partition_vec(arg, l_store, r_store + L, iv, kv, pivot, &mut mn, &mut mx);
            l_store += L - amount;
            r_store -= amount;
        }
    }
    *smallest = reduce_min(mn);
    *biggest = reduce_max(mx);
    l_store
}

/// `argsort_` on `[left, right]` (inclusive).
fn argsort_rec<K: Key>(keys: &[K], arg: &mut [u64], left: usize, right: usize, max_iters: i64) {
    assert!(
        max_iters > 0,
        "x86-simd-sort fell back to std::sort (not modelled)"
    );
    if right + 1 - left <= 256 {
        argsort_n(keys, &mut arg[left..=right]);
        return;
    }
    // get_pivot_64bit: the third of four sorted samples
    let size = (right - left) / 4;
    let mut s = [
        keys[arg[left + 4 * size] as usize],
        keys[arg[left + 3 * size] as usize],
        keys[arg[left + 2 * size] as usize],
        keys[arg[left + size] as usize],
    ];
    s.sort_by(|a, b| {
        if a.lt(*b) {
            std::cmp::Ordering::Less
        } else if b.lt(*a) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });
    let pivot = s[2];
    let mut smallest = K::type_max();
    let mut biggest = K::type_min();
    let pi = argpartition(
        keys,
        arg,
        left,
        right + 1,
        pivot,
        &mut smallest,
        &mut biggest,
        4,
    );
    if !pivot.eq_(smallest) {
        argsort_rec(keys, arg, left, pi - 1, max_iters - 1);
    }
    if !pivot.eq_(biggest) {
        argsort_rec(keys, arg, pi, right, max_iters - 1);
    }
}

fn argsort_generic<K: Key>(v: &[K]) -> Vec<usize> {
    let n = v.len();
    let mut arg: Vec<u64> = (0..n as u64).collect();
    if n > 1 && !(1..n).all(|i| !v[i].lt(v[i - 1])) {
        let max_iters = 2 * (n as f64).log2() as i64;
        argsort_rec(v, &mut arg, 0, n - 1, max_iters);
    }
    arg.into_iter().map(|a| a as usize).collect()
}

/// `np.argsort(v)` for `int64` on NumPy 2.3.5 / AVX2 (x86-simd-sort).
pub fn argsort_i64(v: &[i64]) -> Vec<usize> {
    argsort_generic(v)
}

/// `np.argsort(v)` for `float64` without NaN on NumPy 2.3.5 / AVX2 (with a NaN NumPy
/// takes `std::sort`, not modelled: panics).
pub fn argsort_f64(v: &[f64]) -> Vec<usize> {
    assert!(
        !v.iter().any(|x| x.is_nan()),
        "argsort_f64: NaN input takes std::sort"
    );
    argsort_generic(v)
}
