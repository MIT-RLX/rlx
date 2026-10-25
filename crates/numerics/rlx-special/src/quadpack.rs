// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! SciPy's QUADPACK (`scipy/integrate/__quadpack.c`, the C translation SciPy
//! ≥ 1.13 ships) — the infinite-interval driver `dqagie` with its 15-point
//! Kronrod rule `dqk15i`, the ε-algorithm `dqelg` and the error-list sort
//! `dqpsrt` — and the Studentized-range distribution built on it
//! (`scipy.stats.studentized_range` with `df = inf`).
//!
//! FMA contraction follows clang's `-ffp-contract=on`: the Kronrod accumulations
//! `resk = resk + wgk[j]*fsum` are single fused operations.

const EPMACH: f64 = 2.220446049250313e-016;
const UFLOW: f64 = 2.2250738585072014e-308;
const OFLOW: f64 = 1.7976931348623157e+308;

/// `dqk15i(f, boun, inf, a, b)` → `(result, abserr, resabs, resasc)`.
fn dqk15i(
    f: &mut dyn FnMut(f64) -> f64,
    boun: f64,
    inf: i32,
    a: f64,
    b: f64,
) -> (f64, f64, f64, f64) {
    const WG: [f64; 8] = [
        0.000000000000000000000000000000000,
        0.129484966168869693270611432679082,
        0.000000000000000000000000000000000,
        0.279705391489276667901467771423780,
        0.000000000000000000000000000000000,
        0.381830050505118944950369775488975,
        0.000000000000000000000000000000000,
        0.417959183673469387755102040816327,
    ];
    const XGK: [f64; 8] = [
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,
        0.586087235467691130294144838258730,
        0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
        0.000000000000000000000000000000000,
    ];
    const WGK: [f64; 8] = [
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
        0.209482141084727828012999174891714,
    ];
    let dinf = f64::from(inf.min(1));
    let centr = 0.5 * (a + b);
    let hlgth = 0.5 * (b - a);
    let tabsc1 = boun + dinf * (1.0 - centr) / centr;
    let mut fval1 = f(tabsc1);
    if inf == 2 {
        fval1 += f(-tabsc1);
    }
    let fc = (fval1 / centr) / centr;
    let mut resg = fc * WG[7];
    let mut resk = fc * WGK[7];
    let mut resabs = resk.abs();
    let mut fv1 = [0.0f64; 7];
    let mut fv2 = [0.0f64; 7];
    for j in 0..7 {
        let absc = hlgth * XGK[j];
        let absc1 = centr - absc;
        let absc2 = centr + absc;
        let t1 = boun + dinf * (1.0 - absc1) / absc1;
        let t2 = boun + dinf * (1.0 - absc2) / absc2;
        // evaluation order: f(t1), f(t2), then (inf == 2) f(-t1), f(-t2)
        let mut f1 = f(t1);
        let mut f2 = f(t2);
        if inf == 2 {
            f1 += f(-t1);
            f2 += f(-t2);
        }
        let f1 = (f1 / absc1) / absc1;
        let f2 = (f2 / absc2) / absc2;
        fv1[j] = f1;
        fv2[j] = f2;
        let fsum = f1 + f2;
        resg = WG[j].mul_add(fsum, resg);
        resk = WGK[j].mul_add(fsum, resk);
        resabs = WGK[j].mul_add(f1.abs() + f2.abs(), resabs);
    }
    let reskh = resk * 0.5;
    let mut resasc = WGK[7] * (fc - reskh).abs();
    for j in 0..7 {
        resasc = WGK[j].mul_add((fv1[j] - reskh).abs() + (fv2[j] - reskh).abs(), resasc);
    }
    let result = resk * hlgth;
    resabs *= hlgth;
    resasc *= hlgth;
    let mut abserr = ((resk - resg) * hlgth).abs();
    if resasc != 0.0 && abserr != 0.0 {
        abserr = resasc * (1.0f64).min((200.0 * abserr / resasc).powf(1.5));
    }
    if resabs > UFLOW / (50.0 * EPMACH) {
        abserr = (EPMACH * 50.0 * resabs).max(abserr);
    }
    (result, abserr, resabs, resasc)
}

/// `dqelg(n, epstab, result, abserr, res3la, nres)` — Wynn's ε-algorithm.
fn dqelg(
    n: &mut usize,
    epstab: &mut [f64; 52],
    result: &mut f64,
    abserr: &mut f64,
    res3la: &mut [f64; 3],
    nres: &mut usize,
) {
    let starting_n = *n;
    *nres += 1;
    *abserr = OFLOW;
    *result = epstab[*n];
    if *n < 2 {
        *abserr = abserr.max(5.0 * EPMACH * result.abs());
        return;
    }
    let limexp = 49usize;
    epstab[*n + 2] = epstab[*n];
    let newelm = *n / 2;
    epstab[*n] = OFLOW;
    let mut k1 = *n;
    for i in 0..newelm {
        let k2 = k1 - 1;
        let k3 = k1 - 2;
        let mut res = epstab[k1 + 2];
        let e0 = epstab[k3];
        let e1 = epstab[k2];
        let e2 = res;
        let e1abs = e1.abs();
        let delta2 = e2 - e1;
        let err2 = delta2.abs();
        let tol2 = e2.abs().max(e1abs) * EPMACH;
        let delta3 = e1 - e0;
        let err3 = delta3.abs();
        let tol3 = e1abs.max(e0.abs()) * EPMACH;
        if !(err2 > tol2 || err3 > tol3) {
            *result = res;
            *abserr = err2 + err3;
            *abserr = abserr.max(5.0 * EPMACH * result.abs());
            return;
        }
        let e3 = epstab[k1];
        epstab[k1] = e1;
        let delta1 = e1 - e3;
        let err1 = delta1.abs();
        let tol1 = e1abs.max(e3.abs()) * EPMACH;
        if err1 <= tol1 || err2 <= tol2 || err3 <= tol3 {
            *n = i + i;
            break;
        }
        let ss = 1.0 / delta1 + 1.0 / delta2 - 1.0 / delta3;
        let epsinf = (ss * e1).abs();
        if !(epsinf > 1e-4) {
            *n = i + i;
            break;
        }
        res = e1 + 1.0 / ss;
        epstab[k1] = res;
        k1 -= 2;
        let error = err2 + (res - e2).abs() + err3;
        if !(error > *abserr) {
            *abserr = error;
            *result = res;
        }
    }
    if *n == limexp {
        *n = 2 * (limexp / 2);
    }
    let n2_rem = starting_n % 2;
    for i in 0..=newelm {
        epstab[2 * i + n2_rem] = epstab[2 * i + 2 + n2_rem];
    }
    if *n != starting_n {
        // QUADPACK's `indx = starting_n - n` walk, kept as written
        #[allow(clippy::explicit_counter_loop)]
        let mut indx = starting_n - *n;
        #[allow(clippy::explicit_counter_loop)]
        for i in 0..=*n {
            epstab[i] = epstab[indx];
            indx += 1;
        }
    }
    if *nres < 4 {
        res3la[*nres - 1] = *result;
        *abserr = OFLOW;
    } else {
        *abserr =
            (*result - res3la[2]).abs() + (*result - res3la[1]).abs() + (*result - res3la[0]).abs();
        res3la[0] = res3la[1];
        res3la[1] = res3la[2];
        res3la[2] = *result;
    }
    *abserr = abserr.max(5.0 * EPMACH * result.abs());
}

/// `dqpsrt(limit, last, maxerr, ermax, elist, iord, nrmax)`.
fn dqpsrt(
    limit: usize,
    last: usize,
    maxerr: &mut usize,
    ermax: &mut f64,
    elist: &[f64],
    iord: &mut [usize],
    nrmax: &mut usize,
) {
    if last <= 2 {
        iord[0] = 0;
        iord[1] = 1;
        *maxerr = iord[*nrmax];
        *ermax = elist[*maxerr];
        return;
    }
    let errmax = elist[*maxerr];
    while *nrmax > 0 && errmax > elist[iord[*nrmax - 1]] {
        iord[*nrmax] = iord[*nrmax - 1];
        *nrmax -= 1;
    }
    let jupbn = if last > limit / 2 + 2 {
        limit - last + 2
    } else {
        last - 1
    };
    let errmin = elist[last - 1];
    let jbnd = jupbn as isize - 1;
    let ibeg = *nrmax as isize + 1;
    let mut found: Option<usize> = None;
    if ibeg <= jbnd {
        let mut i = ibeg;
        while i <= jbnd {
            if errmax >= elist[iord[i as usize]] {
                found = Some(i as usize);
                break;
            }
            iord[i as usize - 1] = iord[i as usize];
            i += 1;
        }
    }
    let Some(i) = found else {
        iord[jbnd as usize] = *maxerr;
        iord[jupbn] = last - 1;
        *maxerr = iord[*nrmax];
        *ermax = elist[*maxerr];
        return;
    };
    // LINE60
    iord[i - 1] = *maxerr;
    let mut k = jbnd;
    let mut j = i as isize;
    while j <= jbnd {
        if errmin < elist[iord[k as usize]] {
            break;
        }
        iord[(k + 1) as usize] = iord[k as usize];
        k -= 1;
        j += 1;
    }
    if errmin < elist[iord[k as usize]] {
        iord[(k + 1) as usize] = last - 1;
    } else {
        iord[i] = last - 1;
    }
    *maxerr = iord[*nrmax];
    *ermax = elist[*maxerr];
}

/// `dqagie` result: `(result, abserr, neval, ier, last)`.
pub type QagieResult = (f64, f64, i32, i32, usize);

/// `scipy.integrate.quad(f, a, b)` for an infinite range — `_quadpack._qagie(f,
/// bound, inf, epsabs, epsrel, limit)`: `inf = 1` for `(bound, ∞)`, `-1` for
/// `(-∞, bound)`, `2` for `(-∞, ∞)`.
pub fn dqagie(
    f: &mut dyn FnMut(f64) -> f64,
    bound: f64,
    inf: i32,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
) -> QagieResult {
    let mut alist = vec![0.0f64; limit + 1];
    let mut blist = vec![0.0f64; limit + 1];
    let mut rlist = vec![0.0f64; limit + 1];
    let mut elist = vec![0.0f64; limit + 1];
    let mut iord = vec![0usize; limit + 1];
    let mut ier = 0i32;
    let mut last: usize;
    let mut small = 0.0f64;
    let mut correc = 0.0f64;
    alist[0] = 0.0;
    blist[0] = 1.0;
    let mut erlarg = 0.0f64;
    let mut ertest = 0.0f64;
    if epsabs <= 0.0 && epsrel < (50.0 * EPMACH).max(0.5e-28) {
        // `*ier = 6; return;` — before LINE130, so not decremented
        return (0.0, 0.0, 0, 6, 0);
    }
    let boun = if inf == 2 { 0.0 } else { bound };
    let (mut result, mut abserr, defabs, resabs0) = dqk15i(f, boun, inf, 0.0, 1.0);
    last = 1;
    rlist[0] = result;
    elist[0] = abserr;
    iord[0] = 0;
    let dres = result.abs();
    let mut errbnd = epsabs.max(epsrel * dres);
    if abserr <= 100.0 * EPMACH * defabs && abserr > errbnd {
        ier = 2;
    }
    if limit == 1 {
        ier = 1;
    }
    let finish = |result: f64, abserr: f64, last: usize, ier: i32| -> QagieResult {
        let mut neval = 30 * last as i32 - 15;
        if inf == 2 {
            neval *= 2;
        }
        (
            result,
            abserr,
            neval,
            if ier > 2 { ier - 1 } else { ier },
            last,
        )
    };
    if ier != 0 || (abserr <= errbnd && abserr != resabs0) || abserr == 0.0 {
        return finish(result, abserr, last, ier);
    }
    let mut rlist2 = [0.0f64; 52];
    let mut res3la = [0.0f64; 3];
    rlist2[0] = result;
    let mut errmax = abserr;
    let mut maxerr = 0usize;
    let mut area = result;
    let mut errsum = abserr;
    abserr = OFLOW;
    let mut nrmax = 0usize;
    let mut nres = 0usize;
    let mut ktmin = 0;
    let mut numrl2 = 1usize;
    let mut extrap = false;
    let mut noext = false;
    let mut ierror = 0;
    let (mut iroff1, mut iroff2, mut iroff3) = (0, 0, 0);
    let ksgn = if dres > (1.0 - 50.0 * EPMACH) * defabs {
        1
    } else {
        -1
    };
    let mut l = 1usize;
    // `goto` targets after the loop
    enum Exit {
        Line115,
        Fallthrough,
    }
    let mut exit = Exit::Fallthrough;
    while l < limit {
        last = l + 1;
        let a1 = alist[maxerr];
        let b1 = 0.5 * (alist[maxerr] + blist[maxerr]);
        let a2 = b1;
        let b2 = blist[maxerr];
        let erlast = errmax;
        let (area1, error1, _, defab1) = dqk15i(f, boun, inf, a1, b1);
        let (area2, error2, _, defab2) = dqk15i(f, boun, inf, a2, b2);
        let area12 = area1 + area2;
        let error12 = error1 + error2;
        errsum = errsum + error12 - errmax;
        area = area + area12 - rlist[maxerr];
        if defab1 != error1 && defab2 != error2 {
            if !((rlist[maxerr] - area12).abs() > 1.0e-5 * area12.abs() || error12 < 0.99 * errmax)
            {
                if extrap {
                    iroff2 += 1;
                } else {
                    iroff1 += 1;
                }
            }
            if l > 9 && error12 > errmax {
                iroff3 += 1;
            }
        }
        rlist[maxerr] = area1;
        rlist[l] = area2;
        errbnd = epsabs.max(epsrel * area.abs());
        if iroff1 + iroff2 >= 10 || iroff3 >= 20 {
            ier = 2;
        }
        if iroff2 >= 5 {
            ierror = 3;
        }
        if last == limit {
            ier = 1;
        }
        // `(1.0 + 100.0*epmach)*(fabs(a2) + 1000.0*uflow)`: the inner sum fuses
        if a1.abs().max(b2.abs()) <= (1.0 + 100.0 * EPMACH) * 1000.0f64.mul_add(UFLOW, a2.abs()) {
            ier = 4;
        }
        if !(error2 > error1) {
            alist[l] = a2;
            blist[maxerr] = b1;
            blist[l] = b2;
            elist[maxerr] = error1;
            elist[l] = error2;
        } else {
            alist[maxerr] = a2;
            alist[l] = a1;
            blist[l] = b1;
            rlist[maxerr] = area2;
            rlist[l] = area1;
            elist[maxerr] = error2;
            elist[l] = error1;
        }
        dqpsrt(
            limit,
            last,
            &mut maxerr,
            &mut errmax,
            &elist,
            &mut iord,
            &mut nrmax,
        );
        if errsum <= errbnd {
            exit = Exit::Line115;
            break;
        }
        if ier != 0 {
            break;
        }
        if l == 1 {
            // LINE80
            small = 0.375;
            erlarg = errsum;
            ertest = errbnd;
            rlist2[1] = area;
            l += 1;
            continue;
        }
        if noext {
            l += 1;
            continue;
        }
        erlarg -= erlast;
        if (b1 - a1).abs() > small {
            erlarg += error12;
        }
        let mut go_line90 = false;
        if !extrap {
            if blist[maxerr] - alist[maxerr] > small {
                l += 1;
                continue;
            }
            extrap = true;
            nrmax = 1;
        }
        if !(ierror == 3 || erlarg <= ertest) {
            let jupbnd = if last > 2 + limit / 2 {
                limit + 3 - last
            } else {
                last
            };
            let mut k = nrmax;
            while k < jupbnd {
                maxerr = iord[nrmax];
                errmax = elist[maxerr];
                if (blist[maxerr] - alist[maxerr]).abs() > small {
                    go_line90 = true;
                    break;
                }
                nrmax += 1;
                k += 1;
            }
        }
        if go_line90 {
            l += 1;
            continue;
        }
        // LINE60
        numrl2 += 1;
        rlist2[numrl2] = area;
        let (mut reseps, mut abseps) = (0.0, 0.0);
        dqelg(
            &mut numrl2,
            &mut rlist2,
            &mut reseps,
            &mut abseps,
            &mut res3la,
            &mut nres,
        );
        ktmin += 1;
        if ktmin > 5 && abserr < 1.0e-3 * errsum {
            ier = 5;
        }
        if abseps < abserr {
            ktmin = 0;
            abserr = abseps;
            result = reseps;
            correc = erlarg;
            ertest = epsabs.max(epsrel * reseps.abs());
            if abserr <= ertest {
                break;
            }
        }
        // LINE70
        if numrl2 == 0 {
            noext = true;
        }
        if ier == 5 {
            break;
        }
        maxerr = iord[0];
        errmax = elist[maxerr];
        nrmax = 0;
        extrap = false;
        small *= 0.5;
        erlarg = errsum;
        l += 1;
    }
    let line115 = |l: usize, rlist: &[f64]| -> f64 {
        let mut r = 0.0f64;
        for k in 0..=l.min(rlist.len() - 1) {
            r += rlist[k];
        }
        r
    };
    if let Exit::Line115 = exit {
        return finish(line115(l, &rlist), errsum, last, ier);
    }
    if abserr == OFLOW {
        return finish(line115(l, &rlist), errsum, last, ier);
    }
    let goto_110 = |result: f64, abserr: f64, ier: i32| -> QagieResult {
        let mut ier = ier;
        if ksgn == -1 && result.abs().max(area.abs()) <= defabs * 0.01 {
            return finish(result, abserr, last, ier);
        }
        if 0.01 > result / area || result / area > 100.0 || errsum > area.abs() {
            ier = 6;
        }
        finish(result, abserr, last, ier)
    };
    if ier + ierror == 0 {
        return goto_110(result, abserr, ier);
    }
    if ierror == 3 {
        abserr += correc;
    }
    if ier == 0 {
        ier = 3;
    }
    if result != 0.0 && area != 0.0 {
        // LINE105
        if abserr / result.abs() > errsum / area.abs() {
            return finish(line115(l, &rlist), errsum, last, ier);
        }
        return goto_110(result, abserr, ier);
    }
    if abserr > errsum {
        return finish(line115(l, &rlist), errsum, last, ier);
    }
    if area == 0.0 {
        return finish(result, abserr, last, ier);
    }
    goto_110(result, abserr, ier)
}

unsafe extern "C" {
    /// The C library's `erfc` — what SciPy's Cython integrands call (`libc.math`).
    /// Not the `libm` crate: its fdlibm `erfc` differs from macOS libm's on ~31 % of
    /// arguments (measured against `math.erfc`).
    #[link_name = "erfc"]
    fn c_erfc(x: f64) -> f64;
}

/// The C library `erfc(x)`.
pub fn libc_erfc(x: f64) -> f64 {
    // SAFETY: a pure libm function of one double.
    unsafe { c_erfc(x) }
}

/// `scipy.stats.studentized_range.cdf(q, k, df=inf)`:
/// `∫ k φ(z) (Φ(z+q) − Φ(z))^(k−1) dz` over the real line by `dqagie`
/// (`epsabs=1e-11, epsrel=1e-12, limit=50`), clipped to `[0, 1]`. φ and Φ are
/// SciPy's Cython helpers on the C library's `exp` / `erfc`.
pub fn studentized_range_cdf_inf(q: f64, k: f64) -> f64 {
    let phi = |z: f64| 0.3989422804014327 * (-0.5 * z * z).exp();
    let big_phi = |z: f64| 0.5 * libc_erfc(-z * 0.7071067811865475);
    let mut f = |z: f64| k * phi(z) * (big_phi(z + q) - big_phi(z)).powf(k - 1.0);
    let (r, ..) = dqagie(&mut f, 0.0, 2, 1e-11, 1e-12, 50);
    r.clamp(0.0, 1.0)
}

/// `scipy.stats.studentized_range.sf(q, k, df=inf)` = `1 - cdf` (no dedicated `_sf`).
pub fn studentized_range_sf_inf(q: f64, k: f64) -> f64 {
    1.0 - studentized_range_cdf_inf(q, k)
}
