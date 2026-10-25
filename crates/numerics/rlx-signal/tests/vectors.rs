// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bit-exact parity against SciPy 1.17.1 / NumPy 2.3.5 (`tests/vectors/scipy_signal.txt`,
//! from `tools/gen_vectors.py`). Tolerance is **zero**: every value is compared as its
//! IEEE-754 bit pattern (long arrays through an FNV-1a hash of those bits).

use std::collections::{BTreeMap, HashMap};

use rlx_signal::complex::C;
use rlx_signal::fir::{Window, firwin, get_window};
use rlx_signal::iir::*;

fn hx(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).unwrap())
}
fn arr(s: &str) -> Vec<f64> {
    if s.is_empty() {
        vec![]
    } else {
        s.split(',').map(hx).collect()
    }
}
fn carr(s: &str) -> Vec<C> {
    if s.is_empty() {
        return vec![];
    }
    s.split(',')
        .map(|c| {
            let (r, i) = c.split_once(':').unwrap();
            C::new(hx(r), hx(i))
        })
        .collect()
}
fn fnv(v: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    }
    h
}
/// `Err` names the first differing element.
fn same(got: &[f64], want: &str) -> Result<(), String> {
    if let Some(rest) = want.strip_prefix("fnv:") {
        let (hh, n) = rest.split_once(':').unwrap();
        if got.len() != n.parse::<usize>().unwrap() {
            return Err(format!("length {} != {n}", got.len()));
        }
        return if fnv(got) == u64::from_str_radix(hh, 16).unwrap() {
            Ok(())
        } else {
            Err("hash differs".into())
        };
    }
    let w = arr(want);
    if got.len() != w.len() {
        return Err(format!("length {} != {}", got.len(), w.len()));
    }
    for (i, (g, w)) in got.iter().zip(&w).enumerate() {
        if g.to_bits() != w.to_bits() {
            return Err(format!(
                "[{i}] {g:e} != {w:e} ({} ulp)",
                (i128::from(g.to_bits() as i64) - i128::from(w.to_bits() as i64)).abs()
            ));
        }
    }
    Ok(())
}
fn same_c(got: &[C], want: &str) -> Result<(), String> {
    let w = carr(want);
    if got.len() != w.len() {
        return Err(format!("length {} != {}", got.len(), w.len()));
    }
    for (i, (g, w)) in got.iter().zip(&w).enumerate() {
        if g.re.to_bits() != w.re.to_bits() || g.im.to_bits() != w.im.to_bits() {
            return Err(format!("[{i}] {g:?} != {w:?}"));
        }
    }
    Ok(())
}
fn lcg(n: usize, seed: u64) -> Vec<f64> {
    (0..n as u64)
        .map(|k| {
            ((k.wrapping_mul(2654435761).wrapping_add(seed)) % (1u64 << 32)) as f64 / 4294967296.0
                - 0.5
        })
        .collect()
}
/// `np.cumsum(lcg(n, s)) * 0.1 + np.sin(0.05 * t) + lcg(n, s + 1)`.
fn sig(n: usize, seed: u64) -> Vec<f64> {
    let (a, b) = (lcg(n, seed), lcg(n, seed + 1));
    let mut c = 0.0;
    (0..n)
        .map(|i| {
            c += a[i];
            c * 0.1 + std::hint::black_box(0.05 * i as f64).sin() + b[i]
        })
        .collect()
}
fn sos_of(v: &[f64]) -> Vec<Sos> {
    v.chunks(6).map(|c| c.try_into().unwrap()).collect()
}
fn pad(s: &str) -> PadType {
    match s {
        "odd" => PadType::Odd,
        "even" => PadType::Even,
        "constant" => PadType::Constant,
        _ => PadType::None,
    }
}

#[test]
fn matches_scipy_signal_bit_for_bit() {
    let text = include_str!("vectors/scipy_signal.txt");
    let mut fails: Vec<String> = Vec::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for line in text.lines() {
        let mut parts = line.split(';');
        let op = parts.next().unwrap();
        let kv: HashMap<&str, &str> = parts.map(|p| p.split_once('=').unwrap()).collect();
        let g = |k: &str| kv[k];
        let res: Result<(), String> = (|| match op {
            "buttap" => {
                let z = buttap(g("n").parse().unwrap());
                same_c(&z.p.to_complex(), g("p"))?;
                same(&[z.k], g("k"))
            }
            "butter" => {
                let bt = match g("btype") {
                    "lowpass" => BType::Lowpass,
                    "highpass" => BType::Highpass,
                    _ => BType::Bandpass,
                };
                let zpk = butter_zpk(g("n").parse().unwrap(), &arr(g("wn")), bt)?;
                same_c(&zpk.z.to_complex(), g("z")).map_err(|e| format!("z {e}"))?;
                same_c(&zpk.p.to_complex(), g("p")).map_err(|e| format!("p {e}"))?;
                same(&[zpk.k], g("k")).map_err(|e| format!("k {e}"))?;
                let (b, a) = zpk2tf(&zpk);
                same(&b, g("b")).map_err(|e| format!("b {e}"))?;
                same(&a, g("a")).map_err(|e| format!("a {e}"))?;
                let sos = zpk2sos(&zpk)?;
                same(&sos.concat(), g("sos")).map_err(|e| format!("sos {e}"))
            }
            "lfilter_zi" => same(&lfilter_zi(&arr(g("b")), &arr(g("a")))?, g("zi")),
            "lfilter" => {
                let x = sig(400, g("seed").parse().unwrap());
                let (y, zf) = lfilter(&arr(g("b")), &arr(g("a")), &x, Some(&arr(g("zi"))));
                same(&y, g("y")).map_err(|e| format!("y {e}"))?;
                same(&zf, g("zf")).map_err(|e| format!("zf {e}"))
            }
            "sosfilt_zi" => same(&sosfilt_zi(&sos_of(&arr(g("sos"))))?.concat(), g("zi")),
            "sosfilt" => {
                let x = sig(400, g("seed").parse().unwrap());
                let zi: Vec<[f64; 2]> = arr(g("zi")).chunks(2).map(|c| [c[0], c[1]]).collect();
                let (y, zf) = sosfilt(&sos_of(&arr(g("sos"))), &x, Some(&zi));
                same(&y, g("y")).map_err(|e| format!("y {e}"))?;
                same(&zf.concat(), g("zf")).map_err(|e| format!("zf {e}"))
            }
            "filtfilt" | "sosfiltfilt" => {
                let x = sig(400, g("seed").parse().unwrap());
                let pl = g("padlen").parse::<usize>().ok();
                let pt = pad(g("padtype"));
                let got = if op == "filtfilt" {
                    filtfilt(&arr(g("b")), &arr(g("a")), &x, pt, pl)
                } else {
                    sosfiltfilt(&sos_of(&arr(g("sos"))), &x, pt, pl)
                };
                match (got, kv.get("err")) {
                    (Err(_), Some(_)) => Ok(()),
                    (Ok(_), Some(_)) => Err("SciPy raised, port did not".into()),
                    (Err(e), None) => Err(e),
                    (Ok(y), None) => same(&y, g("y")),
                }
            }
            "window" => same(
                &get_window(Window::parse(g("name")).unwrap(), g("m").parse().unwrap()),
                g("w"),
            ),
            "firwin" => {
                let fs = if g("fs") == "none" {
                    None
                } else {
                    Some(hx(g("fs")))
                };
                let h = firwin(
                    g("numtaps").parse().unwrap(),
                    &arr(g("cutoff")),
                    Window::parse(g("window")).unwrap(),
                    g("pass_zero") == "1",
                    fs,
                )?;
                same(&h, g("h"))
            }
            o => Err(format!("unknown op {o}")),
        })();
        *counts.entry(op.to_string()).or_default() += 1;
        if let Err(e) = res {
            fails.push(format!(
                "{op} {}: {e}",
                line.chars().take(90).collect::<String>()
            ));
        }
    }
    eprintln!("cases: {counts:?}");
    let total: usize = counts.values().sum();
    assert!(
        fails.is_empty(),
        "{} of {total} differ:\n{}",
        fails.len(),
        fails
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(total > 2300, "only {total} cases ran");
}
