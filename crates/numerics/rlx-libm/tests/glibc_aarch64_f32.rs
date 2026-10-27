//! `logf` / `sincosf` against glibc on Linux aarch64 (the torch reference), on
//! every platform: `tests/vectors/glibc_aarch64_f32.txt` holds FNV-1a hashes of
//! glibc's output bits over strided ranges of input bit patterns
//! (`tools/glibc_ref_f32.py`). Zero tolerance.

fn fnv(h: &mut u64, v: f32) {
    for b in v.to_bits().to_le_bytes() {
        *h = (*h ^ u64::from(b)).wrapping_mul(0x100000001b3);
    }
}

#[test]
fn logf_and_sincosf_match_glibc_aarch64() {
    let text = include_str!("vectors/glibc_aarch64_f32.txt");
    let mut checked = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let t: Vec<&str> = line.split(' ').collect();
        let p = |i: usize| t[i].parse::<u64>().unwrap();
        let (a, b, s, count) = (p(1), p(2), p(3), p(4) as usize);
        let ins: Vec<u32> = (a..b).step_by(s as usize).map(|v| v as u32).collect();
        assert_eq!(ins.len(), count, "{line}");
        let (mut h0, mut h1) = (0xcbf29ce484222325u64, 0xcbf29ce484222325u64);
        for &i in &ins {
            let x = f32::from_bits(i);
            if t[0] == "logf" {
                fnv(&mut h0, rlx_libm::logf(x));
            } else {
                let (sv, cv) = rlx_libm::sincosf(x);
                fnv(&mut h0, sv);
                fnv(&mut h1, cv);
            }
        }
        assert_eq!(format!("{h0:016x}"), t[5], "{} on {a}..{b} step {s}", t[0]);
        if t[0] == "sincosf" {
            assert_eq!(format!("{h1:016x}"), t[6], "cos on {a}..{b} step {s}");
        }
        checked += count;
    }
    assert!(checked > 3_000_000, "{checked}");
}
