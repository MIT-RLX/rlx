//! `rlx_linux_ref::x87::F80` against x87 hardware (`tests/fixtures/x87_ops.txt` from
//! `tools/x87_ref.c` on x86-64 Linux): every 80-bit result and its store to double.

use rlx_linux_ref::x87::F80;

#[test]
fn matches_x87_hardware() {
    let text = include_str!("fixtures/x87_ops.txt");
    let (mut bad, mut n) = (vec![], 0);
    for line in text.lines() {
        let t: Vec<&str> = line.split(' ').collect();
        let h = |s: &str| f64::from_bits(u64::from_str_radix(s, 16).unwrap());
        let (a, b) = (F80::from_f64(h(t[1])), F80::from_f64(h(t[2])));
        let r = match t[0] {
            "add" => a.add(b),
            "sub" => a.sub(b),
            "mul" => a.mul(b),
            "div" => a.div(b),
            "sqrt" => a.sqrt(),
            "sumsq" => a.mul(a).add(b.mul(b)),
            "hyp" => a.mul(a).add(b.mul(b)).sqrt(),
            "divmuladd" => a.div(b).add(a.mul(b)),
            op => panic!("{op}"),
        };
        let (se, m) = r.to_bits80();
        let want_se = u16::from_str_radix(&t[3][..4], 16).unwrap();
        let want_m = u64::from_str_radix(&t[3][4..], 16).unwrap();
        n += 1;
        if (se, m) != (want_se, want_m)
            || r.to_f64().to_bits() != u64::from_str_radix(t[4], 16).unwrap()
        {
            bad.push(line.to_string());
        }
    }
    assert!(n > 10000);
    assert!(
        bad.is_empty(),
        "{} of {n} differ: {:#?}",
        bad.len(),
        &bad[..bad.len().min(8)]
    );
}
