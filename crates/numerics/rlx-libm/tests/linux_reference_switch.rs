//! `platform::use_linux_reference` must reach `reference::*` on Apple with the
//! `libm` feature: before it, the system libm; after it, the glibc ports — the
//! arm wasm always takes. Its own test binary, because the switch is
//! process-wide and one-way.
//!
//! (A browser/native comparison found the native `portable` build on macOS
//! computing Apple's `logf` on 55 of 12288 probe inputs: `reference` chose its
//! arm at compile time only, so the switch never reached it.)

#[cfg(all(target_vendor = "apple", feature = "libm"))]
#[test]
fn use_linux_reference_switches_reference_to_the_ports() {
    use rlx_libm::{platform, reference};
    // inputs where Apple's logf / expf / sinf disagree with glibc's
    let (lx, ex, sx): (Vec<f32>, Vec<f32>, Vec<f32>) = {
        let mut lx = Vec::new();
        let mut ex = Vec::new();
        let mut sx = Vec::new();
        let mut i = 0u32;
        while (lx.len() < 4 || ex.len() < 4 || sx.len() < 4) && i < 4_000_000 {
            let x = f32::from_bits(0x3c00_0000 + i * 97);
            if lx.len() < 4 && x.ln() != rlx_libm::logf(x) {
                lx.push(x);
            }
            let e = x * 10.0 - 20.0;
            if ex.len() < 4 && e.exp() != rlx_libm::expf(e) {
                ex.push(e);
            }
            if sx.len() < 4 && x.sin() != rlx_libm::sincosf(x).0 {
                sx.push(x);
            }
            i += 1;
        }
        (lx, ex, sx)
    };
    assert!(
        !lx.is_empty() && !ex.is_empty() && !sx.is_empty(),
        "no disagreeing inputs found: the test would pin nothing"
    );

    assert!(platform::apple_reference());
    for &x in &lx {
        assert_eq!(
            reference::logf(x).to_bits(),
            x.ln().to_bits(),
            "logf({x:e}) before the switch: Apple's"
        );
    }
    for &x in &ex {
        assert_eq!(
            reference::expf(x).to_bits(),
            x.exp().to_bits(),
            "expf({x:e}) before the switch: Apple's"
        );
    }

    platform::use_linux_reference();
    assert!(!platform::apple_reference());
    for &x in &lx {
        assert_eq!(
            reference::logf(x).to_bits(),
            rlx_libm::logf(x).to_bits(),
            "logf({x:e}) after the switch: glibc's"
        );
    }
    for &x in &ex {
        assert_eq!(
            reference::expf(x).to_bits(),
            rlx_libm::expf(x).to_bits(),
            "expf({x:e}) after the switch: glibc's"
        );
    }
    for &x in &sx {
        assert_eq!(
            reference::sincosf(x),
            rlx_libm::sincosf(x),
            "sincosf({x:e}) after the switch: glibc's"
        );
        assert_eq!(
            reference::sincosf_merged(x),
            rlx_libm::sincosf(x),
            "sincosf_merged({x:e}) after the switch"
        );
    }
}
