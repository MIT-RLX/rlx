// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
//! Expert paging against a **real published checkpoint**.
//!
//! The synthetic tests in `paging.rs` prove the cache behaves; they cannot prove
//! the bank geometry is right, because they invent the geometry they then check.
//! The failure that matters is off-by-a-dimension: GGUF stores shapes
//! innermost-first, so reading `[4096, 2048, 8]` as 4096 experts instead of 8
//! produces a `bytes_per_expert` that is wrong by 512x, every read lands
//! mid-block, and the result is not an error — it is a bank of plausible noise
//! that dequantizes to garbage weights.
//!
//! So this compares paged bytes against the same bytes read the ordinary way.
//!
//! Needs the 397 MB subset from `rlx-glm5next`'s `scripts/glm5next_subset.py`
//! (`just fetch-glm5next-subset`); skips cleanly when it is absent, since CI has
//! no weights.

#![cfg(feature = "gguf")]

use rlx_distributed::{ExpertPager, GGUF_MOE_BANKS, register_gguf_expert_banks};

const SUBSET: &str = "/tmp/rlx-weights/glm5next/GLM-5.3-Flash-blk0-blk3-e8.gguf";

fn subset() -> Option<&'static str> {
    std::path::Path::new(SUBSET).is_file().then_some(SUBSET)
}

/// Every paged expert must be byte-identical to that expert's slice of the
/// bank as read whole. This is the geometry check: expert count, slice size,
/// base offset and stride all have to be right simultaneously for it to pass.
#[test]
fn paged_experts_match_the_whole_bank() {
    let Some(path) = subset() else {
        eprintln!("skipping: {SUBSET} not present (run `just fetch-glm5next-subset`)");
        return;
    };

    let mut pager = ExpertPager::new(64 << 20);
    let n = register_gguf_expert_banks(&mut pager, path, &GGUF_MOE_BANKS).expect("register banks");
    assert!(
        n > 0,
        "no routed-expert banks found in the subset — the suffixes or the \
         blk. prefix parsing is wrong"
    );

    let g = rlx_gguf::GgufFile::from_path(path).expect("open gguf");
    let mut checked = 0usize;
    for t in g.tensors.values() {
        let Some(rest) = t.name.strip_prefix("blk.") else {
            continue;
        };
        let Some((idx, tail)) = rest.split_once('.') else {
            continue;
        };
        if !GGUF_MOE_BANKS.contains(&tail) {
            continue;
        }
        let layer: usize = idx.parse().unwrap();
        let bank = tail.strip_suffix(".weight").unwrap();

        let whole = g.tensor_bytes(t).expect("bank bytes");
        // Innermost-first: the expert count is the LAST dimension.
        let num_experts = *t.shape.last().unwrap();
        assert!(
            num_experts > 1,
            "{}: shape {:?} gives {num_experts} experts — if the expert axis \
             were read from the wrong end this test could not tell",
            t.name,
            t.shape
        );
        let per = whole.len() / num_experts;

        for id in [0, num_experts / 2, num_experts - 1] {
            let paged = pager.expert(layer, bank, id).expect("page expert");
            assert_eq!(
                paged.len(),
                per,
                "{}: paged expert {id} is {} bytes, the bank cuts into {per}",
                t.name,
                paged.len()
            );
            assert_eq!(
                &paged[..],
                &whole[id * per..(id + 1) * per],
                "{}: paged expert {id} differs from its slice of the bank — the \
                 offset or stride is wrong, and a misaligned quant block reads \
                 as noise rather than as an error",
                t.name
            );
            checked += 1;
        }
    }
    assert!(checked >= 3, "nothing was actually compared");
    eprintln!("paging: verified {checked} experts across {n} banks");
}

/// A gather is the concatenation of the fired experts, in fired order — the
/// operand a `GroupedMatMul` over a `top_k`-expert bank needs.
#[test]
fn a_gather_is_a_top_k_bank() {
    let Some(path) = subset() else {
        eprintln!("skipping: {SUBSET} not present");
        return;
    };
    let mut pager = ExpertPager::new(64 << 20);
    register_gguf_expert_banks(&mut pager, path, &GGUF_MOE_BANKS).expect("register");

    let g = rlx_gguf::GgufFile::from_path(path).expect("open gguf");
    let (name, t) = g
        .tensors
        .iter()
        .find(|(n, _)| n.ends_with("ffn_gate_exps.weight"))
        .expect("the subset must contain a gate bank");
    let layer: usize = name
        .strip_prefix("blk.")
        .unwrap()
        .split('.')
        .next()
        .unwrap()
        .parse()
        .unwrap();

    let whole = g.tensor_bytes(t).unwrap();
    let num_experts = *t.shape.last().unwrap();
    let per = whole.len() / num_experts;

    // A plausible routing draw: unsorted, with a repeat.
    let fired: Vec<usize> = vec![num_experts - 1, 1, 0, 1];
    let gathered = pager.gather(layer, "ffn_gate_exps", &fired).unwrap();
    assert_eq!(gathered.len(), fired.len() * per);
    for (slot, &id) in fired.iter().enumerate() {
        assert_eq!(
            &gathered[slot * per..(slot + 1) * per],
            &whole[id * per..(id + 1) * per],
            "slot {slot} should hold expert {id}; a gather must preserve the \
             caller's order or every token is paired with the wrong gate"
        );
    }

    // What paging buys: this layer's routed bank, versus what a token reads.
    let bank_bytes = whole.len();
    assert!(
        gathered.len() < bank_bytes,
        "a {}-of-{num_experts} gather ({} bytes) is not smaller than the bank \
         ({bank_bytes} bytes)",
        fired.len(),
        gathered.len()
    );
    eprintln!(
        "paging: {} of {num_experts} experts = {:.1} MB of a {:.1} MB bank",
        fired.len(),
        gathered.len() as f64 / 1e6,
        bank_bytes as f64 / 1e6
    );
}

/// Registration reads the header only. A pager for a 93 GB checkpoint must not
/// need 93 GB — that is the whole proposition.
#[test]
fn registering_does_not_load_the_banks() {
    let Some(path) = subset() else {
        eprintln!("skipping: {SUBSET} not present");
        return;
    };
    let mut pager = ExpertPager::new(1 << 20);
    register_gguf_expert_banks(&mut pager, path, &GGUF_MOE_BANKS).expect("register");
    assert_eq!(
        pager.resident_bytes(),
        0,
        "registration pulled bytes into the cache"
    );
    assert!(
        pager.total_bank_bytes() > 0,
        "the banks report no bytes at all"
    );
    assert_eq!(pager.stats().bytes_read, 0, "registration read tensor data");
}
