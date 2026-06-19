// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Integration test for the pipeline_schedule! macro (plan #11).
//!
//! Lives outside the macro crate because proc macros can't be
//! invoked from within their own crate.

use rlx_runtime::pipeline_schedule;

// Linear chain: declaration order is the topo order.
pipeline_schedule! {
    name: LinearChain,
    stages: {
        a => [],
        b => [a],
        c => [b],
        d => [c],
    }
}

// Diamond: two parallel branches that converge. Two valid orders;
// our impl uses Kahn's with declaration-tie-break, so the result
// is deterministic.
pipeline_schedule! {
    name: Diamond,
    stages: {
        load     => [],
        branch_l => [load],
        branch_r => [load],
        join     => [branch_l, branch_r],
    }
}

// A realistic kernel-pipeline shape — what the BERT attention
// block would describe declaratively.
pipeline_schedule! {
    name: AttentionBlock,
    stages: {
        qkv_proj  => [],
        narrow_q  => [qkv_proj],
        narrow_k  => [qkv_proj],
        narrow_v  => [qkv_proj],
        attention => [narrow_q, narrow_k, narrow_v],
        out_proj  => [attention],
    }
}

#[test]
fn linear_chain_order_matches_declaration() {
    assert_eq!(LinearChain::ORDER, &["a", "b", "c", "d"]);
}

#[test]
fn diamond_order_respects_dependencies() {
    let order = Diamond::ORDER;
    let pos = |name: &str| order.iter().position(|s| *s == name).unwrap();
    // load before everything else.
    assert!(pos("load") < pos("branch_l"));
    assert!(pos("load") < pos("branch_r"));
    // Both branches before join.
    assert!(pos("branch_l") < pos("join"));
    assert!(pos("branch_r") < pos("join"));
}

#[test]
fn attention_block_order_is_topological() {
    let order = AttentionBlock::ORDER;
    let pos = |name: &str| order.iter().position(|s| *s == name).unwrap();
    assert!(pos("qkv_proj") < pos("narrow_q"));
    assert!(pos("qkv_proj") < pos("narrow_k"));
    assert!(pos("qkv_proj") < pos("narrow_v"));
    assert!(pos("narrow_q") < pos("attention"));
    assert!(pos("narrow_k") < pos("attention"));
    assert!(pos("narrow_v") < pos("attention"));
    assert!(pos("attention") < pos("out_proj"));
}

#[test]
fn deps_table_preserves_declarations() {
    let deps = AttentionBlock::DEPS;
    // Find each by name.
    let lookup = |name: &str| deps.iter().find(|(n, _)| *n == name).unwrap().1;
    assert_eq!(lookup("qkv_proj"), &[] as &[&str]);
    assert_eq!(lookup("narrow_q"), &["qkv_proj"]);
    assert_eq!(lookup("attention"), &["narrow_q", "narrow_k", "narrow_v"]);
}
