// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Query-only: open /dev/accel and dump every live hwctx's firmware state
// (start_col / num_col / submit / complete / errors). Does NOT create a context,
// so it can inspect another process's hwctx (e.g. XRT's) while it runs.
//
//   cargo run -p rlx-xdna --features direct --example direct_query

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("direct_query is Linux-only");
}

#[cfg(target_os = "linux")]
fn main() {
    let npu = rlx_xdna::direct::Npu::open("").expect("open /dev/accel/accel0");
    match npu.hwctx_report() {
        Ok(r) => print!("live hwctx state:\n{r}"),
        Err(e) => eprintln!("hwctx query failed: {e}"),
    }
}
