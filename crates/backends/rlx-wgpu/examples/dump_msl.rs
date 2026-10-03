// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Emit the Metal Shading Language that naga generates for one of the shipped
//! WGSL kernels, so a hot loop can be read the way the driver sees it.
//!
//! WGSL has no way to say "keep this in registers". A fixed-size local array
//! indexed by a loop variable *may* become registers or *may* become a stack
//! array in `thread` address space, and the difference is an order of magnitude
//! in a GEMM inner loop. That decision is not visible in the WGSL; it is
//! visible here.
//!
//! ```text
//! cargo run -p rlx-wgpu --example dump_msl -- matmul_wide
//! ```

fn main() {
    let which = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "matmul_wide".into());
    let src = match which.as_str() {
        "matmul" => rlx_wgpu::kernels::MATMUL_WGSL,
        "matmul_wide" => rlx_wgpu::kernels::MATMUL_WIDE_WGSL,
        "matmul_wide_nv" => rlx_wgpu::kernels::MATMUL_WIDE_NV_WGSL,
        other => {
            eprintln!("unknown kernel {other}; try matmul | matmul_wide | matmul_wide_nv");
            std::process::exit(2);
        }
    };
    let module = naga::front::wgsl::parse_str(src).expect("wgsl parse");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("wgsl validate");
    let opts = naga::back::msl::Options::default();
    let pipeline = naga::back::msl::PipelineOptions::default();
    let (msl, _) =
        naga::back::msl::write_string(&module, &info, &opts, &pipeline).expect("msl out");
    println!("{msl}");
}
