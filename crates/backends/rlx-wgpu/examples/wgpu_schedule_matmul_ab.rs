// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Does generating `matmul` from its schedule beat the hand-written WGSL —
//! and is the answer the same on Apple and NVIDIA?**
//!
//! The third arm of the experiment in `rlx-cuda/examples/schedule_matmul_ab.rs`
//! and `rlx-metal/examples/schedule_sgemm_ab.rs`. The treatment is the same
//! structural change: `stages`-deep rotation of the workgroup tiles, which
//! takes the K loop from **two** `workgroupBarrier()` calls to **one**.
//!
//! # Why wgpu is the arm that settles the question
//!
//! Metal said the rotation *loses* (0.91x at 2 stages, 0.83x at 3). Two
//! explanations fit that equally well:
//!
//! 1. the rotation costs more occupancy than the barrier saves — a property of
//!    the transformation, true on any GPU;
//! 2. Apple GPUs have unusually cheap barriers and tight threadgroup memory — a
//!    property of the vendor.
//!
//! One measurement cannot separate them. This one can: the *same WGSL* compiles
//! through naga to Metal on Apple silicon and to SPIR-V/Vulkan on NVIDIA, so
//! running it on both holds the source constant and varies only the GPU. Run it
//! on each and compare the columns.
//!
//! # Arms
//!
//! | arm | source | stages | barriers / K iter |
//! |---|---|---|---|
//! | `shipping` | `kernels/matmul.wgsl`, the real shader | 1 | 2 |
//! | `serial` | **emitted** from the schedule describing it | 1 | 2 |
//! | `pipe:N` | **emitted** from the pipelined schedule | N | 1 |
//!
//! `serial` is the control. It is the same physical schedule as the baseline,
//! so it must land at ~1.00x; if it does not, the emitter is contributing
//! something and no `pipe:N` number is attributable. On Metal this control is
//! what caught a first-touch ordering bias that had made every later arm look
//! 20% faster than it was.
//!
//! # Protocol
//!
//! * **Bit-exactness gates every arm** against the shipping shader's output.
//!   All arms accumulate in the same `kk` order over the same tile contents, so
//!   a differing bit is a bug, not a faster kernel.
//! * **Every arm is warmed before any is timed, and samples are round-robin**,
//!   so first-touch cost and thermal drift land on all arms alike.
//! * **Timing is wall clock around submit + `poll(Wait)`.** wgpu's timestamp
//!   queries need `Features::TIMESTAMP_QUERY`, which is not available on every
//!   adapter this is meant to run on; rather than have the protocol differ
//!   silently between machines, all arms get the same host-side timer and the
//!   dispatch is sized large enough that submit overhead is a small share. The
//!   `serial` control is what proves that share is not distorting the result.
//! * **Ratios only.** Absolute ms here are not comparable to the CUDA or Metal
//!   examples — different timing source, no L2 flush.
//!
//! # Naming
//!
//! `wgpu_`-prefixed rather than sharing `schedule_matmul_ab` with the CUDA
//! example. Cargo writes every package's examples into one
//! `target/release/examples/` directory, so two same-named examples in one
//! workspace overwrite each other's unsuffixed binary — a harness that ran
//! `./target/release/examples/schedule_matmul_ab` twice would have measured
//! whichever built last, under both labels.
//!
//! ```sh
//! cargo run --release -p rlx-wgpu --features schedule-codegen \
//!     --example wgpu_schedule_matmul_ab
//! cargo run --release -p rlx-wgpu --features schedule-codegen \
//!     --example wgpu_schedule_matmul_ab -- --stages 2,3,4
//! ```

#[cfg(not(feature = "schedule-codegen"))]
fn main() {
    eprintln!(
        "this example needs `--features schedule-codegen`:\n  \
         cargo run --release -p rlx-wgpu --features schedule-codegen \
         --example wgpu_schedule_matmul_ab"
    );
}

#[cfg(feature = "schedule-codegen")]
fn main() {
    use std::time::Instant;

    use rlx_wgpu::kernel_schedule_emit::{
        EMITTED_ENTRY, EmitFacts, WgslTile, emit_wgsl, matmul_pipelined_schedule, matmul_schedule,
    };
    use rlx_wgpu::kernels::{MATMUL_WGSL, MatmulParams};

    const WARMUP: usize = 5;
    const ITERS: usize = 30;
    const TILE: WgslTile = WgslTile::SHIPPING;

    /// Shapes spanning decode through prefill. Every one reaches the treatment:
    /// the staging is bounds-checked, so unlike the CUDA sweep there is no
    /// alignment gate to fall out of.
    const SHAPES: &[(&str, usize, usize, usize)] = &[
        ("decode, LM hidden", 1, 4096, 4096),
        ("decode, mlp up", 1, 2048, 8192),
        ("tiny batch", 4, 4096, 4096),
        ("small batch", 32, 4096, 4096),
        ("medium batch", 128, 4096, 4096),
        ("short prefill", 256, 2048, 2048),
        ("medium prefill", 512, 2048, 2048),
        ("prefill, wide hidden", 512, 4096, 4096),
        ("large prefill", 1024, 2048, 2048),
        ("large square", 2048, 2048, 2048),
        ("fat k, small mn", 192, 8192, 512),
        ("tall, short k (dW)", 4096, 512, 512),
    ];

    fn fill(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    let Some(dev) = rlx_wgpu::device::wgpu_device() else {
        eprintln!("no wgpu adapter");
        std::process::exit(1);
    };
    let (device, queue) = (&dev.device, &dev.queue);

    let args: Vec<String> = std::env::args().collect();
    let stages: Vec<usize> = args
        .iter()
        .position(|a| a == "--stages")
        .and_then(|i| args.get(i + 1))
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2, 3]);

    // `matmul.wgsl`'s binding layout: storage(rw) arena @0, uniform params @1.
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sched-ab bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sched-ab layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let build = |label: &str, wgsl: &str, entry: &str| -> wgpu::ComputePipeline {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };

    // Build every pipeline up front — a shader compile inside the timing loop
    // would land entirely on the first sample of whichever arm went first.
    let mut arms: Vec<(String, wgpu::ComputePipeline, Option<EmitFacts>)> = Vec::new();
    arms.push((
        "shipping".into(),
        build("shipping", MATMUL_WGSL, "matmul"),
        None,
    ));
    let mut add = |label: String, sched: &rlx_ir::kernel_schedule::KernelSchedule| {
        match emit_wgsl(sched, TILE, rlx_ir::kernel_schedule::Target::PORTABLE) {
            Ok((src, facts)) => arms.push((
                label.clone(),
                build(&label, &src, EMITTED_ENTRY),
                Some(facts),
            )),
            // Refused, never silently replaced by the baseline: an arm that
            // quietly ran the shipping shader would report 1.00x and read as
            // "no difference" rather than "did not run".
            Err(e) => eprintln!("arm `{label}` could not be emitted: {e} — EXCLUDED"),
        }
    };
    add("serial".into(), &matmul_schedule(TILE));
    for s in &stages {
        add(format!("pipe:{s}"), &matmul_pipelined_schedule(TILE, *s));
    }

    println!("schedule_matmul_ab (wgpu) — is the emitted WGSL faster than matmul.wgsl?\n");
    println!("  adapter    : {} via {:?}", dev.name, dev.backend);
    println!("  workload   : f32 GEMM via rlx's own tiled `matmul` shader");
    println!("  oracle     : the shipping shader's output, same shape, same kk order");
    println!("  tolerance  : bit-exact; any differing element disqualifies the arm");
    println!("  timing     : wall clock around submit+poll, median of {ITERS}, {WARMUP} warmup");
    println!("  references : rlx's own shipping shader only — not a vendor library\n");
    for (label, _, facts) in &arms {
        match facts {
            Some(f) => println!(
                "  arm {label:<9}: stages={} barriers/iter={} wg_bytes={} threads={}",
                f.stages, f.barriers_per_k_iter, f.workgroup_bytes, f.threads
            ),
            None => println!("  arm {label:<9}: hand-written WGSL (baseline)"),
        }
    }
    println!();

    struct Row {
        shape: &'static str,
        base_ms: f64,
        spread: f64,
        cols: Vec<(String, f64, bool)>,
    }
    let mut rows: Vec<Row> = Vec::new();

    for (shape, m, k, n) in SHAPES {
        let (m, k, n) = (*m, *k, *n);
        let a = fill(m * k, 0x5eed_1234);
        let b = fill(k * n, 0xbeef_9876);

        // One arena, `matmul.wgsl`'s layout: A then B then C.
        let (a_off, b_off, c_off) = (0usize, m * k, m * k + k * n);
        let arena_len = c_off + m * n;
        let mut host = vec![0f32; arena_len];
        host[a_off..a_off + m * k].copy_from_slice(&a);
        host[b_off..b_off + k * n].copy_from_slice(&b);

        let arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("arena"),
            size: (arena_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&arena, 0, bytemuck::cast_slice(&host));

        let params = MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            a_off: a_off as u32,
            b_off: b_off as u32,
            c_off: c_off as u32,
            batch: 1,
            a_batch_stride: (m * k) as u32,
            b_batch_stride: (k * n) as u32,
            c_batch_stride: (m * n) as u32,
            has_bias: 0,
            bias_off: 0,
            act_id: 0xFFFF,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let pbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<MatmulParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&pbuf, 0, bytemuck::bytes_of(&params));

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sched-ab bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: arena.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pbuf.as_entire_binding(),
                },
            ],
        });

        let gx = (n as u32).div_ceil(TILE.tile_n as u32);
        let gy = (m as u32).div_ceil(TILE.tile_m as u32);
        let run = |pipe: &wgpu::ComputePipeline| -> f64 {
            let t0 = Instant::now();
            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cp.set_pipeline(pipe);
                cp.set_bind_group(0, &bg, &[]);
                cp.dispatch_workgroups(gx, gy, 1);
            }
            let sub = queue.submit(Some(enc.finish()));
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: Some(sub),
                timeout: None,
            });
            t0.elapsed().as_secs_f64() * 1e3
        };

        // Read C back through a staging buffer.
        let read_c = || -> Vec<f32> {
            let bytes = (m * n * 4) as u64;
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_buffer_to_buffer(&arena, (c_off * 4) as u64, &staging, 0, bytes);
            let sub = queue.submit(Some(enc.finish()));
            let slice = staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: Some(sub),
                timeout: None,
            });
            let view = slice.get_mapped_range().expect("readback mapped");
            let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
            drop(view);
            staging.unmap();
            out
        };

        // ── Phase 1: correctness, one run per arm ────────────────────────
        let mut reference: Vec<f32> = Vec::new();
        let mut exact: Vec<bool> = Vec::new();
        for (i, (label, pipe, _)) in arms.iter().enumerate() {
            run(pipe);
            let out = read_c();
            if i == 0 {
                reference = out;
            } else {
                let ok = out.len() == reference.len()
                    && out
                        .iter()
                        .zip(&reference)
                        .all(|(x, y)| x.to_bits() == y.to_bits());
                if !ok {
                    let bad = out
                        .iter()
                        .zip(&reference)
                        .position(|(x, y)| x.to_bits() != y.to_bits())
                        .unwrap_or(0);
                    eprintln!(
                        "  {shape}: arm `{label}` NOT bit-exact (element {bad}: {:e} vs {:e})",
                        out[bad], reference[bad]
                    );
                }
                exact.push(ok);
            }
        }

        // ── Phase 2: warm EVERY arm before any is timed ──────────────────
        for (_, pipe, _) in &arms {
            for _ in 0..WARMUP {
                run(pipe);
            }
        }

        // ── Phase 3: round-robin sampling ────────────────────────────────
        let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(ITERS); arms.len()];
        for _ in 0..ITERS {
            for (i, (_, pipe, _)) in arms.iter().enumerate() {
                samples[i].push(run(pipe));
            }
        }
        let stat = |v: &mut Vec<f64>| -> (f64, f64) {
            v.sort_by(|x, y| x.partial_cmp(y).expect("finite timings"));
            (v[v.len() / 2], v[v.len() - 1] / v[0].max(f64::MIN_POSITIVE))
        };
        let (base_ms, spread) = stat(&mut samples[0]);
        let mut cols = Vec::new();
        for i in 1..arms.len() {
            let (med, _) = stat(&mut samples[i]);
            cols.push((arms[i].0.clone(), med, exact[i - 1]));
        }
        rows.push(Row {
            shape,
            base_ms,
            spread,
            cols,
        });
    }

    // ── Report ──────────────────────────────────────────────────────────
    let headers: Vec<String> = rows
        .first()
        .map(|r| r.cols.iter().map(|(l, _, _)| l.clone()).collect())
        .unwrap_or_default();
    print!("{:<22} {:>11} {:>7}", "shape", "shipping ms", "spread");
    for h in &headers {
        print!(" {h:>12}");
    }
    println!();
    println!("{}", "-".repeat(42 + 13 * headers.len()));

    let mut logsum = vec![0.0f64; headers.len()];
    let mut counted = vec![0usize; headers.len()];
    for r in &rows {
        print!("{:<22} {:>11.4} {:>6.2}x", r.shape, r.base_ms, r.spread);
        for (i, (_, ms, ok)) in r.cols.iter().enumerate() {
            if *ok {
                let sp = r.base_ms / ms;
                print!(" {sp:>11.3}x");
                logsum[i] += sp.ln();
                counted[i] += 1;
            } else {
                print!(" {:>12}", "WRONG");
            }
        }
        println!();
    }
    println!("{}", "-".repeat(42 + 13 * headers.len()));
    print!("{:<22} {:>11} {:>7}", "geomean (bit-exact)", "1.000", "");
    for i in 0..headers.len() {
        if counted[i] == 0 {
            print!(" {:>12}", "n/a");
        } else {
            print!(" {:>11.3}x", (logsum[i] / counted[i] as f64).exp());
        }
    }
    println!();

    for (i, h) in headers.iter().enumerate() {
        if counted[i] < rows.len() {
            println!(
                "\nNOTE: arm `{h}` contributed {}/{} shapes; the rest were not bit-exact.",
                counted[i],
                rows.len()
            );
        }
    }
    let worst = rows.iter().map(|r| r.spread).fold(0.0f64, f64::max);
    if worst > 1.5 {
        println!(
            "\nWARNING: worst within-arm spread was {worst:.2}x over {ITERS} samples. Rows \
             that\nnoisy are not a measurement — check the machine is idle and rerun."
        );
    }
    println!(
        "\nRun this on BOTH an Apple adapter (Metal) and an NVIDIA one (Vulkan). Same WGSL,\n\
         so a difference between the two columns is the GPU, not the schedule."
    );
}
