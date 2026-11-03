// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! rlxsl's WGSL helpers on the GPU against IEEE f32, where the operands make
//! their intermediates small: native wgpu compiles Metal in fast math, which
//! flushes subnormals to zero, so a correction term that lands below `2^-126`
//! is lost there and kept by a browser. `rlx_div_rn` and `rlx_sqrt_rn` must be
//! correctly rounded whenever the result is normal; `rlx_exp` (into its
//! subnormal results) and `rlx_log` (of subnormal inputs too) within one ULP of
//! the f64 value, `rlx_tanh` within two.

use wgpu::util::DeviceExt;

const OPS: [&str; 6] = ["div", "sqrt", "rsqrt", "exp", "log", "tanh"];

/// `rlx_<op>(a[i], b[i])` for every `i`, on the GPU.
fn gpu(op: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let dev = rlx_wgpu::device::wgpu_device().unwrap();
    let call = [
        "rlx_div_rn(a[i], b[i])",
        "rlx_sqrt_rn(a[i])",
        "rlx_rsqrt_rn(a[i])",
        "rlx_exp(a[i])",
        "rlx_log(a[i])",
        "rlx_tanh(a[i])",
    ][op];
    let src = format!(
        "@group(0) @binding(0) var<storage, read> a: array<f32>;\n\
         @group(0) @binding(1) var<storage, read> b: array<f32>;\n\
         @group(0) @binding(2) var<storage, read_write> o: array<f32>;\n\
         @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {{\n\
           let i = g.x; if (i >= arrayLength(&o)) {{ return; }} _ = b[i]; o[i] = {call};\n\
         }}\n{}",
        rlxsl::exact::wgsl_numeric_helpers()
    );
    let (d, q) = (&dev.device, &dev.queue);
    let m = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let p = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &m,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let buf = |v: &[f32]| {
        d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(v),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let (ab, bb) = (buf(a), buf(b));
    let bytes = (a.len() * 4) as u64;
    let ob = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let rb = d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &p.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: ab.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bb.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ob.as_entire_binding(),
            },
        ],
    });
    let mut e = d.create_command_encoder(&Default::default());
    {
        let mut pass = e.begin_compute_pass(&Default::default());
        pass.set_pipeline(&p);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups((a.len() as u32).div_ceil(64), 1, 1);
    }
    e.copy_buffer_to_buffer(&ob, 0, &rb, 0, bytes);
    q.submit([e.finish()]);
    rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    bytemuck::cast_slice(&rb.slice(..).get_mapped_range().unwrap()).to_vec()
}

/// Deterministic spread of f32s with exponents in `[lo, hi]`.
fn spread(n: usize, lo: i32, hi: i32, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let e = lo + ((s >> 33) % (hi - lo + 1) as u64) as i32;
            let m = 1.0 + ((s >> 8) & 0x7fffff) as f32 / 8388608.0;
            m * 2f32.powi(e)
        })
        .collect()
}

fn ulps(a: f32, b: f32) -> u32 {
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32
}

#[test]
fn helpers_are_exact_over_small_intermediates() {
    if rlx_ir::env::skip_unless_device("wgpu", true, rlx_wgpu::is_available()) {
        eprintln!("no wgpu adapter — skipping");
        return;
    }
    let n = 1 << 16;
    let mut report = Vec::new();
    for (op, name) in OPS.iter().enumerate() {
        // (a, b, the reference, allowed ULPs)
        let (a, b): (Vec<f32>, Vec<f32>) = match *name {
            // small dividends with normal quotients, the other half anywhere
            "div" => {
                let mut a = spread(n / 2, -126, -60, 1);
                let mut b = spread(n / 2, -60, -1, 2);
                a.extend(spread(n / 2, -60, 60, 3));
                b.extend(spread(n / 2, -60, 60, 4));
                (a, b)
            }
            "sqrt" | "rsqrt" => (spread(n, -126, 127, 5), vec![0.0; n]),
            "exp" => {
                let a: Vec<f32> = (0..n)
                    .map(|i| -104.0 + 192.0 * i as f32 / n as f32)
                    .collect();
                (a, vec![0.0; n])
            }
            "tanh" => {
                let mut a = spread(n / 2, -40, 4, 7);
                a.iter_mut().skip(1).step_by(2).for_each(|v| *v = -*v);
                a.extend((0..n / 2).map(|i| -12.0 + 24.0 * i as f32 / (n / 2) as f32));
                (a, vec![0.0; n])
            }
            // normal inputs, and subnormal ones (bits 1 ..= 0x7fffff)
            _ => {
                let mut a = spread(n - 4096, -126, 127, 6);
                a.extend((0..4096u32).map(|i| f32::from_bits(1 + i * 2047)));
                (a, vec![0.0; n])
            }
        };
        let got = gpu(op, &a, &b);
        let mut bad = 0;
        let mut first = None;
        for i in 0..n {
            let (want, tol) = match *name {
                "div" => (a[i] / b[i], 0),
                "sqrt" => (a[i].sqrt(), 0),
                "rsqrt" => ((1.0 / (a[i] as f64).sqrt()) as f32, 1),
                "exp" => ((a[i] as f64).exp() as f32, 1),
                "tanh" => ((a[i] as f64).tanh() as f32, 2),
                _ => ((a[i] as f64).ln() as f32, 1),
            };
            if !want.is_normal() && want != 0.0 && *name == "div" {
                continue; // subnormal quotients: flushed by Metal, out of scope
            }
            if ulps(got[i], want) > tol {
                bad += 1;
                first.get_or_insert((a[i], b[i], got[i], want));
            }
        }
        if bad > 0 {
            report.push(format!("{name}: {bad}/{n} off, first {first:?}"));
        }
    }
    assert!(report.is_empty(), "\n{}", report.join("\n"));
}
