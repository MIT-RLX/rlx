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

//
// DX12 backend probe for CoopF16Vk (runs when WGPU_BACKEND=dx12).

use rlx_ir::{DType, Graph, Shape};
use rlx_wgpu::backend::WgpuExecutable;

#[test]
fn coop_f16_vk_on_dx12_backend() {
    let dev = match rlx_wgpu::device::wgpu_device() {
        Some(d) => d,
        None => {
            eprintln!("no wgpu adapter, skipping");
            return;
        }
    };
    if dev.backend != wgpu::Backend::Dx12 {
        eprintln!("adapter is {:?}, not DX12 — skipping", dev.backend);
        return;
    }
    if !rlx_wgpu::device::coop_f16_16x16_supported() {
        eprintln!("DX12 adapter lacks 16×16 f16 coop support, skipping");
        return;
    }
    assert!(
        rlx_wgpu::kernels::matmul_coop_f16_vulkan_kernel(&dev.device).is_some(),
        "CoopF16Vk kernel failed to compile on DX12"
    );

    const M: usize = 64;
    const K: usize = 64;
    const N: usize = 64;
    let mut g = Graph::new("coop_f16_vk_dx12");
    let a = g.input("a", Shape::new(&[M, K], DType::F32));
    let b = g.param("b", Shape::new(&[K, N], DType::F32));
    let c = g.matmul(a, b, Shape::new(&[M, N], DType::F32));
    g.set_outputs(vec![c]);

    let mut exe = WgpuExecutable::compile(g);
    exe.set_param("b", &vec![0.01_f32; K * N]);
    let out = exe.run(&[("a", &vec![0.01_f32; M * K])])[0][0];
    assert!(
        out.is_finite(),
        "DX12 CoopF16Vk produced non-finite output: {out}"
    );
}
