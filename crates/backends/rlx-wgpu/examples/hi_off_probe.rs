// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
fn main() {
    let Some(dev) = rlx_wgpu::device::wgpu_device() else {
        return;
    };
    let lim = dev.device.limits();
    let size = (8u64 << 30).min(lim.max_buffer_size);
    let buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe"),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let n = 16 * 1024 * 1024; // 16M f32 = 64 MiB
    let payload: Vec<f32> = (0..n).map(|i| (i % 1000) as f32).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&payload);
    let hi = 6u64 << 30;
    // chunked write like Arena::write_bytes_range
    const CHUNK: usize = 64 * 1024 * 1024;
    let mut off = 0usize;
    while off < bytes.len() {
        let n = (bytes.len() - off).min(CHUNK);
        dev.queue
            .write_buffer(&buf, hi + off as u64, &bytes[off..off + n]);
        off += n;
    }
    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
    // read first and last 64 floats
    for (label, off) in [("head", hi), ("tail", hi + (bytes.len() as u64) - 256)] {
        let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&buf, off, &staging, 0, 256);
        dev.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (s, r) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |x| {
            let _ = s.send(x);
        });
        let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());
        r.recv().unwrap().unwrap();
        let view = slice.get_mapped_range().unwrap();
        let out: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging.unmap();
        eprintln!("{label}: {:?}", &out[..4]);
    }
}
