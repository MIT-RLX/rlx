//! Probe: can a compute shader read/write at a 6 GiB offset into one buffer?
fn main() {
    let Some(dev) = rlx_wgpu::device::wgpu_device() else {
        eprintln!("no device");
        return;
    };
    let lim = dev.device.limits();
    let size = (8u64 << 30).min(lim.max_buffer_size);
    let bind_cap = lim.max_storage_buffer_binding_size;
    eprintln!(
        "max_buf={:.3}GiB bind_cap={:.3}GiB",
        size as f64 / (1u64 << 30) as f64,
        bind_cap as f64 / (1u64 << 30) as f64
    );
    if size < (7u64 << 30) {
        eprintln!("skip");
        return;
    }
    let buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe"),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let hi = 6u64 << 30;
    let payload: Vec<f32> = (0..256).map(|i| (i as f32) + 1.0).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&payload);
    dev.queue.write_buffer(&buf, hi, bytes);
    // also clear low region
    let zeros = vec![0u8; bytes.len()];
    dev.queue.write_buffer(&buf, 0, &zeros);

    // Kernel: out[i] = in[i] with window bound at `hi`, size = bind window covering payload
    let shader = dev
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                r#"
struct P { n: u32, in_off: u32, out_off: u32, _p: u32 }
@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@group(0) @binding(1) var<uniform> params: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }
    arena[params.out_off + i] = arena[params.in_off + i] * 2.0;
}
"#,
            )),
        });
    let bgl = dev
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
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
    let layout = dev
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let pipeline = dev
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct P {
        n: u32,
        in_off: u32,
        out_off: u32,
        _p: u32,
    }

    // Case A: bind window at hi, copy hi -> hi+1KB (both in window)
    let win = 256u64 * 1024; // 256 KiB window
    let u = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    dev.queue.write_buffer(
        &u,
        0,
        bytemuck::bytes_of(&P {
            n: 256,
            in_off: 0,
            out_off: 256,
            _p: 0,
        }),
    );
    let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buf,
                    offset: hi,
                    size: wgpu::BufferSize::new(win),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: u.as_entire_binding(),
            },
        ],
    });
    let mut enc = dev
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(4, 1, 1);
    }
    dev.queue.submit(Some(enc.finish()));
    let _ = dev.device.poll(wgpu::PollType::wait_indefinitely());

    // read hi+256*4
    let staging = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = dev
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(&buf, hi + 256 * 4, &staging, 0, 256 * 4);
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
    eprintln!(
        "caseA out[0]={:?} expect 2.0  out[255]={:?} expect 512.0  ok={}",
        out[0],
        out[255],
        out[0] == 2.0 && out[255] == 512.0
    );
    eprintln!("done");
}
