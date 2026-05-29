// RLX — versatile ML compiler + runtime.

use crate::buffer::Arena;
use rlx_ir::DType;

pub fn run_fft1d(
    arena: &Arena,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src_byte_off: usize,
    dst_byte_off: usize,
    outer: usize,
    n_complex: usize,
    inverse: bool,
    norm_tag: u32,
    dtype: DType,
) {
    let meta = rlx_ir::fft::FftMeta {
        outer,
        n_complex,
        axis_extent: match dtype {
            DType::C64 => n_complex,
            DType::F32 | DType::F64 => n_complex * 2,
            other => panic!("fft_host: unsupported dtype {other:?}"),
        },
    };
    let row_bytes = meta.row_bytes(dtype);
    let (span_off, span_len) =
        rlx_ir::fft::fft_arena_byte_span(src_byte_off, dst_byte_off, row_bytes, outer);
    let mut host = arena.read_bytes_range(device, queue, span_off, span_len);
    unsafe {
        rlx_cpu::thunk::execute_fft1d(
            src_byte_off - span_off,
            dst_byte_off - span_off,
            outer,
            n_complex,
            inverse,
            norm_tag,
            dtype,
            host.as_mut_ptr(),
        );
    }
    arena.write_bytes_range(queue, span_off, &host);
}
