// RLX — versatile ML compiler + runtime.

use cudarc::driver::{CudaSlice, CudaStream};
use rlx_ir::DType;
use std::sync::Arc;

pub fn run_fft1d(
    stream: &Arc<CudaStream>,
    buffer: &mut CudaSlice<f32>,
    arena_size_bytes: usize,
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
    let _ = arena_size_bytes;

    stream.synchronize().expect("rlx-cuda: fft pre-sync failed");

    let mut host = vec![0u8; span_len];
    stream
        .memcpy_dtoh(
            &buffer.slice(span_off..span_off + span_len),
            bytemuck::cast_slice_mut(&mut host),
        )
        .expect("rlx-cuda: fft partial dtoh failed");

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

    stream
        .memcpy_htod(
            bytemuck::cast_slice(&host),
            &mut buffer.slice_mut(span_off..span_off + span_len),
        )
        .expect("rlx-cuda: fft partial htod failed");
}
