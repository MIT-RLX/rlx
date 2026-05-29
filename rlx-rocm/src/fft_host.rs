// RLX — versatile ML compiler + runtime.

use crate::device::RocmContext;
use crate::hip::HipBuffer;
use rlx_ir::DType;

pub fn run_fft1d(
    ctx: &RocmContext,
    buffer: &HipBuffer<f32>,
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
    let rt = &ctx.runtime;

    let mut host = vec![0u8; span_len];

    unsafe {
        let _ = (rt.hip_stream_sync)(ctx.default_stream);
        let src_ptr = (buffer.ptr as usize + span_off) as crate::hip::HipDeviceptr;
        let _ = (rt.hip_memcpy_dtoh)(host.as_mut_ptr() as *mut _, src_ptr, span_len);
    }

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

    unsafe {
        let dst_ptr = (buffer.ptr as usize + span_off) as crate::hip::HipDeviceptr;
        let _ = (rt.hip_memcpy_htod)(dst_ptr, host.as_ptr() as *const _, span_len);
    }
}
