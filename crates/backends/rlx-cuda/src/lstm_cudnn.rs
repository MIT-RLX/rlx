// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! cuDNN LSTM forward for Kitten / DynQuantLSTM (batch=1, f32).
//!
//! Packs gate-major PyTorch-order `(i,f,g,o)` weights into a cuDNN weight
//! space via [`cudnnGetRNNWeightParams`], then runs [`cudnnRNNForward`] in
//! inference mode. Opt out with `RLX_CUDA_LSTM_CUDNN=0`.

use cudarc::cudnn::sys as cudnn_sys;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use std::sync::{Arc, Mutex, OnceLock};

fn status_ok(s: cudnn_sys::cudnnStatus_t) -> bool {
    s == cudnn_sys::cudnnStatus_t::CUDNN_STATUS_SUCCESS
}

fn work_pool() -> &'static Mutex<Option<(usize, CudaSlice<u8>)>> {
    static P: OnceLock<Mutex<Option<(usize, CudaSlice<u8>)>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(None))
}

fn ensure_work<'a>(
    stream: &Arc<CudaStream>,
    need: usize,
    pool: &'a mut Option<(usize, CudaSlice<u8>)>,
) -> Option<&'a mut CudaSlice<u8>> {
    let need = need.max(1);
    let grow = match pool.as_ref() {
        Some((cap, _)) => *cap < need,
        None => true,
    };
    if grow {
        let buf = stream.alloc_zeros::<u8>(need).ok()?;
        *pool = Some((need, buf));
    }
    Some(&mut pool.as_mut().unwrap().1)
}

fn env_cudnn_enabled() -> bool {
    match std::env::var("RLX_CUDA_LSTM_CUDNN") {
        Ok(v) => {
            let v = v.trim();
            !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
        }
        Err(_) => true,
    }
}

fn make_rnn_desc(
    input_size: usize,
    hidden: usize,
    bidirectional: bool,
) -> Option<cudnn_sys::cudnnRNNDescriptor_t> {
    unsafe {
        let mut rnn_desc: cudnn_sys::cudnnRNNDescriptor_t = std::ptr::null_mut();
        if !status_ok(cudnn_sys::cudnnCreateRNNDescriptor(&mut rnn_desc)) {
            return None;
        }
        let dir_mode = if bidirectional {
            cudnn_sys::cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL
        } else {
            cudnn_sys::cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL
        };
        if !status_ok(cudnn_sys::cudnnSetRNNDescriptor_v8(
            rnn_desc,
            cudnn_sys::cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD,
            cudnn_sys::cudnnRNNMode_t::CUDNN_LSTM,
            cudnn_sys::cudnnRNNBiasMode_t::CUDNN_RNN_SINGLE_INP_BIAS,
            dir_mode,
            cudnn_sys::cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT,
            cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
            cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
            cudnn_sys::cudnnMathType_t::CUDNN_DEFAULT_MATH,
            input_size as i32,
            hidden as i32,
            hidden as i32,
            1,
            std::ptr::null_mut(),
            // CUDNN_RNN_PADDED_IO_ENABLED — required for UNPACKED layouts.
            1u32,
        )) {
            let _ = cudnn_sys::cudnnDestroyRNNDescriptor(rnn_desc);
            return None;
        }
        Some(rnn_desc)
    }
}

fn htod_raw(stream: &Arc<CudaStream>, dst_dev: u64, host: &[f32]) -> bool {
    if host.is_empty() {
        return true;
    }
    let Ok(mut tmp) = stream.alloc_zeros::<f32>(host.len()) else {
        return false;
    };
    if stream.memcpy_htod(host, &mut tmp).is_err() {
        return false;
    }
    let (src, src_rec) = tmp.device_ptr(stream);
    let ok = unsafe {
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(dst_dev, src, host.len() * 4, stream.cu_stream())
    };
    drop(src_rec);
    ok == cudarc::driver::sys::CUresult::CUDA_SUCCESS
}

fn transpose_rm(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}

fn copy_matrix_to_desc(
    stream: &Arc<CudaStream>,
    desc: cudnn_sys::cudnnTensorDescriptor_t,
    dest: *mut std::ffi::c_void,
    src_row_major: &[f32],
    rows: usize,
    cols: usize,
) -> bool {
    if src_row_major.len() < rows * cols || dest.is_null() {
        return false;
    }
    unsafe {
        let mut dtype = cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT;
        let mut nb = 0i32;
        let mut dim = [0i32; 8];
        let mut stride = [0i32; 8];
        if !status_ok(cudnn_sys::cudnnGetTensorNdDescriptor(
            desc,
            8,
            &mut dtype,
            &mut nb,
            dim.as_mut_ptr(),
            stride.as_mut_ptr(),
        )) {
            return false;
        }
        let mut n = 1usize;
        for d in dim.iter().take(nb as usize) {
            n *= (*d).max(1) as usize;
        }
        if n != rows * cols {
            if rlx_ir::env::flag("RLX_CUDA_LSTM_CUDNN_TRACE") {
                eprintln!(
                    "[lstm_cudnn] matrix size mismatch desc_n={n} expect={} dims={:?}",
                    rows * cols,
                    &dim[..nb as usize]
                );
            }
            return false;
        }
        let row_major = match nb {
            2 => {
                dim[0] as usize == rows
                    && dim[1] as usize == cols
                    && stride[1] == 1
                    && stride[0] == cols as i32
            }
            3 => {
                dim[0] == 1
                    && dim[1] as usize == rows
                    && dim[2] as usize == cols
                    && stride[2] == 1
                    && stride[1] == cols as i32
            }
            _ => false,
        };
        let data = if row_major {
            src_row_major.to_vec()
        } else {
            // Column-major [rows, cols] ≡ transpose of row-major.
            transpose_rm(src_row_major, rows, cols)
        };
        htod_raw(stream, dest as u64, &data)
    }
}

/// Build a cuDNN weight-space buffer from gate-major `[4h, k]` W/R and Wb bias.
pub fn pack_weight_space(
    stream: &Arc<CudaStream>,
    input_size: usize,
    hidden: usize,
    bidirectional: bool,
    w_gate_major: &[f32],
    r_gate_major: &[f32],
    bias_wb: &[f32],
) -> Option<CudaSlice<u8>> {
    if !env_cudnn_enabled() {
        return None;
    }
    let handle = crate::device::cuda_dnn_handle()?;
    let dirs = if bidirectional { 2 } else { 1 };
    let h4 = 4 * hidden;
    if w_gate_major.len() < dirs * h4 * input_size
        || r_gate_major.len() < dirs * h4 * hidden
        || bias_wb.len() < dirs * h4
    {
        return None;
    }

    unsafe {
        let _ = cudarc::cudnn::result::set_stream(
            handle,
            stream.cu_stream() as cudnn_sys::cudaStream_t,
        );
        let rnn_desc = make_rnn_desc(input_size, hidden, bidirectional)?;

        let mut weight_bytes: usize = 0;
        if !status_ok(cudnn_sys::cudnnGetRNNWeightSpaceSize(
            handle,
            rnn_desc,
            &mut weight_bytes,
        )) || weight_bytes == 0
        {
            let _ = cudnn_sys::cudnnDestroyRNNDescriptor(rnn_desc);
            return None;
        }

        let weight_space = stream.alloc_zeros::<u8>(weight_bytes).ok()?;
        let (ws_ptr, ws_rec) = weight_space.device_ptr(stream);

        let mut m_desc: cudnn_sys::cudnnTensorDescriptor_t = std::ptr::null_mut();
        let mut b_desc: cudnn_sys::cudnnTensorDescriptor_t = std::ptr::null_mut();
        if !status_ok(cudnn_sys::cudnnCreateTensorDescriptor(&mut m_desc))
            || !status_ok(cudnn_sys::cudnnCreateTensorDescriptor(&mut b_desc))
        {
            drop(ws_rec);
            let _ = cudnn_sys::cudnnDestroyRNNDescriptor(rnn_desc);
            return None;
        }

        let mut pack_ok = true;
        'dirs: for dir in 0..dirs {
            let pseudo = dir as i32;
            let w_base = dir * h4 * input_size;
            let r_base = dir * h4 * hidden;
            let b_base = dir * h4;
            for gate in 0..4i32 {
                let mut m_addr: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut b_addr: *mut std::ffi::c_void = std::ptr::null_mut();
                if !status_ok(cudnn_sys::cudnnGetRNNWeightParams(
                    handle,
                    rnn_desc,
                    pseudo,
                    weight_bytes,
                    ws_ptr as usize as *const std::ffi::c_void,
                    gate,
                    m_desc,
                    &mut m_addr,
                    b_desc,
                    &mut b_addr,
                )) || m_addr.is_null()
                {
                    pack_ok = false;
                    break 'dirs;
                }
                let g = gate as usize;
                let w_gate = &w_gate_major
                    [w_base + g * hidden * input_size..w_base + (g + 1) * hidden * input_size];
                if !copy_matrix_to_desc(stream, m_desc, m_addr, w_gate, hidden, input_size) {
                    pack_ok = false;
                    break 'dirs;
                }
                if !b_addr.is_null() {
                    let bias = &bias_wb[b_base + g * hidden..b_base + (g + 1) * hidden];
                    if !htod_raw(stream, b_addr as u64, bias) {
                        pack_ok = false;
                        break 'dirs;
                    }
                }
            }
            for gate in 0..4i32 {
                let lin = 4 + gate;
                let mut m_addr: *mut std::ffi::c_void = std::ptr::null_mut();
                let mut b_addr: *mut std::ffi::c_void = std::ptr::null_mut();
                if !status_ok(cudnn_sys::cudnnGetRNNWeightParams(
                    handle,
                    rnn_desc,
                    pseudo,
                    weight_bytes,
                    ws_ptr as usize as *const std::ffi::c_void,
                    lin,
                    m_desc,
                    &mut m_addr,
                    b_desc,
                    &mut b_addr,
                )) || m_addr.is_null()
                {
                    pack_ok = false;
                    break 'dirs;
                }
                let g = gate as usize;
                let r_gate =
                    &r_gate_major[r_base + g * hidden * hidden..r_base + (g + 1) * hidden * hidden];
                if !copy_matrix_to_desc(stream, m_desc, m_addr, r_gate, hidden, hidden) {
                    pack_ok = false;
                    break 'dirs;
                }
            }
        }

        let _ = cudnn_sys::cudnnDestroyTensorDescriptor(m_desc);
        let _ = cudnn_sys::cudnnDestroyTensorDescriptor(b_desc);
        let _ = cudnn_sys::cudnnDestroyRNNDescriptor(rnn_desc);
        drop(ws_rec);

        if pack_ok {
            if rlx_ir::env::flag("RLX_CUDA_LSTM_CUDNN_TRACE") {
                eprintln!(
                    "[lstm_cudnn] packed weight_space={weight_bytes}B in={input_size} h={hidden} dirs={dirs}"
                );
            }
            Some(weight_space)
        } else {
            None
        }
    }
}

/// Run cuDNN LSTM. `workspace` holds both X (at `x_off`) and Y (at `y_off`)
/// in batch-major layout:
/// - x: `[batch, seq, input]`
/// - y: `[batch, seq, dirs*hidden]`
#[allow(clippy::too_many_arguments)]
pub fn forward_workspace(
    stream: &Arc<CudaStream>,
    weight_space: &CudaSlice<u8>,
    workspace: &mut CudaSlice<f32>,
    x_off: usize,
    y_off: usize,
    batch: usize,
    seq: usize,
    input_size: usize,
    hidden: usize,
    bidirectional: bool,
) -> bool {
    if !env_cudnn_enabled() || batch == 0 || seq == 0 {
        return false;
    }
    let Some(handle) = crate::device::cuda_dnn_handle() else {
        return false;
    };
    let dirs = if bidirectional { 2 } else { 1 };

    unsafe {
        if cudarc::cudnn::result::set_stream(handle, stream.cu_stream() as cudnn_sys::cudaStream_t)
            .is_err()
        {
            return false;
        }
        let Some(rnn_desc) = make_rnn_desc(input_size, hidden, bidirectional) else {
            return false;
        };

        let mut x_desc: cudnn_sys::cudnnRNNDataDescriptor_t = std::ptr::null_mut();
        let mut y_desc: cudnn_sys::cudnnRNNDataDescriptor_t = std::ptr::null_mut();
        if !status_ok(cudnn_sys::cudnnCreateRNNDataDescriptor(&mut x_desc))
            || !status_ok(cudnn_sys::cudnnCreateRNNDataDescriptor(&mut y_desc))
        {
            let _ = cudnn_sys::cudnnDestroyRNNDescriptor(rnn_desc);
            return false;
        }
        let seq_lens: Vec<i32> = vec![seq as i32; batch];
        let mut pad = 0f32;
        let layout = cudnn_sys::cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
        if !status_ok(cudnn_sys::cudnnSetRNNDataDescriptor(
            x_desc,
            cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
            layout,
            seq as i32,
            batch as i32,
            input_size as i32,
            seq_lens.as_ptr(),
            &mut pad as *mut f32 as *mut std::ffi::c_void,
        )) || !status_ok(cudnn_sys::cudnnSetRNNDataDescriptor(
            y_desc,
            cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
            layout,
            seq as i32,
            batch as i32,
            (dirs * hidden) as i32,
            seq_lens.as_ptr(),
            &mut pad as *mut f32 as *mut std::ffi::c_void,
        )) {
            let _ = cudnn_sys::cudnnDestroyRNNDataDescriptor(x_desc);
            let _ = cudnn_sys::cudnnDestroyRNNDataDescriptor(y_desc);
            let _ = cudnn_sys::cudnnDestroyRNNDescriptor(rnn_desc);
            return false;
        }

        let mut h_desc: cudnn_sys::cudnnTensorDescriptor_t = std::ptr::null_mut();
        let mut c_desc: cudnn_sys::cudnnTensorDescriptor_t = std::ptr::null_mut();
        if !status_ok(cudnn_sys::cudnnCreateTensorDescriptor(&mut h_desc))
            || !status_ok(cudnn_sys::cudnnCreateTensorDescriptor(&mut c_desc))
        {
            let _ = cudnn_sys::cudnnDestroyRNNDataDescriptor(x_desc);
            let _ = cudnn_sys::cudnnDestroyRNNDataDescriptor(y_desc);
            let _ = cudnn_sys::cudnnDestroyRNNDescriptor(rnn_desc);
            return false;
        }
        let dim = [dirs as i32, batch as i32, hidden as i32];
        let stride = [(batch * hidden) as i32, hidden as i32, 1];
        if !status_ok(cudnn_sys::cudnnSetTensorNdDescriptor(
            h_desc,
            cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
            3,
            dim.as_ptr(),
            stride.as_ptr(),
        )) || !status_ok(cudnn_sys::cudnnSetTensorNdDescriptor(
            c_desc,
            cudnn_sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
            3,
            dim.as_ptr(),
            stride.as_ptr(),
        )) {
            let _ = cudnn_sys::cudnnDestroyTensorDescriptor(h_desc);
            let _ = cudnn_sys::cudnnDestroyTensorDescriptor(c_desc);
            let _ = cudnn_sys::cudnnDestroyRNNDataDescriptor(x_desc);
            let _ = cudnn_sys::cudnnDestroyRNNDataDescriptor(y_desc);
            let _ = cudnn_sys::cudnnDestroyRNNDescriptor(rnn_desc);
            return false;
        }

        let mut work_bytes: usize = 0;
        let mut reserve_bytes: usize = 0;
        if !status_ok(cudnn_sys::cudnnGetRNNTempSpaceSizes(
            handle,
            rnn_desc,
            cudnn_sys::cudnnForwardMode_t::CUDNN_FWD_MODE_INFERENCE,
            x_desc,
            &mut work_bytes,
            &mut reserve_bytes,
        )) {
            let _ = cudnn_sys::cudnnDestroyTensorDescriptor(h_desc);
            let _ = cudnn_sys::cudnnDestroyTensorDescriptor(c_desc);
            let _ = cudnn_sys::cudnnDestroyRNNDataDescriptor(x_desc);
            let _ = cudnn_sys::cudnnDestroyRNNDataDescriptor(y_desc);
            let _ = cudnn_sys::cudnnDestroyRNNDescriptor(rnn_desc);
            return false;
        }

        let mut work_guard = match work_pool().lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        let work = if work_bytes > 0 {
            match ensure_work(stream, work_bytes, &mut work_guard) {
                Some(w) => w,
                None => return false,
            }
        } else {
            match ensure_work(stream, 1, &mut work_guard) {
                Some(w) => w,
                None => return false,
            }
        };
        let (work_ptr, work_rec) = if work_bytes > 0 {
            let (p, r) = work.device_ptr_mut(stream);
            (p as usize as *mut std::ffi::c_void, Some(r))
        } else {
            (std::ptr::null_mut(), None)
        };

        let (ws_ptr, ws_rec) = workspace.device_ptr_mut(stream);
        let (w_ptr, w_rec) = weight_space.device_ptr(stream);
        let x_dev = (ws_ptr + (x_off as u64) * 4) as usize as *const std::ffi::c_void;
        let y_dev = (ws_ptr + (y_off as u64) * 4) as usize as *mut std::ffi::c_void;

        let st = cudnn_sys::cudnnRNNForward(
            handle,
            rnn_desc,
            cudnn_sys::cudnnForwardMode_t::CUDNN_FWD_MODE_INFERENCE,
            std::ptr::null(),
            x_desc,
            x_dev,
            y_desc,
            y_dev,
            h_desc,
            std::ptr::null(),
            std::ptr::null_mut(),
            c_desc,
            std::ptr::null(),
            std::ptr::null_mut(),
            weight_space.len(),
            w_ptr as usize as *const std::ffi::c_void,
            work_bytes,
            work_ptr,
            0,
            std::ptr::null_mut(),
        );
        drop(ws_rec);
        drop(w_rec);
        drop(work_rec);

        let _ = cudnn_sys::cudnnDestroyTensorDescriptor(h_desc);
        let _ = cudnn_sys::cudnnDestroyTensorDescriptor(c_desc);
        let _ = cudnn_sys::cudnnDestroyRNNDataDescriptor(x_desc);
        let _ = cudnn_sys::cudnnDestroyRNNDataDescriptor(y_desc);
        let _ = cudnn_sys::cudnnDestroyRNNDescriptor(rnn_desc);

        let ok = status_ok(st);
        if rlx_ir::env::flag("RLX_CUDA_LSTM_CUDNN_TRACE") {
            eprintln!(
                "[lstm_cudnn] forward ok={ok} status={st:?} seq={seq} in={input_size} h={hidden} dirs={dirs} work={work_bytes}"
            );
        }
        ok
    }
}
