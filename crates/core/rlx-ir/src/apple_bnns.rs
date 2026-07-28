// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Small Apple BNNS host primitives for Accelerate fully-connected layers.
//!
//! On macOS, [`NativeBnnsFullyConnected`] wraps `BNNSFilterCreateLayerFullyConnected`
//! with f16 weights/inputs and f32 output. Off macOS the same type is a stub
//! whose [`NativeBnnsFullyConnected::new`] / [`NativeBnnsFullyConnected::apply`]
//! always return `None` (no-ops for cross-compile / non-Apple hosts).

#[cfg(target_os = "macos")]
mod imp {
    use core::ffi::c_void;
    use core::ptr::NonNull;

    const BNNS_LAYOUT_VECTOR: u32 = 0x1_0000;
    const BNNS_LAYOUT_ROW_MAJOR_MATRIX: u32 = 0x2_0000;
    const BNNS_FLOAT16: u32 = 0x1_0000 | 16;
    const BNNS_FLOAT32: u32 = 0x1_0000 | 32;
    const BNNS_ACTIVATION_IDENTITY: u32 = 0;
    const BNNS_ACTIVATION_RELU: u32 = 1;

    #[repr(C)]
    struct BnnsNdArrayDescriptor {
        flags: u32,
        layout: u32,
        size: [usize; 8],
        stride: [usize; 8],
        data: *mut c_void,
        data_type: u32,
        table_data: *mut c_void,
        table_data_type: u32,
        data_scale: f32,
        data_bias: f32,
    }

    #[repr(C)]
    struct BnnsActivation {
        function: u32,
        alpha: f32,
        beta: f32,
        iscale: i32,
        ioffset: i32,
        ishift: i32,
        iscale_per_channel: *const i32,
        ioffset_per_channel: *const i32,
        ishift_per_channel: *const i32,
    }

    #[repr(C)]
    struct BnnsLayerParametersFullyConnected {
        i_desc: BnnsNdArrayDescriptor,
        w_desc: BnnsNdArrayDescriptor,
        o_desc: BnnsNdArrayDescriptor,
        bias: BnnsNdArrayDescriptor,
        activation: BnnsActivation,
    }

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn BNNSFilterCreateLayerFullyConnected(
            layer_params: *const BnnsLayerParametersFullyConnected,
            filter_params: *const c_void,
        ) -> *mut c_void;
        fn BNNSFilterApply(filter: *mut c_void, input: *const c_void, output: *mut c_void) -> i32;
        fn BNNSFilterDestroy(filter: *mut c_void);
    }

    fn vector_desc(data: *mut c_void, n: usize, data_type: u32) -> BnnsNdArrayDescriptor {
        BnnsNdArrayDescriptor {
            flags: 0,
            layout: BNNS_LAYOUT_VECTOR,
            size: [n, 0, 0, 0, 0, 0, 0, 0],
            stride: [1, 0, 0, 0, 0, 0, 0, 0],
            data,
            data_type,
            table_data: core::ptr::null_mut(),
            table_data_type: 0,
            data_scale: 1.0,
            data_bias: 0.0,
        }
    }

    fn matrix_desc(
        data: *mut c_void,
        rows: usize,
        cols: usize,
        data_type: u32,
    ) -> BnnsNdArrayDescriptor {
        BnnsNdArrayDescriptor {
            flags: 0,
            layout: BNNS_LAYOUT_ROW_MAJOR_MATRIX,
            size: [cols, rows, 0, 0, 0, 0, 0, 0],
            stride: [1, cols, 0, 0, 0, 0, 0, 0],
            data,
            data_type,
            table_data: core::ptr::null_mut(),
            table_data_type: 0,
            data_scale: 1.0,
            data_bias: 0.0,
        }
    }

    fn empty_desc() -> BnnsNdArrayDescriptor {
        vector_desc(core::ptr::null_mut(), 0, BNNS_FLOAT32)
    }

    /// Prepacked BNNS fully connected layer with f16 operands and f32 output.
    pub struct NativeBnnsFullyConnected {
        filter: NonNull<c_void>,
        input_size: usize,
        output_size: usize,
        _weights_out_in: Vec<u16>,
        _bias: Option<Vec<f32>>,
    }

    impl NativeBnnsFullyConnected {
        /// Build from f16 bit patterns laid out as `[input, output]`.
        pub fn new(
            weights_in_out: &[u16],
            input_size: usize,
            output_size: usize,
            bias: Option<&[f32]>,
            relu: bool,
        ) -> Option<Self> {
            if input_size == 0
                || output_size == 0
                || weights_in_out.len() != input_size.checked_mul(output_size)?
                || bias.is_some_and(|values| values.len() != output_size)
            {
                return None;
            }

            let mut weights_out_in = vec![0u16; weights_in_out.len()];
            for i in 0..input_size {
                for o in 0..output_size {
                    weights_out_in[o * input_size + i] = weights_in_out[i * output_size + o];
                }
            }
            let mut bias = bias.map(<[f32]>::to_vec);
            let mut input = vec![0u16; input_size];
            let mut output = vec![0.0f32; output_size];
            let params = BnnsLayerParametersFullyConnected {
                i_desc: vector_desc(input.as_mut_ptr().cast(), input_size, BNNS_FLOAT16),
                w_desc: matrix_desc(
                    weights_out_in.as_mut_ptr().cast(),
                    output_size,
                    input_size,
                    BNNS_FLOAT16,
                ),
                o_desc: vector_desc(output.as_mut_ptr().cast(), output_size, BNNS_FLOAT32),
                bias: bias.as_mut().map_or_else(empty_desc, |values| {
                    vector_desc(values.as_mut_ptr().cast(), values.len(), BNNS_FLOAT32)
                }),
                activation: BnnsActivation {
                    function: if relu {
                        BNNS_ACTIVATION_RELU
                    } else {
                        BNNS_ACTIVATION_IDENTITY
                    },
                    alpha: 0.0,
                    beta: 0.0,
                    iscale: 0,
                    ioffset: 0,
                    ishift: 0,
                    iscale_per_channel: core::ptr::null(),
                    ioffset_per_channel: core::ptr::null(),
                    ishift_per_channel: core::ptr::null(),
                },
            };
            // SAFETY: descriptors point to live buffers with matching shapes.
            let filter = unsafe { BNNSFilterCreateLayerFullyConnected(&params, core::ptr::null()) };
            Some(Self {
                filter: NonNull::new(filter)?,
                input_size,
                output_size,
                _weights_out_in: weights_out_in,
                _bias: bias,
            })
        }

        /// Apply to one f16 input vector and return f32 output.
        pub fn apply(&mut self, input: &[u16]) -> Option<Vec<f32>> {
            if input.len() != self.input_size {
                return None;
            }
            let mut output = vec![0.0f32; self.output_size];
            // SAFETY: filter is live and pointers match its input/output types.
            let rc = unsafe {
                BNNSFilterApply(
                    self.filter.as_ptr(),
                    input.as_ptr().cast(),
                    output.as_mut_ptr().cast(),
                )
            };
            (rc == 0 && output.iter().all(|value| value.is_finite())).then_some(output)
        }
    }

    impl Drop for NativeBnnsFullyConnected {
        fn drop(&mut self) {
            // SAFETY: filter is non-null and destroyed exactly once.
            unsafe { BNNSFilterDestroy(self.filter.as_ptr()) };
        }
    }
}

#[cfg(target_os = "macos")]
pub use imp::NativeBnnsFullyConnected;

/// Stub BNNS fully-connected layer (non-macOS): construction and apply are no-ops.
#[cfg(not(target_os = "macos"))]
pub struct NativeBnnsFullyConnected;

#[cfg(not(target_os = "macos"))]
impl NativeBnnsFullyConnected {
    /// Always `None` — BNNS is unavailable off macOS.
    pub fn new(
        _weights_in_out: &[u16],
        _input_size: usize,
        _output_size: usize,
        _bias: Option<&[f32]>,
        _relu: bool,
    ) -> Option<Self> {
        None
    }

    /// Always `None` — BNNS is unavailable off macOS.
    pub fn apply(&mut self, _input: &[u16]) -> Option<Vec<f32>> {
        None
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn f16_fully_connected_applies_bias() {
        let weights = [0x3c00, 0, 0, 0, 0x3c00, 0];
        let mut fc =
            NativeBnnsFullyConnected::new(&weights, 2, 3, Some(&[10.0, 20.0, 30.0]), false)
                .unwrap();
        assert_eq!(fc.apply(&[0x4000, 0x4200]).unwrap(), [12.0, 23.0, 30.0]);
    }
}
