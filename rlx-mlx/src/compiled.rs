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

//! Persistent compiled graphs via `mlx::compile`.
//!
//! Strategy: define a Rust callback that lowers an `rlx_ir::Graph`
//! given pre-built leaf MLX arrays. Hand it (with userdata) to the
//! shim's `rlx_mlx_compile`, which wraps it as a `std::function` and
//! feeds it to `mc::compile`. The resulting compiled fn caches the
//! optimized trace internally and replays it on subsequent calls (as
//! long as input shapes match).
//!
//! Lifetime contract:
//!   - `CompiledFn::compile` boxes the userdata so its address is
//!     stable for the compiled handle's lifetime.
//!   - On Drop we free the C++ side first, *then* drop the box —
//!     the C++ wrapper might still hold the userdata pointer until
//!     mc::compile has been finalized.

use std::collections::HashMap;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::ptr;

use rlx_ir::{Graph, NodeId};

use crate::array::{Array, MlxError, check};
use crate::ffi::{self, RLX_MLX_OK, mlx_array_t, mlx_compiled_t};
use crate::lower::{self, LeafKey, lower_with_env};

/// State carried by the C++ wrapper through the lowering callback.
/// Stays alive for the lifetime of the [`CompiledFn`].
struct CompiledState {
    graph: Graph,
    leaf_order: Vec<(NodeId, LeafKey)>,
}

pub struct CompiledFn {
    handle: *mut mlx_compiled_t,
    /// Boxed so its address is stable; the C++ side stores it as a
    /// raw pointer in the std::function closure.
    _state: Box<CompiledState>,
    /// Stash the leaf order separately so callers can build inputs
    /// in the right order without reaching into the box.
    leaf_order: Vec<(NodeId, LeafKey)>,
}

unsafe impl Send for CompiledFn {}

impl CompiledFn {
    /// Compile `graph` once and return a handle that replays the
    /// optimized trace on subsequent calls.
    pub fn compile(graph: Graph) -> Result<Self, MlxError> {
        let leaf_order = lower::leaf_order(&graph);
        let state = Box::new(CompiledState {
            graph,
            leaf_order: leaf_order.clone(),
        });
        let mut handle: *mut mlx_compiled_t = ptr::null_mut();
        let rc = unsafe {
            ffi::rlx_mlx_compile(
                lower_callback,
                state.as_ref() as *const CompiledState as *mut c_void,
                /*shapeless=*/ 0,
                &mut handle,
            )
        };
        check(rc)?;
        Ok(Self {
            handle,
            _state: state,
            leaf_order,
        })
    }

    /// Leaf-order (NodeId, LeafKey) pairs. Callers use this to build
    /// the positional input vector for `invoke`.
    pub fn leaf_order(&self) -> &[(NodeId, LeafKey)] {
        &self.leaf_order
    }

    /// Invoke the compiled function. `inputs` must match `leaf_order`
    /// in length and order. Returns owned output arrays.
    pub fn invoke(&self, inputs: &[Array]) -> Result<Vec<Array>, MlxError> {
        if inputs.len() != self.leaf_order.len() {
            return Err(MlxError(format!(
                "CompiledFn: expected {} leaves, got {}",
                self.leaf_order.len(),
                inputs.len()
            )));
        }
        let in_handles: Vec<*mut mlx_array_t> = inputs.iter().map(|a| a.ptr).collect();
        const CAP: usize = 64;
        let mut out_handles: Vec<*mut mlx_array_t> = vec![ptr::null_mut(); CAP];
        let mut n_out: usize = 0;
        let rc = unsafe {
            ffi::rlx_mlx_compiled_call(
                self.handle,
                in_handles.as_ptr(),
                in_handles.len(),
                out_handles.as_mut_ptr(),
                CAP,
                &mut n_out,
            )
        };
        check(rc)?;
        out_handles.truncate(n_out);
        Ok(out_handles.into_iter().map(Array::from_raw).collect())
    }
}

impl Drop for CompiledFn {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::rlx_mlx_compiled_free(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}

/// extern "C" trampoline that bridges from the C++ wrapper back into
/// Rust. `ud` is a `*const CompiledState`. Inputs are MLX array
/// pointers C++ has handed us; we wrap them as `Array`s (taking
/// ownership of the wrapper Handles — the underlying mc::array is
/// shared-ptr-counted so the C++-side originals are unaffected),
/// run the lowering, and write output pointers back to C++ while
/// `mem::forget`-ing our wrappers (transferring Handle ownership).
unsafe extern "C" fn lower_callback(
    ud: *mut c_void,
    inputs: *const *mut mlx_array_t,
    n_inputs: usize,
    out_outputs: *mut *mut mlx_array_t,
    cap: usize,
    out_n_outputs: *mut usize,
) -> c_int {
    // SAFETY: ud is the *const CompiledState we boxed in
    // CompiledFn::compile and that box outlives the compiled handle.
    let state: &CompiledState = unsafe { &*(ud as *const CompiledState) };

    // Catch Rust panics so they don't unwind across the FFI boundary.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_callback(state, inputs, n_inputs, out_outputs, cap, out_n_outputs)
    }));

    match result {
        Ok(Ok(())) => RLX_MLX_OK,
        Ok(Err(e)) => {
            // Set the C++-side last_error so the wrapper's
            // std::runtime_error carries the real cause across to
            // whoever called rlx_mlx_compiled_call.
            let msg = format!("rlx-mlx compile callback: {e}");
            if let Ok(c) = std::ffi::CString::new(msg) {
                unsafe {
                    ffi::rlx_mlx_set_last_error(c.as_ptr());
                }
            }
            1
        }
        Err(_) => {
            let c = std::ffi::CString::new("rlx-mlx compile callback panicked").unwrap();
            unsafe {
                ffi::rlx_mlx_set_last_error(c.as_ptr());
            }
            1
        }
    }
}

fn run_callback(
    state: &CompiledState,
    inputs: *const *mut mlx_array_t,
    n_inputs: usize,
    out_outputs: *mut *mut mlx_array_t,
    cap: usize,
    out_n_outputs: *mut usize,
) -> Result<(), MlxError> {
    if n_inputs != state.leaf_order.len() {
        return Err(MlxError(format!(
            "compile callback: leaf count mismatch ({} vs {})",
            n_inputs,
            state.leaf_order.len()
        )));
    }
    let in_slice = unsafe { std::slice::from_raw_parts(inputs, n_inputs) };

    let mut env: HashMap<NodeId, Array> = HashMap::with_capacity(state.graph.nodes().len());
    for ((id, _key), &ptr) in state.leaf_order.iter().zip(in_slice) {
        env.insert(*id, Array::from_raw(ptr));
    }
    // Compile mode passes empty param maps — sub-graphs that
    // reference parent params won't work in this trace context.
    // Op::If / Op::While inside a compiled trace will fail to find
    // their sub-graph params and error explicitly. Use Lazy/Eager
    // for graphs that contain control flow until per-call leaf
    // rebinding is wired through compile mode.
    let empty_params: HashMap<String, Vec<f32>> = HashMap::new();
    let empty_typed: HashMap<String, (Vec<u8>, rlx_ir::DType)> = HashMap::new();
    let outs = lower_with_env(&state.graph, env, &empty_params, &empty_typed)?;

    if outs.len() > cap {
        return Err(MlxError(format!(
            "compile callback: {} outputs exceeds cap {}",
            outs.len(),
            cap
        )));
    }

    // Hand handles back to C++; forget wrappers so Drop doesn't run.
    let count = outs.len();
    for (i, arr) in outs.into_iter().enumerate() {
        unsafe {
            *out_outputs.add(i) = arr.ptr;
        }
        std::mem::forget(arr);
    }
    unsafe {
        *out_n_outputs = count;
    }
    Ok(())
}
