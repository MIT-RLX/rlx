#![allow(unsafe_op_in_unsafe_fn)]

//! Arena slice accessors: turn a `(offset, len)` pair into a `&[T]` / `&mut [T]`
//! view over the execution arena. These are the hottest functions in the thunk
//! executor — hundreds of call sites in the dispatch loop — hence
//! `#[inline(always)]` on every one.

/// Generic raw-cast slice helper. The concrete per-dtype `sl_*` / `sl_mut_*`
/// helpers below are what the rest of the `thunk` module uses at call sites
/// with fixed dtypes; the custom-op dispatcher uses these generic forms to
/// enumerate every `DType` uniformly without listing one helper per dtype.
#[inline(always)]
pub(crate) unsafe fn sl_typed<T>(offset: usize, base: *mut u8, len: usize) -> &'static [T] {
    if offset == usize::MAX {
        return &[];
    }
    unsafe { std::slice::from_raw_parts(base.add(offset) as *const T, len) }
}

#[inline(always)]
pub(crate) unsafe fn sl_mut_typed<T>(offset: usize, base: *mut u8, len: usize) -> &'static mut [T] {
    if offset == usize::MAX {
        return &mut [];
    }
    unsafe { std::slice::from_raw_parts_mut(base.add(offset) as *mut T, len) }
}

/// Generate a read (`$read`) + write (`$write`) arena accessor pair for a
/// concrete element type `$t`. Both are `#[inline(always)]` so the dispatch
/// loop pays no call overhead, and both guard the `usize::MAX` sentinel that
/// `node_offset` emits for a node with no materialized arena buffer (dead /
/// DCE-orphaned). Without the guard, `base.add(usize::MAX)` overflows the
/// pointer — a non-unwinding abort in debug builds — when such a node (e.g. a
/// DCE-orphaned `Transpose` left in the schedule) executes: reads then yield an
/// empty slice and writes are discarded, both safe no-ops (exec kernels iterate
/// the returned slice).
macro_rules! arena_slice_accessors {
    ($(#[$attr:meta])* $read:ident, $write:ident, $t:ty) => {
        $(#[$attr])*
        #[inline(always)]
        pub(crate) unsafe fn $read(offset: usize, base: *mut u8, len: usize) -> &'static [$t] {
            if offset == usize::MAX {
                return &[];
            }
            unsafe { std::slice::from_raw_parts(base.add(offset) as *const $t, len) }
        }

        $(#[$attr])*
        #[inline(always)]
        pub(crate) unsafe fn $write(offset: usize, base: *mut u8, len: usize) -> &'static mut [$t] {
            if offset == usize::MAX {
                return &mut [];
            }
            unsafe { std::slice::from_raw_parts_mut(base.add(offset) as *mut $t, len) }
        }
    };
}

arena_slice_accessors!(sl, sl_mut, f32);
arena_slice_accessors!(sl_f64, sl_mut_f64, f64);
// i32 / i64 typed accessors — siblings for integer-tensor thunks (Sample /
// Gather index buffers). The i32 pair isn't wired to a live thunk yet.
arena_slice_accessors!(
    #[allow(dead_code)]
    sl_i32,
    sl_mut_i32,
    i32
);
arena_slice_accessors!(sl_i64, sl_mut_i64, i64);
