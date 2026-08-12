// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hand-rolled Metal bindings — the subset of the API this backend uses, spoken
//! straight to the Objective-C runtime.
//!
//! **Why.** This replaces the `metal` crate (`metal-rs`). Its only defect for us
//! was transitive: it pins `block 0.1.6`, whose
//! `extern { static _NSConcreteStackBlock: Class; }` over an uninhabited
//! `enum Class {}` trips the `static of uninhabited type` future-incompatibility
//! lint — an error in a future rustc. `metal-rs` still pins `block` as of 0.33,
//! and `block` is unmaintained, so no version bump clears it. We use none of
//! `block`: this backend installs no Metal completion handlers anywhere, so the
//! dependency was pure cost. Dropping `metal-rs` also drops `foreign-types` and
//! `core-graphics-types`.
//!
//! **Scope** is deliberately *our* surface, not Metal's — compute only. No
//! render pipeline, no heaps, no events, no fences. Adding an API here is a
//! deliberate act, which is the point: the dependency's surface was unbounded
//! and ours is a list you can read.
//!
//! **Compatibility.** Names and signatures mirror `metal-rs` so the ~2500 call
//! sites in this crate keep compiling unchanged — and, more importantly, so does
//! the public `MetalGpuKernel` seam, which hands downstream kernel authors a
//! live `ComputeCommandEncoderRef` and `BufferRef`. Downstream code moves from
//! `use metal::…` to `use rlx_metal::mtl::…` (or `use rlx_metal::mtl as metal;`)
//! and is otherwise source-compatible. The `Ref` types also
//! `unsafe impl objc::Message`, exactly as `metal-rs` did, so the 160 raw
//! `msg_send!` sites in this crate (MPSGraph, ICB) keep working.
//!
//! **Ownership** follows the `foreign-types` split `metal-rs` used:
//! - `Foo` owns a +1 reference and `release`s on drop; `Clone` retains.
//! - `FooRef` is a borrowed view reached by `Deref`, never released.
//!
//! Cocoa's naming rule decides which constructor applies: selectors beginning
//! `new`/`alloc`/`copy` (and C `…Create…`) return +1 and map to
//! [`from_retained`]; everything else is autoreleased and is either borrowed as
//! a `&FooRef` tied to its owner or explicitly retained via `from_autoreleased`.
//! Getting that backwards is a leak or a use-after-free, so each call site below
//! names which rule it is following.

#![allow(non_upper_case_globals)]

use objc::runtime::Object;
use objc::{Encode, Encoding, class, msg_send, sel, sel_impl};
use std::ffi::c_void;
use std::ops::{BitOr, BitOrAssign, Deref};
use std::path::Path;

// ---------------------------------------------------------------------------
// Objective-C plumbing
// ---------------------------------------------------------------------------

/// Stand-in for `extern type` (RFC 1861, still unstable): a zero-sized,
/// never-constructed body so `&FooRef` is a thin pointer straight at the
/// Objective-C object. Same trick `foreign-types` used.
#[repr(C)]
pub struct Opaque {
    _private: [u8; 0],
}

/// `NSUTF8StringEncoding`.
const NS_UTF8: usize = 4;

/// Build an `NSString` from a Rust `&str`. Returns +1 — caller releases.
///
/// `initWithBytes:length:encoding:` copies, so `s` need not outlive the call.
unsafe fn nsstring(s: &str) -> *mut Object {
    unsafe {
        let cls = class!(NSString);
        let alloc: *mut Object = msg_send![cls, alloc];
        msg_send![alloc, initWithBytes: s.as_ptr() as *const c_void
                                 length: s.len()
                               encoding: NS_UTF8]
    }
}

/// Render an `NSError` (or any object) via `localizedDescription`, for the
/// `Result::Err` strings `metal-rs` produced.
unsafe fn err_string(err: *mut Object) -> String {
    if err.is_null() {
        return "unknown Metal error (nil NSError)".to_string();
    }
    unsafe {
        let desc: *mut Object = msg_send![err, localizedDescription];
        if desc.is_null() {
            return "unknown Metal error (nil localizedDescription)".to_string();
        }
        let utf8: *const std::os::raw::c_char = msg_send![desc, UTF8String];
        if utf8.is_null() {
            return "unknown Metal error (nil UTF8String)".to_string();
        }
        std::ffi::CStr::from_ptr(utf8)
            .to_string_lossy()
            .into_owned()
    }
}

/// Run `f` inside an Objective-C autorelease pool.
///
/// Most of Metal's per-iteration objects are autoreleased, not `+1`:
/// `commandBuffer`, every compute/blit encoder, every `NSString` Cocoa hands
/// back. Without a pool on the stack they are not leaked — the process-level
/// pool owns them — but they are not *freed* either, so a long decode loop
/// accumulates one command buffer and one encoder per step for the lifetime of
/// the process. Correct refcounting does not help; only a pool boundary does.
///
/// Wrap a step, not a whole run: the point is that the pool drains each time
/// around.
///
/// ```no_run
/// # use rlx_metal::mtl;
/// # fn step() {}
/// # let n_tokens = 0;
/// for _ in 0..n_tokens {
///     mtl::autoreleasepool(|| step());
/// }
/// ```
#[inline]
pub fn autoreleasepool<T, F: FnOnce() -> T>(f: F) -> T {
    objc::rc::autoreleasepool(f)
}

/// Declare an owned/borrowed Objective-C object pair.
macro_rules! mtl_obj {
    ($(#[$meta:meta])* $owned:ident, $borrowed:ident) => {
        $(#[$meta])*
        #[repr(C)]
        pub struct $borrowed(Opaque);

        // Matches `metal-rs`: lets this crate's raw `msg_send!` sites take a
        // `&FooRef` as receiver or argument.
        unsafe impl ::objc::Message for $borrowed {}
        unsafe impl Send for $borrowed {}
        unsafe impl Sync for $borrowed {}

        impl $borrowed {
            /// The underlying Objective-C pointer.
            #[inline]
            pub fn as_ptr(&self) -> *mut Object {
                self as *const $borrowed as *mut Object
            }

            /// Borrow a raw pointer. The caller ties `'a` to whatever owns it.
            #[inline]
            #[allow(dead_code)]
            pub(crate) unsafe fn borrow<'a>(ptr: *mut Object) -> &'a $borrowed {
                unsafe { &*(ptr as *const $borrowed) }
            }
        }

        $(#[$meta])*
        #[repr(transparent)]
        pub struct $owned(*mut Object);

        unsafe impl Send for $owned {}
        unsafe impl Sync for $owned {}

        impl $owned {
            /// Adopt a +1 reference (a `new…`/`alloc`/`copy`/`…Create…` result).
            #[inline]
            #[allow(dead_code)]
            pub(crate) unsafe fn from_retained(ptr: *mut Object) -> Self {
                Self(ptr)
            }

            /// Retain an autoreleased or borrowed pointer into an owned handle.
            #[inline]
            #[allow(dead_code)]
            pub(crate) unsafe fn from_autoreleased(ptr: *mut Object) -> Self {
                if !ptr.is_null() {
                    let _: *mut Object = unsafe { msg_send![ptr, retain] };
                }
                Self(ptr)
            }

            /// The underlying Objective-C pointer.
            #[inline]
            pub fn as_ptr(&self) -> *mut Object {
                self.0
            }
        }

        impl Deref for $owned {
            type Target = $borrowed;
            #[inline]
            fn deref(&self) -> &$borrowed {
                unsafe { &*(self.0 as *const $borrowed) }
            }
        }

        impl Clone for $owned {
            #[inline]
            fn clone(&self) -> Self {
                unsafe { Self::from_autoreleased(self.0) }
            }
        }

        // The `Borrow`/`ToOwned` pair `foreign-types` provided. Call sites lean
        // on it to promote a borrowed encoder or command buffer — both
        // autoreleased — into a handle they can park in a struct, so `to_owned`
        // must retain rather than alias.
        impl std::borrow::Borrow<$borrowed> for $owned {
            #[inline]
            fn borrow(&self) -> &$borrowed {
                self
            }
        }

        impl std::borrow::ToOwned for $borrowed {
            type Owned = $owned;
            #[inline]
            fn to_owned(&self) -> $owned {
                unsafe { $owned::from_autoreleased(self.as_ptr()) }
            }
        }

        impl Drop for $owned {
            #[inline]
            fn drop(&mut self) {
                if !self.0.is_null() {
                    unsafe {
                        let _: () = msg_send![self.0, release];
                    }
                }
            }
        }

        impl std::fmt::Debug for $owned {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({:p})", stringify!($owned), self.0)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Plain-data types
// ---------------------------------------------------------------------------

/// `MTLSize`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MTLSize {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

impl MTLSize {
    #[inline]
    pub const fn new(width: u64, height: u64, depth: u64) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }
}

// Passed by value to `dispatchThreads:…`, so the runtime needs its layout.
unsafe impl Encode for MTLSize {
    fn encode() -> Encoding {
        let u64_enc = u64::encode();
        let encoding = format!("{{?={0}{0}{0}}}", u64_enc.as_str());
        unsafe { Encoding::from_str(&encoding) }
    }
}

/// `NSRange`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct NSRange {
    pub location: u64,
    pub length: u64,
}

impl NSRange {
    #[inline]
    pub const fn new(location: u64, length: u64) -> Self {
        Self { location, length }
    }
}

unsafe impl Encode for NSRange {
    fn encode() -> Encoding {
        let u64_enc = u64::encode();
        let encoding = format!("{{_NSRange={0}{0}}}", u64_enc.as_str());
        unsafe { Encoding::from_str(&encoding) }
    }
}

/// `MTLResourceOptions`. Storage mode occupies bits 4..8, CPU cache mode 0..4 —
/// hence the shifted constants rather than a plain 0/1/2.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MTLResourceOptions(u64);

impl MTLResourceOptions {
    pub const CPUCacheModeDefaultCache: Self = Self(0);
    pub const CPUCacheModeWriteCombined: Self = Self(1);
    pub const StorageModeShared: Self = Self(0 << 4);
    pub const StorageModeManaged: Self = Self(1 << 4);
    pub const StorageModePrivate: Self = Self(2 << 4);
    pub const StorageModeMemoryless: Self = Self(3 << 4);
    pub const HazardTrackingModeUntracked: Self = Self(1 << 8);

    #[inline]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[inline]
    pub const fn bits(self) -> u64 {
        self.0
    }

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl BitOr for MTLResourceOptions {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for MTLResourceOptions {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// `MTLResourceUsage`. Required before executing an indirect command buffer:
/// commands inside an ICB are invisible to the encoder's dependency tracking, so
/// anything they touch must be declared resident explicitly or the GPU reads
/// nothing and writes nowhere — silently, with a `Completed` command buffer.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MTLResourceUsage(u64);

impl MTLResourceUsage {
    pub const Read: Self = Self(1 << 0);
    pub const Write: Self = Self(1 << 1);
    pub const Sample: Self = Self(1 << 2);

    #[inline]
    pub const fn bits(self) -> u64 {
        self.0
    }
}

impl BitOr for MTLResourceUsage {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// `MTLDispatchType` — whether an encoder's dispatches may overlap.
#[repr(u64)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MTLDispatchType {
    Serial = 0,
    Concurrent = 1,
}

/// `MTLCommandBufferStatus`.
#[repr(u64)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MTLCommandBufferStatus {
    NotEnqueued = 0,
    Enqueued = 1,
    Committed = 2,
    Scheduled = 3,
    Completed = 4,
    Error = 5,
}

/// `MTLIndirectCommandType`.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MTLIndirectCommandType(u64);

impl MTLIndirectCommandType {
    pub const Draw: Self = Self(1 << 0);
    pub const DrawIndexed: Self = Self(1 << 1);
    pub const DrawPatches: Self = Self(1 << 2);
    pub const DrawIndexedPatches: Self = Self(1 << 3);
    pub const ConcurrentDispatch: Self = Self(1 << 5);
    pub const ConcurrentDispatchThreads: Self = Self(1 << 6);

    #[inline]
    pub const fn empty() -> Self {
        Self(0)
    }

    #[inline]
    pub const fn bits(self) -> u64 {
        self.0
    }
}

impl BitOr for MTLIndirectCommandType {
    type Output = Self;
    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

// ---------------------------------------------------------------------------
// Object types
// ---------------------------------------------------------------------------

mtl_obj!(
    /// `MTLDevice`.
    Device,
    DeviceRef
);
mtl_obj!(
    /// `MTLCommandQueue`.
    CommandQueue,
    CommandQueueRef
);
mtl_obj!(
    /// `MTLCommandBuffer`.
    CommandBuffer,
    CommandBufferRef
);
mtl_obj!(
    /// `MTLComputeCommandEncoder`.
    ComputeCommandEncoder,
    ComputeCommandEncoderRef
);
mtl_obj!(
    /// `MTLBuffer`.
    Buffer,
    BufferRef
);
mtl_obj!(
    /// `MTLComputePipelineState`.
    ComputePipelineState,
    ComputePipelineStateRef
);
mtl_obj!(
    /// `MTLLibrary`.
    Library,
    LibraryRef
);
mtl_obj!(
    /// `MTLFunction`.
    Function,
    FunctionRef
);
mtl_obj!(
    /// `MTLFunctionConstantValues`.
    FunctionConstantValues,
    FunctionConstantValuesRef
);
mtl_obj!(
    /// `MTLCompileOptions`.
    CompileOptions,
    CompileOptionsRef
);
mtl_obj!(
    /// `MTLComputePipelineDescriptor`.
    ComputePipelineDescriptor,
    ComputePipelineDescriptorRef
);
mtl_obj!(
    /// `MTLIndirectCommandBuffer`.
    IndirectCommandBuffer,
    IndirectCommandBufferRef
);
mtl_obj!(
    /// `MTLIndirectCommandBufferDescriptor`.
    IndirectCommandBufferDescriptor,
    IndirectCommandBufferDescriptorRef
);
mtl_obj!(
    /// `MTLIndirectComputeCommand`.
    IndirectComputeCommand,
    IndirectComputeCommandRef
);
mtl_obj!(
    /// `MTLBlitCommandEncoder`.
    BlitCommandEncoder,
    BlitCommandEncoderRef
);

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> *mut Object;
}

impl Device {
    /// The system default `MTLDevice`, or `None` on a host without one.
    ///
    /// A C `…Create…` entry point, so the result is +1 and adopted directly.
    pub fn system_default() -> Option<Device> {
        unsafe {
            let ptr = MTLCreateSystemDefaultDevice();
            if ptr.is_null() {
                None
            } else {
                Some(Device::from_retained(ptr))
            }
        }
    }
}

impl DeviceRef {
    /// GPU name, borrowed for as long as `self`.
    ///
    /// The `NSString` is autoreleased but owns its UTF-8 buffer for its own
    /// lifetime, and the device outlives any reasonable use of its name — the
    /// same contract `metal-rs` exposed. Callers here copy it immediately.
    pub fn name(&self) -> &str {
        unsafe {
            let ns: *mut Object = msg_send![self.as_ptr(), name];
            if ns.is_null() {
                return "";
            }
            let utf8: *const std::os::raw::c_char = msg_send![ns, UTF8String];
            if utf8.is_null() {
                return "";
            }
            std::ffi::CStr::from_ptr(utf8).to_str().unwrap_or("")
        }
    }

    pub fn registry_id(&self) -> u64 {
        unsafe { msg_send![self.as_ptr(), registryID] }
    }

    /// Bytes the GPU can hold before the system starts evicting.
    pub fn recommended_max_working_set_size(&self) -> u64 {
        unsafe { msg_send![self.as_ptr(), recommendedMaxWorkingSetSize] }
    }

    /// True on Apple Silicon, where the arena needs no host/device copies.
    pub fn has_unified_memory(&self) -> bool {
        unsafe {
            let v: objc::runtime::BOOL = msg_send![self.as_ptr(), hasUnifiedMemory];
            v != objc::runtime::NO
        }
    }

    pub fn max_buffer_length(&self) -> u64 {
        unsafe { msg_send![self.as_ptr(), maxBufferLength] }
    }

    /// `newCommandQueue` — +1.
    pub fn new_command_queue(&self) -> CommandQueue {
        unsafe {
            let ptr: *mut Object = msg_send![self.as_ptr(), newCommandQueue];
            CommandQueue::from_retained(ptr)
        }
    }

    /// `newBufferWithLength:options:` — +1.
    pub fn new_buffer(&self, length: u64, options: MTLResourceOptions) -> Buffer {
        unsafe {
            let ptr: *mut Object = msg_send![self.as_ptr(),
                newBufferWithLength: length
                            options: options.bits()];
            Buffer::from_retained(ptr)
        }
    }

    /// `newBufferWithBytes:length:options:` — +1. Copies `length` bytes.
    pub fn new_buffer_with_data(
        &self,
        bytes: *const c_void,
        length: u64,
        options: MTLResourceOptions,
    ) -> Buffer {
        unsafe {
            let ptr: *mut Object = msg_send![self.as_ptr(),
                newBufferWithBytes: bytes
                            length: length
                           options: options.bits()];
            Buffer::from_retained(ptr)
        }
    }

    /// `newLibraryWithSource:options:error:` — +1.
    pub fn new_library_with_source(
        &self,
        src: &str,
        options: &CompileOptionsRef,
    ) -> Result<Library, String> {
        unsafe {
            let source = nsstring(src);
            let mut err: *mut Object = std::ptr::null_mut();
            let lib: *mut Object = msg_send![self.as_ptr(),
                newLibraryWithSource: source
                             options: options.as_ptr()
                               error: &mut err];
            let _: () = msg_send![source, release];
            if lib.is_null() {
                Err(err_string(err))
            } else {
                Ok(Library::from_retained(lib))
            }
        }
    }

    /// `newLibraryWithFile:error:` — +1.
    pub fn new_library_with_file<P: AsRef<Path>>(&self, file: P) -> Result<Library, String> {
        let path = file.as_ref().to_string_lossy().into_owned();
        unsafe {
            let ns_path = nsstring(&path);
            let mut err: *mut Object = std::ptr::null_mut();
            let lib: *mut Object = msg_send![self.as_ptr(),
                newLibraryWithFile: ns_path
                             error: &mut err];
            let _: () = msg_send![ns_path, release];
            if lib.is_null() {
                Err(err_string(err))
            } else {
                Ok(Library::from_retained(lib))
            }
        }
    }

    /// `newComputePipelineStateWithFunction:error:` — +1.
    pub fn new_compute_pipeline_state_with_function(
        &self,
        function: &FunctionRef,
    ) -> Result<ComputePipelineState, String> {
        unsafe {
            let mut err: *mut Object = std::ptr::null_mut();
            // Reflection costs compile time, so only ask when someone is going
            // to look at it (`RLX_METAL_VALIDATE_BINDINGS=1`).
            // Pipeline creation itself is left untouched: asking Metal for
            // reflection here (`MTLPipelineOptionArgumentInfo`) aborts unrelated
            // encoders with "Command encoder released without endEncoding", so
            // the declared indices come from the MSL we compiled instead.
            let pipe: *mut Object = msg_send![self.as_ptr(),
                newComputePipelineStateWithFunction: function.as_ptr()
                                              error: &mut err];
            if !pipe.is_null() && bind_validate::enabled() {
                bind_validate::record_pipeline_for_function(pipe, function.as_ptr());
            }
            if pipe.is_null() {
                Err(err_string(err))
            } else {
                Ok(ComputePipelineState::from_retained(pipe))
            }
        }
    }

    /// `newComputePipelineStateWithDescriptor:options:reflection:error:` — +1.
    ///
    /// The descriptor form is what the ICB path needs: it is the only way to set
    /// `supportIndirectCommandBuffers` before the pipeline is built.
    /// `options` is `MTLPipelineOptionNone`, `reflection` nil.
    pub fn new_compute_pipeline_state(
        &self,
        descriptor: &ComputePipelineDescriptorRef,
    ) -> Result<ComputePipelineState, String> {
        unsafe {
            let mut err: *mut Object = std::ptr::null_mut();
            let reflection: *mut Object = std::ptr::null_mut();
            let pipe: *mut Object = msg_send![self.as_ptr(),
                newComputePipelineStateWithDescriptor: descriptor.as_ptr()
                                              options: 0u64
                                           reflection: reflection
                                                error: &mut err];
            if pipe.is_null() {
                Err(err_string(err))
            } else {
                if bind_validate::enabled() {
                    let func: *mut Object = msg_send![descriptor.as_ptr(), computeFunction];
                    if !func.is_null() {
                        bind_validate::record_pipeline_for_function(pipe, func);
                    }
                }
                Ok(ComputePipelineState::from_retained(pipe))
            }
        }
    }

    /// `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:` — +1.
    pub fn new_indirect_command_buffer_with_descriptor(
        &self,
        descriptor: &IndirectCommandBufferDescriptorRef,
        max_count: u64,
        options: MTLResourceOptions,
    ) -> IndirectCommandBuffer {
        unsafe {
            let ptr: *mut Object = msg_send![self.as_ptr(),
                newIndirectCommandBufferWithDescriptor: descriptor.as_ptr()
                                       maxCommandCount: max_count
                                               options: options.bits()];
            IndirectCommandBuffer::from_retained(ptr)
        }
    }
}

// ---------------------------------------------------------------------------
// CommandQueue / CommandBuffer
// ---------------------------------------------------------------------------

impl CommandQueueRef {
    /// `commandBuffer` — autoreleased, borrowed for as long as the queue is.
    ///
    /// Mirrors `metal-rs`, which also handed back a borrowed `&CommandBufferRef`
    /// rather than retaining.
    pub fn new_command_buffer(&self) -> &CommandBufferRef {
        unsafe {
            let ptr: *mut Object = msg_send![self.as_ptr(), commandBuffer];
            CommandBufferRef::borrow(ptr)
        }
    }
}

impl CommandBufferRef {
    /// `computeCommandEncoderWithDispatchType:` — autoreleased, borrowed for as
    /// long as the command buffer is.
    pub fn compute_command_encoder_with_dispatch_type(
        &self,
        ty: MTLDispatchType,
    ) -> &ComputeCommandEncoderRef {
        unsafe {
            let ptr: *mut Object = msg_send![self.as_ptr(),
                computeCommandEncoderWithDispatchType: ty as u64];
            ComputeCommandEncoderRef::borrow(ptr)
        }
    }

    /// `computeCommandEncoder` — autoreleased, borrowed.
    pub fn compute_command_encoder(&self) -> &ComputeCommandEncoderRef {
        unsafe {
            let ptr: *mut Object = msg_send![self.as_ptr(), computeCommandEncoder];
            ComputeCommandEncoderRef::borrow(ptr)
        }
    }

    /// Alias `metal-rs` also exposed under this name.
    pub fn new_compute_command_encoder(&self) -> &ComputeCommandEncoderRef {
        self.compute_command_encoder()
    }

    /// `blitCommandEncoder` — autoreleased, borrowed.
    pub fn new_blit_command_encoder(&self) -> &BlitCommandEncoderRef {
        unsafe {
            let ptr: *mut Object = msg_send![self.as_ptr(), blitCommandEncoder];
            BlitCommandEncoderRef::borrow(ptr)
        }
    }

    pub fn commit(&self) {
        unsafe {
            let _: () = msg_send![self.as_ptr(), commit];
        }
    }

    pub fn wait_until_completed(&self) {
        unsafe {
            let _: () = msg_send![self.as_ptr(), waitUntilCompleted];
        }
    }

    /// The command buffer's `error`, rendered; `None` if it completed cleanly.
    ///
    /// A GPU-side fault does not surface as a Rust error — the submit succeeds,
    /// the wait returns, and the output buffer is simply untouched. This is the
    /// only place that failure is visible, so anything that "ran but produced
    /// nothing" should check here first.
    pub fn error_string(&self) -> Option<String> {
        unsafe {
            let err: *mut Object = msg_send![self.as_ptr(), error];
            if err.is_null() {
                None
            } else {
                Some(err_string(err))
            }
        }
    }

    pub fn status(&self) -> MTLCommandBufferStatus {
        unsafe {
            let raw: u64 = msg_send![self.as_ptr(), status];
            match raw {
                0 => MTLCommandBufferStatus::NotEnqueued,
                1 => MTLCommandBufferStatus::Enqueued,
                2 => MTLCommandBufferStatus::Committed,
                3 => MTLCommandBufferStatus::Scheduled,
                4 => MTLCommandBufferStatus::Completed,
                _ => MTLCommandBufferStatus::Error,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ComputeCommandEncoder
// ---------------------------------------------------------------------------

impl ComputeCommandEncoderRef {
    pub fn set_compute_pipeline_state(&self, state: &ComputePipelineStateRef) {
        if bind_validate::enabled() {
            bind_validate::note_pipeline(self.as_ptr(), state.as_ptr());
        }
        unsafe {
            let _: () = msg_send![self.as_ptr(), setComputePipelineState: state.as_ptr()];
        }
    }

    /// `setBuffer:offset:atIndex:`. `None` unbinds the slot.
    pub fn set_buffer(&self, index: u64, buffer: Option<&BufferRef>, offset: u64) {
        let ptr = buffer.map_or(std::ptr::null_mut(), |b| b.as_ptr());
        if bind_validate::enabled() && !ptr.is_null() {
            bind_validate::note_bind(self.as_ptr(), index);
        }
        unsafe {
            let _: () = msg_send![self.as_ptr(),
                setBuffer: ptr
                   offset: offset
                  atIndex: index];
        }
    }

    /// `setBytes:length:atIndex:` — inline constant data, copied by Metal.
    pub fn set_bytes(&self, index: u64, length: u64, bytes: *const c_void) {
        if bind_validate::enabled() {
            bind_validate::note_bind(self.as_ptr(), index);
        }
        unsafe {
            let _: () = msg_send![self.as_ptr(),
                setBytes: bytes
                  length: length
                 atIndex: index];
        }
    }

    pub fn set_threadgroup_memory_length(&self, index: u64, length: u64) {
        unsafe {
            let _: () = msg_send![self.as_ptr(),
                setThreadgroupMemoryLength: length
                                   atIndex: index];
        }
    }

    /// `dispatchThreads:threadsPerThreadgroup:` — non-uniform threadgroups.
    pub fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        if bind_validate::enabled() {
            bind_validate::check_dispatch(self.as_ptr(), "dispatch_threads");
        }
        unsafe {
            let _: () = msg_send![self.as_ptr(),
                       dispatchThreads: threads_per_grid
                 threadsPerThreadgroup: threads_per_threadgroup];
        }
    }

    /// `dispatchThreadgroups:threadsPerThreadgroup:`.
    pub fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        if bind_validate::enabled() {
            bind_validate::check_dispatch(self.as_ptr(), "dispatch_thread_groups");
        }
        unsafe {
            let _: () = msg_send![self.as_ptr(),
                  dispatchThreadgroups: threadgroups_per_grid
                 threadsPerThreadgroup: threads_per_threadgroup];
        }
    }

    /// `useResource:usage:` — declare a resource resident for this encoder.
    ///
    /// Mandatory for anything an ICB command touches: the encoder cannot see
    /// inside an indirect command buffer, so without this the GPU faults or, on
    /// Apple Silicon, quietly does nothing at all.
    pub fn use_resource(&self, resource: &BufferRef, usage: MTLResourceUsage) {
        unsafe {
            let _: () = msg_send![self.as_ptr(),
                useResource: resource.as_ptr()
                      usage: usage.bits()];
        }
    }

    /// `executeCommandsInBuffer:withRange:`.
    pub fn execute_commands_in_buffer(&self, icb: &IndirectCommandBufferRef, range: NSRange) {
        unsafe {
            let _: () = msg_send![self.as_ptr(),
                executeCommandsInBuffer: icb.as_ptr()
                              withRange: range];
        }
    }

    pub fn end_encoding(&self) {
        unsafe {
            let _: () = msg_send![self.as_ptr(), endEncoding];
        }
        // Raised *after* `endEncoding`: unwinding out of an open encoder makes
        // Metal's own dealloc assertion abort the process, which destroys the
        // very message that explains the bug.
        if bind_validate::enabled() {
            bind_validate::finish(self.as_ptr());
        }
    }
}

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------

impl BufferRef {
    /// Host pointer for a shared-storage buffer. Null for private storage.
    pub fn contents(&self) -> *mut c_void {
        unsafe { msg_send![self.as_ptr(), contents] }
    }

    pub fn length(&self) -> u64 {
        unsafe { msg_send![self.as_ptr(), length] }
    }
}

// ---------------------------------------------------------------------------
// ComputePipelineState
// ---------------------------------------------------------------------------

impl ComputePipelineStateRef {
    /// SIMD width — the natural threadgroup granularity for this pipeline.
    pub fn thread_execution_width(&self) -> u64 {
        unsafe { msg_send![self.as_ptr(), threadExecutionWidth] }
    }

    pub fn max_total_threads_per_threadgroup(&self) -> u64 {
        unsafe { msg_send![self.as_ptr(), maxTotalThreadsPerThreadgroup] }
    }

    /// Threadgroup memory the kernel declared statically — what's left over
    /// bounds any `setThreadgroupMemoryLength:` the encoder adds.
    pub fn static_threadgroup_memory_length(&self) -> u64 {
        unsafe { msg_send![self.as_ptr(), staticThreadgroupMemoryLength] }
    }
}

// ---------------------------------------------------------------------------
// BlitCommandEncoder
// ---------------------------------------------------------------------------

impl BlitCommandEncoderRef {
    /// `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:`.
    pub fn copy_from_buffer(
        &self,
        source: &BufferRef,
        source_offset: u64,
        destination: &BufferRef,
        destination_offset: u64,
        size: u64,
    ) {
        unsafe {
            let _: () = msg_send![self.as_ptr(),
                       copyFromBuffer: source.as_ptr()
                         sourceOffset: source_offset
                             toBuffer: destination.as_ptr()
                    destinationOffset: destination_offset
                                 size: size];
        }
    }

    pub fn end_encoding(&self) {
        unsafe {
            let _: () = msg_send![self.as_ptr(), endEncoding];
        }
    }
}

// ---------------------------------------------------------------------------
// Library / Function
// ---------------------------------------------------------------------------

impl LibraryRef {
    /// `newFunctionWithName:` (or the `constantValues:` form) — +1.
    pub fn get_function(
        &self,
        name: &str,
        constants: Option<FunctionConstantValues>,
    ) -> Result<Function, String> {
        unsafe {
            let ns_name = nsstring(name);
            let func: *mut Object = match constants {
                Some(values) => {
                    let mut err: *mut Object = std::ptr::null_mut();
                    let f: *mut Object = msg_send![self.as_ptr(),
                        newFunctionWithName: ns_name
                             constantValues: values.as_ptr()
                                      error: &mut err];
                    if f.is_null() {
                        let _: () = msg_send![ns_name, release];
                        return Err(err_string(err));
                    }
                    f
                }
                None => msg_send![self.as_ptr(), newFunctionWithName: ns_name],
            };
            let _: () = msg_send![ns_name, release];
            if func.is_null() {
                Err(format!("function '{name}' not found in library"))
            } else {
                Ok(Function::from_retained(func))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CompileOptions
// ---------------------------------------------------------------------------

impl CompileOptions {
    /// `[[MTLCompileOptions alloc] init]` — +1.
    pub fn new() -> Self {
        unsafe {
            let cls = class!(MTLCompileOptions);
            let alloc: *mut Object = msg_send![cls, alloc];
            let obj: *mut Object = msg_send![alloc, init];
            CompileOptions::from_retained(obj)
        }
    }
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl CompileOptionsRef {
    pub fn set_fast_math_enabled(&self, enabled: bool) {
        unsafe {
            let _: () =
                msg_send![self.as_ptr(), setFastMathEnabled: enabled as objc::runtime::BOOL];
        }
    }
}

// ---------------------------------------------------------------------------
// ComputePipelineDescriptor
// ---------------------------------------------------------------------------

impl ComputePipelineDescriptor {
    /// `[[MTLComputePipelineDescriptor alloc] init]` — +1.
    pub fn new() -> Self {
        unsafe {
            let cls = class!(MTLComputePipelineDescriptor);
            let alloc: *mut Object = msg_send![cls, alloc];
            let obj: *mut Object = msg_send![alloc, init];
            ComputePipelineDescriptor::from_retained(obj)
        }
    }
}

impl Default for ComputePipelineDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputePipelineDescriptorRef {
    pub fn set_compute_function(&self, function: Option<&FunctionRef>) {
        let ptr = function.map_or(std::ptr::null_mut(), |f| f.as_ptr());
        unsafe {
            let _: () = msg_send![self.as_ptr(), setComputeFunction: ptr];
        }
    }

    /// Must be set *before* the pipeline is built — an ICB rejects a pipeline
    /// compiled without it, and the failure mode is a segfault at encode time
    /// rather than an error.
    pub fn set_support_indirect_command_buffers(&self, support: bool) {
        unsafe {
            let _: () = msg_send![self.as_ptr(),
                setSupportIndirectCommandBuffers: support as objc::runtime::BOOL];
        }
    }

    pub fn support_indirect_command_buffers(&self) -> bool {
        unsafe {
            let v: objc::runtime::BOOL = msg_send![self.as_ptr(), supportIndirectCommandBuffers];
            v != objc::runtime::NO
        }
    }
}

// ---------------------------------------------------------------------------
// Indirect command buffers
// ---------------------------------------------------------------------------

impl IndirectCommandBufferDescriptor {
    /// `[[MTLIndirectCommandBufferDescriptor alloc] init]` — +1.
    pub fn new() -> Self {
        unsafe {
            let cls = class!(MTLIndirectCommandBufferDescriptor);
            let alloc: *mut Object = msg_send![cls, alloc];
            let obj: *mut Object = msg_send![alloc, init];
            IndirectCommandBufferDescriptor::from_retained(obj)
        }
    }
}

impl Default for IndirectCommandBufferDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl IndirectCommandBufferDescriptorRef {
    pub fn set_command_types(&self, types: MTLIndirectCommandType) {
        unsafe {
            let _: () = msg_send![self.as_ptr(), setCommandTypes: types.bits()];
        }
    }

    pub fn set_inherit_buffers(&self, inherit: bool) {
        unsafe {
            let _: () = msg_send![self.as_ptr(), setInheritBuffers: inherit as objc::runtime::BOOL];
        }
    }

    pub fn set_inherit_pipeline_state(&self, inherit: bool) {
        unsafe {
            let _: () = msg_send![self.as_ptr(),
                setInheritPipelineState: inherit as objc::runtime::BOOL];
        }
    }

    pub fn set_max_kernel_buffer_bind_count(&self, count: u64) {
        unsafe {
            let _: () = msg_send![self.as_ptr(), setMaxKernelBufferBindCount: count];
        }
    }
}

impl IndirectCommandBufferRef {
    /// `indirectComputeCommandAtIndex:` — borrowed from the ICB.
    pub fn indirect_compute_command_at_index(&self, index: u64) -> &IndirectComputeCommandRef {
        unsafe {
            let ptr: *mut Object = msg_send![self.as_ptr(),
                indirectComputeCommandAtIndex: index];
            IndirectComputeCommandRef::borrow(ptr)
        }
    }

    pub fn reset_with_range(&self, range: NSRange) {
        unsafe {
            let _: () = msg_send![self.as_ptr(), resetWithRange: range];
        }
    }
}

impl IndirectComputeCommandRef {
    pub fn set_compute_pipeline_state(&self, state: &ComputePipelineStateRef) {
        if bind_validate::enabled() {
            bind_validate::note_pipeline(self.as_ptr(), state.as_ptr());
        }
        unsafe {
            let _: () = msg_send![self.as_ptr(), setComputePipelineState: state.as_ptr()];
        }
    }

    pub fn set_kernel_buffer(&self, index: u64, buffer: Option<&BufferRef>, offset: u64) {
        let ptr = buffer.map_or(std::ptr::null_mut(), |b| b.as_ptr());
        if bind_validate::enabled() && !ptr.is_null() {
            bind_validate::note_bind(self.as_ptr(), index);
        }
        unsafe {
            let _: () = msg_send![self.as_ptr(),
                setKernelBuffer: ptr
                         offset: offset
                        atIndex: index];
        }
    }

    /// Orders this command after the previous one in the ICB.
    pub fn set_barrier(&self) {
        unsafe {
            let _: () = msg_send![self.as_ptr(), setBarrier];
        }
    }

    pub fn concurrent_dispatch_threadgroups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        if bind_validate::enabled() {
            bind_validate::check_dispatch(self.as_ptr(), "icb concurrent_dispatch_threadgroups");
            bind_validate::finish(self.as_ptr());
        }
        unsafe {
            let _: () = msg_send![self.as_ptr(),
                concurrentDispatchThreadgroups: threadgroups_per_grid
                         threadsPerThreadgroup: threads_per_threadgroup];
        }
    }

    pub fn concurrent_dispatch_threads(
        &self,
        threads_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        if bind_validate::enabled() {
            bind_validate::check_dispatch(self.as_ptr(), "icb concurrent_dispatch_threads");
            bind_validate::finish(self.as_ptr());
        }
        unsafe {
            let _: () = msg_send![self.as_ptr(),
                concurrentDispatchThreads: threads_per_grid
                    threadsPerThreadgroup: threads_per_threadgroup];
        }
    }
}

// ---------------------------------------------------------------------------
// Buffer-binding validation
// ---------------------------------------------------------------------------

/// Cross-check, at dispatch time, that every buffer index a pipeline actually
/// uses was bound before the dispatch.
///
/// This backend binds buffers *by integer index* against MSL signatures that
/// live in a 13k-line string in `kernels.rs`, across ~680 call sites. Nothing
/// connects the two: if a kernel's parameter list changes, every later index
/// shifts, and the result is not a compile error, not a crash, and not even a
/// GPU fault. The unbound index reads zero, so a kernel whose `len` moved reads
/// `len == 0`, every thread returns at `if (gid >= len)`, and the command
/// buffer completes cleanly having written nothing. The ICB encoder shipped
/// exactly that, in five places, for months.
///
/// Metal already knows the answer: pipeline reflection lists each argument's
/// index, type, and whether it is *active* (unused arguments are optimised out
/// and must not be required). So we ask it once per pipeline, remember the
/// active buffer indices, and compare at dispatch.
///
/// **Off by default, and gated by env rather than `debug_assertions`** — the
/// workspace gate runs `cargo test --release`, so a `cfg(debug_assertions)`
/// check would be compiled out of the one place it needs to run. When off the
/// cost is a relaxed atomic load. Turn it on with `RLX_METAL_VALIDATE_BINDINGS=1`
/// (or `just validate-metal-bindings`).
pub mod bind_validate {
    use super::*;
    use std::cell::RefCell;
    use std::collections::{BTreeSet, HashMap};
    use std::sync::atomic::{AtomicU8, Ordering};
    use std::sync::{Mutex, OnceLock};

    static STATE: AtomicU8 = AtomicU8::new(0); // 0 = unknown, 1 = off, 2 = on

    /// Whether validation is enabled. Read once, then a relaxed load.
    #[inline]
    pub fn enabled() -> bool {
        match STATE.load(Ordering::Relaxed) {
            1 => false,
            2 => true,
            _ => {
                let on = rlx_ir::env::flag("RLX_METAL_VALIDATE_BINDINGS");
                STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
                on
            }
        }
    }

    /// pipeline pointer → active buffer indices it declares.
    fn declared() -> &'static Mutex<HashMap<usize, BTreeSet<u64>>> {
        static D: OnceLock<Mutex<HashMap<usize, BTreeSet<u64>>>> = OnceLock::new();
        D.get_or_init(Default::default)
    }

    thread_local! {
        /// encoder/command pointer → (pipeline pointer, indices bound so far).
        ///
        /// Per-thread because encoding is per-thread; keyed by target pointer
        /// because several encoders can be open at once.
        static IN_FLIGHT: RefCell<HashMap<usize, (usize, BTreeSet<u64>)>> =
            RefCell::new(HashMap::new());

        /// Violations seen on a target, raised once it is safe to unwind.
        static VIOLATIONS: RefCell<HashMap<usize, Vec<String>>> = RefCell::new(HashMap::new());
    }

    /// Remember what `pipeline`'s kernel declares, looked up by the function's
    /// name in the MSL we compiled.
    ///
    /// A kernel we can't find (anything outside `kernels::msl_source()` — the
    /// separate `.msl` files, MPS-internal pipelines) records nothing and is
    /// simply not validated, rather than falsely flagged.
    pub(super) unsafe fn record_pipeline_for_function(
        pipeline: *mut Object,
        function: *mut Object,
    ) {
        let Some(name) = (unsafe { function_name(function) }) else {
            return;
        };
        if let Some(indices) = crate::kernels::declared_buffer_indices(&name)
            && let Ok(mut d) = declared().lock()
        {
            d.insert(pipeline as usize, indices);
        }
    }

    unsafe fn function_name(function: *mut Object) -> Option<String> {
        if function.is_null() {
            return None;
        }
        unsafe {
            let ns: *mut Object = msg_send![function, name];
            if ns.is_null() {
                return None;
            }
            let utf8: *const std::os::raw::c_char = msg_send![ns, UTF8String];
            if utf8.is_null() {
                return None;
            }
            std::ffi::CStr::from_ptr(utf8)
                .to_str()
                .ok()
                .map(str::to_owned)
        }
    }

    /// Point `target` at a new pipeline, *keeping* what is already bound.
    ///
    /// Buffer bindings are encoder state, not pipeline state: they survive a
    /// `setComputePipelineState:` and callers legitimately bind before setting
    /// the pipeline. Clearing here reported "bound {}" against kernels that
    /// were correctly bound moments earlier — the validator's own false
    /// positive, not a real one.
    ///
    /// The cost of modelling it correctly is that a binding left over from an
    /// earlier dispatch on the same encoder can mask a genuinely missing one.
    /// That matches what the GPU sees, so it is the honest bound to check.
    pub(super) fn note_pipeline(target: *mut Object, pipeline: *mut Object) {
        IN_FLIGHT.with(|m| {
            let mut map = m.borrow_mut();
            let entry = map
                .entry(target as usize)
                .or_insert_with(|| (pipeline as usize, BTreeSet::new()));
            entry.0 = pipeline as usize;
        });
    }

    /// Record a bind. Creates the entry if the caller bound before setting a
    /// pipeline, which is legal.
    pub(super) fn note_bind(target: *mut Object, index: u64) {
        IN_FLIGHT.with(|m| {
            m.borrow_mut()
                .entry(target as usize)
                .or_insert_with(|| (0, BTreeSet::new()))
                .1
                .insert(index);
        });
    }

    /// Record — but do not raise — a dispatch whose pipeline declares a buffer
    /// index this encode never bound.
    ///
    /// Raising here would be self-defeating: unwinding out of an *open* encoder
    /// drops it without `endEncoding`, Metal's own `-[_MTLCommandEncoder
    /// dealloc]` assertion aborts the process, and libtest discards the
    /// captured output — so the panic message that explains the bug is
    /// destroyed by the abort it causes. Violations are held until
    /// [`finish_encoder`], which runs after `endEncoding`.
    pub(super) fn check_dispatch(target: *mut Object, what: &str) {
        let found = IN_FLIGHT.with(|m| {
            let map = m.borrow();
            let (pipe, bound) = map.get(&(target as usize))?;
            let d = declared().lock().ok()?;
            let want = d.get(pipe)?;
            let missing: Vec<u64> = want.difference(bound).copied().collect();
            if missing.is_empty() {
                None
            } else {
                Some(format!(
                    "{what}: buffer index {missing:?} never bound (kernel declares \
                     {want:?}, encode bound {bound:?})"
                ))
            }
        });
        if let Some(msg) = found {
            VIOLATIONS.with(|v| v.borrow_mut().entry(target as usize).or_default().push(msg));
        }
    }

    /// Raise anything [`check_dispatch`] recorded for this target. Safe to call
    /// only once the encoder is closed (or where none is open, as when building
    /// an indirect command buffer).
    pub(super) fn finish(target: *mut Object) {
        let msgs = VIOLATIONS.with(|v| v.borrow_mut().remove(&(target as usize)));
        forget(target);
        if let Some(msgs) = msgs
            && !msgs.is_empty()
        {
            panic!(
                "rlx-metal: dispatch(es) with unbound buffer indices:\n  {}\n\
                 An unbound index reads zero — typically a `len` that becomes 0, so every \
                 thread returns at `if (gid >= len)` and the dispatch writes nothing, with no \
                 error anywhere. Check these call sites against the kernel's parameter list in \
                 kernels.rs.",
                msgs.join("\n  ")
            );
        }
    }

    /// Test hook: force validation on without depending on process env, which
    /// is read once and shared by every test in the binary.
    #[cfg(test)]
    pub(super) fn force_enable() {
        STATE.store(2, Ordering::Relaxed);
    }

    pub(super) fn forget(target: *mut Object) {
        IN_FLIGHT.with(|m| {
            m.borrow_mut().remove(&(target as usize));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shifted storage-mode constants are easy to typo and the failure mode
    /// is a silently wrong allocation, not an error.
    #[test]
    fn resource_option_bits_match_metal() {
        assert_eq!(MTLResourceOptions::StorageModeShared.bits(), 0);
        assert_eq!(MTLResourceOptions::StorageModeManaged.bits(), 16);
        assert_eq!(MTLResourceOptions::StorageModePrivate.bits(), 32);
        assert_eq!(MTLResourceOptions::empty().bits(), 0);
        let combined = MTLResourceOptions::StorageModePrivate
            | MTLResourceOptions::HazardTrackingModeUntracked;
        assert_eq!(combined.bits(), 32 | 256);
        assert!(combined.contains(MTLResourceOptions::StorageModePrivate));
    }

    #[test]
    fn indirect_command_type_bits_match_metal() {
        assert_eq!(MTLIndirectCommandType::ConcurrentDispatch.bits(), 32);
        assert_eq!(MTLIndirectCommandType::ConcurrentDispatchThreads.bits(), 64);
        let both = MTLIndirectCommandType::ConcurrentDispatch
            | MTLIndirectCommandType::ConcurrentDispatchThreads;
        assert_eq!(both.bits(), 96);
    }

    /// `MTLSize` and `NSRange` are passed *by value* through `objc_msgSend`, so
    /// their layout has to be exactly the C struct or arguments land in the
    /// wrong registers.
    #[test]
    fn by_value_structs_have_c_layout() {
        assert_eq!(std::mem::size_of::<MTLSize>(), 24);
        assert_eq!(std::mem::align_of::<MTLSize>(), 8);
        assert_eq!(std::mem::size_of::<NSRange>(), 16);
        assert_eq!(std::mem::align_of::<NSRange>(), 8);
    }

    #[test]
    fn dispatch_type_and_status_are_u64_reprs() {
        assert_eq!(MTLDispatchType::Serial as u64, 0);
        assert_eq!(MTLDispatchType::Concurrent as u64, 1);
        assert_eq!(MTLCommandBufferStatus::Completed as u64, 4);
        assert_eq!(MTLCommandBufferStatus::Error as u64, 5);
    }

    fn responds(obj: *mut Object, selector: &str) -> bool {
        unsafe {
            let sel = objc::runtime::Sel::register(selector);
            let ok: objc::runtime::BOOL = msg_send![obj, respondsToSelector: sel];
            ok != objc::runtime::NO
        }
    }

    /// Every selector this module sends must actually exist on the receiver.
    ///
    /// This is the failure mode hand-rolled bindings have and generated ones
    /// don't: a misspelled selector is not a compile error and usually not even
    /// a crash — `objc_msgSend` to an unrecognised selector on a nil-tolerant
    /// path just does nothing, so the symptom is silently wrong output far from
    /// the typo. Checking `respondsToSelector:` pins the whole surface at once.
    #[test]
    fn every_selector_we_send_exists() {
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping");
            return;
        };

        let mut missing: Vec<String> = Vec::new();
        let mut check = |obj: *mut Object, what: &str, sels: &[&str]| {
            for s in sels {
                if !responds(obj, s) {
                    missing.push(format!("{what}: {s}"));
                }
            }
        };

        check(
            device.as_ptr(),
            "MTLDevice",
            &[
                "newCommandQueue",
                "newBufferWithLength:options:",
                "newBufferWithBytes:length:options:",
                "newLibraryWithSource:options:error:",
                "newLibraryWithFile:error:",
                "newComputePipelineStateWithFunction:error:",
                "newComputePipelineStateWithDescriptor:options:reflection:error:",
                "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",
                "name",
                "registryID",
                "recommendedMaxWorkingSetSize",
                "hasUnifiedMemory",
                "maxBufferLength",
            ],
        );

        let queue = device.new_command_queue();
        check(queue.as_ptr(), "MTLCommandQueue", &["commandBuffer"]);

        let cb = queue.new_command_buffer();
        check(
            cb.as_ptr(),
            "MTLCommandBuffer",
            &[
                "computeCommandEncoderWithDispatchType:",
                "computeCommandEncoder",
                "blitCommandEncoder",
                "commit",
                "waitUntilCompleted",
                "status",
            ],
        );

        let enc = cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
        check(
            enc.as_ptr(),
            "MTLComputeCommandEncoder",
            &[
                "setComputePipelineState:",
                "setBuffer:offset:atIndex:",
                "setBytes:length:atIndex:",
                "setThreadgroupMemoryLength:atIndex:",
                "dispatchThreads:threadsPerThreadgroup:",
                "dispatchThreadgroups:threadsPerThreadgroup:",
                "useResource:usage:",
                "executeCommandsInBuffer:withRange:",
                "endEncoding",
            ],
        );
        enc.end_encoding();

        let blit = cb.new_blit_command_encoder();
        check(
            blit.as_ptr(),
            "MTLBlitCommandEncoder",
            &[
                "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
                "endEncoding",
            ],
        );
        blit.end_encoding();

        let buf = device.new_buffer(256, MTLResourceOptions::StorageModeShared);
        check(buf.as_ptr(), "MTLBuffer", &["contents", "length"]);

        // A trivial kernel gives us a real library / function / pipeline.
        let opts = CompileOptions::new();
        check(opts.as_ptr(), "MTLCompileOptions", &["setFastMathEnabled:"]);
        let lib = device
            .new_library_with_source("kernel void nop(){}", &opts)
            .expect("compile nop kernel");
        check(lib.as_ptr(), "MTLLibrary", &["newFunctionWithName:"]);
        let func = lib.get_function("nop", None).expect("nop function");
        let pipe = device
            .new_compute_pipeline_state_with_function(&func)
            .expect("nop pipeline");
        check(
            pipe.as_ptr(),
            "MTLComputePipelineState",
            &[
                "threadExecutionWidth",
                "maxTotalThreadsPerThreadgroup",
                "staticThreadgroupMemoryLength",
            ],
        );

        let pdesc = ComputePipelineDescriptor::new();
        check(
            pdesc.as_ptr(),
            "MTLComputePipelineDescriptor",
            &[
                "setComputeFunction:",
                "setSupportIndirectCommandBuffers:",
                "supportIndirectCommandBuffers",
            ],
        );

        let idesc = IndirectCommandBufferDescriptor::new();
        check(
            idesc.as_ptr(),
            "MTLIndirectCommandBufferDescriptor",
            &[
                "setCommandTypes:",
                "setInheritBuffers:",
                "setInheritPipelineState:",
                "setMaxKernelBufferBindCount:",
            ],
        );
        idesc.set_command_types(MTLIndirectCommandType::ConcurrentDispatch);
        idesc.set_inherit_buffers(false);
        idesc.set_inherit_pipeline_state(false);
        idesc.set_max_kernel_buffer_bind_count(8);

        let icb = device.new_indirect_command_buffer_with_descriptor(
            &idesc,
            1,
            MTLResourceOptions::StorageModeShared,
        );
        check(
            icb.as_ptr(),
            "MTLIndirectCommandBuffer",
            &["indirectComputeCommandAtIndex:", "resetWithRange:"],
        );
        let cmd = icb.indirect_compute_command_at_index(0);
        check(
            cmd.as_ptr(),
            "MTLIndirectComputeCommand",
            &[
                "setComputePipelineState:",
                "setKernelBuffer:offset:atIndex:",
                "setBarrier",
                "concurrentDispatchThreadgroups:threadsPerThreadgroup:",
                "concurrentDispatchThreads:threadsPerThreadgroup:",
            ],
        );

        assert!(
            missing.is_empty(),
            "selectors not found:\n  {}",
            missing.join("\n  ")
        );
    }

    /// Minimal end-to-end indirect command buffer: encode one dispatch on the
    /// CPU, execute it, read the result back.
    ///
    /// Deliberately independent of `crate::icb` so a failure localises. ICB has
    /// exactly one silent failure mode — a `Completed` command buffer, no error,
    /// and an untouched output — and it is reached by several different
    /// mistakes (missing `useResource:`, a pipeline built without
    /// `supportIndirectCommandBuffers`, a descriptor missing the dispatch type).
    /// This pins the mechanism itself.
    #[test]
    fn indirect_command_buffer_executes_one_dispatch() {
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        const N: u64 = 64;

        let opts = CompileOptions::new();
        let lib = device
            .new_library_with_source(
                "#include <metal_stdlib>\n\
                 using namespace metal;\n\
                 kernel void icb_fill(device float* out [[buffer(0)]],\n\
                                      constant uint& n [[buffer(1)]],\n\
                                      uint gid [[thread_position_in_grid]]) {\n\
                     if (gid < n) out[gid] = 42.0f;\n\
                 }\n",
                &opts,
            )
            .expect("compile icb_fill");
        let func = lib.get_function("icb_fill", None).expect("icb_fill");

        // The pipeline must be built from a descriptor with the ICB support flag
        // set; a pipeline made by `new_compute_pipeline_state_with_function` is
        // rejected by an indirect command.
        let pdesc = ComputePipelineDescriptor::new();
        pdesc.set_support_indirect_command_buffers(true);
        pdesc.set_compute_function(Some(&func));
        let pipe = device
            .new_compute_pipeline_state(&pdesc)
            .expect("icb pipeline");

        let out = device.new_buffer(N * 4, MTLResourceOptions::StorageModeShared);
        let n_val: u32 = N as u32;
        let n_buf = device.new_buffer_with_data(
            &n_val as *const u32 as *const c_void,
            4,
            MTLResourceOptions::StorageModeShared,
        );

        let idesc = IndirectCommandBufferDescriptor::new();
        idesc.set_command_types(
            MTLIndirectCommandType::ConcurrentDispatch
                | MTLIndirectCommandType::ConcurrentDispatchThreads,
        );
        idesc.set_inherit_buffers(false);
        idesc.set_inherit_pipeline_state(false);
        idesc.set_max_kernel_buffer_bind_count(2);
        let icb = device.new_indirect_command_buffer_with_descriptor(
            &idesc,
            1,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = icb.indirect_compute_command_at_index(0);
        cmd.set_compute_pipeline_state(&pipe);
        cmd.set_kernel_buffer(0, Some(&out), 0);
        cmd.set_kernel_buffer(1, Some(&n_buf), 0);
        let tew = pipe.thread_execution_width().clamp(1, N);
        cmd.concurrent_dispatch_threads(MTLSize::new(N, 1, 1), MTLSize::new(tew, 1, 1));
        // `crate::icb` sets a barrier on every command for serial semantics;
        // exercise it here so this probe stays representative of that path.
        cmd.set_barrier();

        let queue = device.new_command_queue();
        let cb = queue.new_command_buffer();
        let enc = cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
        enc.use_resource(&out, MTLResourceUsage::Read | MTLResourceUsage::Write);
        enc.use_resource(&n_buf, MTLResourceUsage::Read);
        enc.execute_commands_in_buffer(&icb, NSRange::new(0, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        assert_eq!(
            cb.status(),
            MTLCommandBufferStatus::Completed,
            "ICB command buffer did not complete: {:?}",
            cb.error_string()
        );
        assert!(cb.error_string().is_none(), "{:?}", cb.error_string());

        let got = unsafe { std::slice::from_raw_parts(out.contents() as *const f32, N as usize) };
        assert!(
            got.iter().all(|&v| v == 42.0),
            "ICB dispatch did not run: first 8 = {:?}",
            &got[..8]
        );
    }

    /// The binding validator must actually fire on an under-bound dispatch.
    ///
    /// A validator that silently passes everything is worse than none — it
    /// reads as coverage. This deliberately reproduces the ICB defect in
    /// miniature: bind buffer 0, leave the kernel's `len` at buffer 1 unbound,
    /// dispatch. Without the check that dispatch completes cleanly and writes
    /// nothing; with it, it panics naming the missing index.
    #[test]
    fn binding_validator_catches_an_unbound_buffer() {
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        bind_validate::force_enable();

        // Must be a kernel from `kernels::msl_source()` — that is where the
        // declared indices are read from, and an unknown kernel is deliberately
        // not validated. `silu_inplace` takes (data, len) at buffers 0 and 1.
        let opts = CompileOptions::new();
        let lib = device
            .new_library_with_source(&crate::kernels::msl_source(), &opts)
            .expect("assembled MSL compiles");
        let func = lib
            .get_function("silu_inplace", None)
            .expect("silu_inplace");
        // Built after `force_enable`, so its declared indices are recorded.
        let pipe = device
            .new_compute_pipeline_state_with_function(&func)
            .expect("pipeline");
        let data = device.new_buffer(64 * 4, MTLResourceOptions::StorageModeShared);

        let queue = device.new_command_queue();
        let cb = queue.new_command_buffer();
        let enc = cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&data), 0);
        // buffer(1) — `len` — deliberately left unbound: the exact shape of the
        // ICB defect, where it read 0 and the dispatch quietly did nothing.
        enc.dispatch_threads(MTLSize::new(16, 1, 1), MTLSize::new(16, 1, 1));

        // The violation is raised by `end_encoding`, not the dispatch.
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            enc.end_encoding();
        }));
        cb.commit();
        cb.wait_until_completed();

        let msg = match caught {
            Err(p) => p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default(),
            Ok(()) => String::new(),
        };
        assert!(
            msg.contains("never bound"),
            "validator did not flag the unbound buffer(1); got: {msg:?}"
        );
    }

    /// The descriptor flags must actually stick — a setter that silently no-ops
    /// is indistinguishable from one that works until output goes wrong.
    #[test]
    fn descriptor_setters_take_effect() {
        let Some(_device) = Device::system_default() else {
            eprintln!("no Metal device; skipping");
            return;
        };
        let desc = ComputePipelineDescriptor::new();
        assert!(!desc.support_indirect_command_buffers());
        desc.set_support_indirect_command_buffers(true);
        assert!(
            desc.support_indirect_command_buffers(),
            "setSupportIndirectCommandBuffers: did not stick"
        );
    }
}
