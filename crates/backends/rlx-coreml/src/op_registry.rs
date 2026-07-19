// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// Per-backend kernel registry for `Op::Custom` on CoreML / ANE.
//
// Custom ops run on the host (hybrid execution) — same bytes-in/bytes-out
// contract as `rlx_cpu::op_registry::CpuKernel` and `rlx_metal::op_registry::
// MetalKernel`.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use rlx_ir::Shape;

/// Host-side kernel for one `Op::Custom` name on the CoreML path.
pub trait CoremlKernel: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;

    fn execute(
        &self,
        inputs: &[(&[u8], &Shape)],
        output: (&mut [u8], &Shape),
        attrs: &[u8],
    ) -> Result<(), String>;
}

pub struct CoremlKernelRegistry {
    kernels: RwLock<HashMap<String, Arc<dyn CoremlKernel>>>,
}

impl CoremlKernelRegistry {
    pub fn new() -> Self {
        Self {
            kernels: RwLock::new(HashMap::new()),
        }
    }

    pub fn register(&self, k: Arc<dyn CoremlKernel>) {
        let name = k.name().to_string();
        let mut g = self.kernels.write().unwrap();
        if g.contains_key(&name) {
            eprintln!(
                "rlx-coreml: CoremlKernel '{name}' was already registered — \
                 replacing the previous entry"
            );
        }
        g.insert(name, k);
    }

    pub fn lookup(&self, name: &str) -> Option<Arc<dyn CoremlKernel>> {
        self.kernels.read().unwrap().get(name).cloned()
    }
}

impl Default for CoremlKernelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub fn global_coreml_kernels() -> &'static CoremlKernelRegistry {
    static R: OnceLock<CoremlKernelRegistry> = OnceLock::new();
    R.get_or_init(CoremlKernelRegistry::new)
}

pub fn register_coreml_kernel(k: Arc<dyn CoremlKernel>) {
    global_coreml_kernels().register(k);
}

/// Register rlx-coreml's built-in host-delegate kernels exactly once, before
/// any lookup — currently the `collective.*` ops (host/transport delegates over
/// the registered rlx-cpu kernel). See [`crate::collective`].
fn ensure_builtins_registered() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(crate::collective::register);
}

pub fn lookup_coreml_kernel(name: &str) -> Option<Arc<dyn CoremlKernel>> {
    ensure_builtins_registered();
    global_coreml_kernels().lookup(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::DType;

    #[derive(Debug)]
    struct StubKernel;
    impl CoremlKernel for StubKernel {
        fn name(&self) -> &str {
            "stub.coreml"
        }
        fn execute(
            &self,
            _inputs: &[(&[u8], &Shape)],
            _output: (&mut [u8], &Shape),
            _attrs: &[u8],
        ) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn register_and_lookup_round_trips() {
        let reg = CoremlKernelRegistry::new();
        reg.register(Arc::new(StubKernel));
        let k = reg
            .lookup("stub.coreml")
            .expect("registered kernel must be findable");
        assert_eq!(k.name(), "stub.coreml");
    }

    #[test]
    fn execute_signature_compiles_and_runs() {
        let k: Arc<dyn CoremlKernel> = Arc::new(StubKernel);
        let in_shape = Shape::new(&[4], DType::F32);
        let out_shape = Shape::new(&[4], DType::F32);
        let in_bytes = vec![0u8; 16];
        let mut out_bytes = vec![0u8; 16];
        k.execute(&[(&in_bytes, &in_shape)], (&mut out_bytes, &out_shape), &[])
            .expect("stub kernel must succeed");
    }
}
