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

//! Custom-op extensibility (plan #25).
//!
//! Borrowed from MAX's `extensibility/compiler_internal/` /
//! `extensibility/tensor/` pattern: downstream users register their
//! own ops + executors without forking the framework.
//!
//! Today this is the data layer — a registry mapping a string op
//! name to an executor closure. The matching `#[rlx_op]` proc macro
//! is the syntactic sugar layer; adding it is straightforward when a
//! real consumer needs less boilerplate.
//!
//! The IR doesn't model custom ops natively (there's no
//! `Op::Custom("name")` variant) — they enter through the runtime
//! via `CustomOpRegistry::execute` rather than as graph nodes.
//! That's deliberate: the optimizer's fusion patterns can't reason
//! about ops it doesn't know, so custom ops should be opaque
//! "black box" sub-stages rather than first-class IR citizens.
//! Promote them to real ops once a fusion pattern would benefit.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Boxed executor: takes (read-only inputs) → produces an owned output.
/// `Vec<Vec<f32>>` for now since that matches the `rlx_runtime`
/// `CompiledGraph::run` signature; revisit when the runtime moves to
/// `Buffer` (plan #59) end-to-end.
pub type CustomOpFn = Box<dyn Fn(&[&[f32]]) -> Vec<f32> + Send + Sync>;

struct Registry {
    map: Mutex<HashMap<String, CustomOpFn>>,
}

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| Registry {
        map: Mutex::new(HashMap::new()),
    })
}

/// Register a custom op under `name`. Idempotent — re-registering
/// replaces. Names are arbitrary strings; convention: dotted
/// namespacing like `"my-crate.my-op"`.
pub fn register<F>(name: impl Into<String>, f: F)
where
    F: Fn(&[&[f32]]) -> Vec<f32> + Send + Sync + 'static,
{
    let r = registry();
    let mut m = r.map.lock().expect("custom-op registry poisoned");
    m.insert(name.into(), Box::new(f));
}

/// Execute a previously-registered op. Returns `None` if the op
/// isn't registered.
pub fn execute(name: &str, inputs: &[&[f32]]) -> Option<Vec<f32>> {
    let r = registry();
    let m = r.map.lock().expect("custom-op registry poisoned");
    m.get(name).map(|f| f(inputs))
}

/// Snapshot of registered op names (sorted, deterministic).
pub fn registered() -> Vec<String> {
    let r = registry();
    let m = r.map.lock().expect("custom-op registry poisoned");
    let mut v: Vec<String> = m.keys().cloned().collect();
    v.sort();
    v
}

#[doc(hidden)]
pub fn clear_for_tests() {
    let r = registry();
    r.map.lock().unwrap().clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_then_execute() {
        clear_for_tests();
        register("test.identity", |ins| ins[0].to_vec());
        let out = execute("test.identity", &[&[1.0, 2.0, 3.0]]).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn unknown_op_returns_none() {
        clear_for_tests();
        assert!(execute("nope", &[]).is_none());
    }

    #[test]
    fn re_register_replaces() {
        clear_for_tests();
        register("test.f", |_| vec![1.0]);
        register("test.f", |_| vec![2.0]);
        assert_eq!(execute("test.f", &[]).unwrap(), vec![2.0]);
    }
}
