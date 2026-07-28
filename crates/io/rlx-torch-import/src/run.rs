// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Weight loading + CPU execution (compile HIR → set params → run).

use crate::call::Lowered;
use anyhow::{Context, Result, anyhow, bail};
use half::{bf16, f16};
use rlx_ir::{DType, HirModule};
use std::collections::HashMap;
use std::path::Path;

/// A named tensor decoded to f32 with its shape.
pub type F32Tensor = (Vec<f32>, Vec<usize>);

/// Load every tensor in a safetensors file, decoding to f32.
pub fn load_safetensors_f32(path: &Path) -> Result<HashMap<String, F32Tensor>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if bytes.is_empty() {
        return Ok(HashMap::new());
    }
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .with_context(|| format!("parsing {}", path.display()))?;
    let mut out = HashMap::new();
    for name in st.names() {
        let view = st.tensor(name)?;
        let shape: Vec<usize> = view.shape().to_vec();
        let raw = view.data();
        let data =
            decode_f32(raw, view.dtype()).with_context(|| format!("decoding tensor {name}"))?;
        out.insert(name.to_string(), (data, shape));
    }
    Ok(out)
}

fn decode_f32(raw: &[u8], dt: safetensors::Dtype) -> Result<Vec<f32>> {
    use safetensors::Dtype as D;
    Ok(match dt {
        D::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        D::F64 => raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        D::F16 => raw
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        D::BF16 => raw
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        D::I64 => raw
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        D::I32 => raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        D::I16 => raw
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        D::U8 => raw.iter().map(|&b| b as f32).collect(),
        D::I8 => raw.iter().map(|&b| b as i8 as f32).collect(),
        D::BOOL => raw.iter().map(|&b| (b != 0) as u8 as f32).collect(),
        other => bail!("cannot decode safetensors dtype {other:?} to f32"),
    })
}

/// A named tensor kept in its native encoding (raw bytes + rlx dtype + shape).
pub type RawTensor = (Vec<u8>, DType, Vec<usize>);

fn st_dtype_to_rlx(dt: safetensors::Dtype) -> Result<DType> {
    use safetensors::Dtype as D;
    Ok(match dt {
        D::F32 => DType::F32,
        D::F64 => DType::F64,
        D::F16 => DType::F16,
        D::BF16 => DType::BF16,
        D::I64 => DType::I64,
        D::I32 => DType::I32,
        D::I16 => DType::I16,
        D::I8 => DType::I8,
        D::U8 => DType::U8,
        D::BOOL => DType::Bool,
        other => bail!("unsupported safetensors dtype {other:?}"),
    })
}

/// Load every tensor keeping its native dtype + raw bytes (for typed inputs).
pub fn load_safetensors_raw(path: &Path) -> Result<HashMap<String, RawTensor>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = HashMap::new();
    if bytes.is_empty() {
        return Ok(out);
    }
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    for name in st.names() {
        let view = st.tensor(name)?;
        out.insert(
            name.to_string(),
            (
                view.data().to_vec(),
                st_dtype_to_rlx(view.dtype())?,
                view.shape().to_vec(),
            ),
        );
    }
    Ok(out)
}

/// Decode raw bytes of an rlx dtype to f32 (for output comparison).
pub fn decode_rlx_f32(bytes: &[u8], dt: DType) -> Vec<f32> {
    match dt {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        DType::F64 => bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        DType::I64 => bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::I32 => bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        DType::I16 => bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        DType::U8 => bytes.iter().map(|&b| b as f32).collect(),
        DType::I8 => bytes.iter().map(|&b| b as i8 as f32).collect(),
        DType::Bool => bytes.iter().map(|&b| (b != 0) as u8 as f32).collect(),
        _ => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    }
}

/// Assemble the full runtime param map (loaded weights + synthesized zeros),
/// keyed by HIR param name (the state_dict FQN).
pub fn assemble_params(
    lo: &Lowered,
    weights: &HashMap<String, F32Tensor>,
) -> Result<HashMap<String, Vec<f32>>> {
    let mut params = HashMap::new();
    for p in &lo.params {
        // All params are bound as f32 (integer params are f32 nodes + a cast in
        // hir_build); `load_safetensors_f32` already decodes int weights to f32.
        let (data, _shape) = weights
            .get(&p.key)
            .ok_or_else(|| anyhow!("weight {:?} missing from safetensors", p.key))?;
        params.insert(p.key.clone(), data.clone());
    }
    for z in &lo.zero_params {
        let numel: usize = z.shape.iter().product();
        params.insert(z.key.clone(), vec![0.0f32; numel]);
    }
    Ok(params)
}

/// Integer params (kept in native dtype + raw bytes for `set_param_typed`).
pub fn assemble_typed_params(
    lo: &Lowered,
    raw: &HashMap<String, RawTensor>,
) -> Result<Vec<(String, Vec<u8>, DType)>> {
    let mut out = Vec::new();
    for p in &lo.params {
        if crate::call::is_float_dtype(p.dtype) {
            continue;
        }
        let (bytes, dt, _) = raw
            .get(&p.key)
            .ok_or_else(|| anyhow!("integer weight {:?} missing from safetensors", p.key))?;
        out.push((p.key.clone(), bytes.clone(), *dt));
    }
    Ok(out)
}

/// Compile the HIR on CPU, bind params, run once. Returns one `Vec<f32>` per
/// graph output.
/// Parse a device string ("cpu", "cuda", "metal", …), defaulting to CPU.
#[cfg(feature = "runtime")]
pub fn parse_device(s: &str) -> Result<rlx_runtime::Device> {
    if s.is_empty() || s == "cpu" {
        return Ok(rlx_runtime::Device::Cpu);
    }
    rlx_runtime::parse_device(s).map_err(|e| anyhow!("bad device {s:?}: {e}"))
}

#[cfg(feature = "runtime")]
pub fn run_cpu(
    hir: HirModule,
    params: &HashMap<String, Vec<f32>>,
    typed_params: &[(String, Vec<u8>, DType)],
    inputs: &[(String, Vec<f32>)],
    device: rlx_runtime::Device,
) -> Result<Vec<Vec<f32>>> {
    use rlx_runtime::{CompileOptions, Session};
    let mut compiled = Session::new(device)
        .compile_hir_with(hir, &CompileOptions::default())
        .map_err(|e| anyhow!("HIR compile: {e}"))?;
    for (name, data) in params {
        compiled.set_param(name.as_str(), data);
    }
    for (name, bytes, dt) in typed_params {
        compiled.set_param_typed(name.as_str(), bytes, *dt);
    }
    let refs: Vec<(&str, &[f32])> = inputs
        .iter()
        .map(|(n, d)| (n.as_str(), d.as_slice()))
        .collect();
    Ok(compiled.run(&refs))
}

/// Run a HIR with dynamic (symbolic) input dims, specialized to `binding`.
///
/// The imported HIR carries `Dim::Dynamic(sym)` on the axes the front-end marked
/// dynamic; the compile pipeline re-infers the symbolic graph and `binding` binds
/// each symbol to a concrete size for this run. The same HIR can be run at any
/// binding (each specialization is compiled + cached).
#[cfg(feature = "runtime")]
pub fn run_dynamic(
    hir: HirModule,
    params: &HashMap<String, Vec<f32>>,
    inputs: &[(String, Vec<f32>)],
    device: rlx_runtime::Device,
    binding: &rlx_ir::DimBinding,
) -> Result<Vec<Vec<f32>>> {
    use rlx_runtime::{CompileOptions, DynamicDimCompileCache};
    let mut cache = DynamicDimCompileCache::new(device, 4);
    let opts = CompileOptions::new();
    let compiled = cache
        .get_or_specialize(0, binding, move || hir, &opts)
        .map_err(|e| anyhow!("dynamic HIR compile: {e}"))?;
    for (name, data) in params {
        compiled.set_param(name.as_str(), data);
    }
    let refs: Vec<(&str, &[f32])> = inputs
        .iter()
        .map(|(n, d)| (n.as_str(), d.as_slice()))
        .collect();
    Ok(compiled.run(&refs))
}

/// Like [`run_cpu`] but with typed (e.g. integer) inputs. Returns each output
/// decoded to f32.
#[cfg(feature = "runtime")]
pub fn run_cpu_typed(
    hir: HirModule,
    params: &HashMap<String, Vec<f32>>,
    typed_params: &[(String, Vec<u8>, DType)],
    inputs: &[(String, Vec<u8>, DType)],
    device: rlx_runtime::Device,
) -> Result<Vec<Vec<f32>>> {
    use rlx_runtime::{CompileOptions, Session};
    let mut compiled = Session::new(device)
        .compile_hir_with(hir, &CompileOptions::default())
        .map_err(|e| anyhow!("HIR compile: {e}"))?;
    for (name, data) in params {
        compiled.set_param(name.as_str(), data);
    }
    for (name, bytes, dt) in typed_params {
        compiled.set_param_typed(name.as_str(), bytes, *dt);
    }
    let refs: Vec<(&str, &[u8], DType)> = inputs
        .iter()
        .map(|(n, d, dt)| (n.as_str(), d.as_slice(), *dt))
        .collect();
    Ok(compiled
        .run_typed(&refs)
        .into_iter()
        .map(|(bytes, dt)| decode_rlx_f32(&bytes, dt))
        .collect())
}
