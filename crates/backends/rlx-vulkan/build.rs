// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compile the GLSL compute shaders under `shaders/*.comp` to SPIR-V at
//! build time using `naga` (pure Rust). The resulting `.spv` blobs are
//! written to `OUT_DIR` and a generated `shaders_generated.rs` embeds them
//! via `include_bytes!`. No external toolchain (glslang / shaderc / Vulkan
//! SDK) is required, and there is no runtime shader compilation.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let shader_dir = Path::new(&manifest_dir).join("shaders");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=shaders/arena_f32.inc");
    println!("cargo:rerun-if-changed=shaders/arena_u32.inc");
    println!("cargo:rerun-if-changed=build.rs");

    let mut entries: Vec<String> = Vec::new(); // shader names (sorted)

    let mut files: Vec<_> = fs::read_dir(&shader_dir)
        .unwrap_or_else(|e| panic!("rlx-vulkan: cannot read {}: {e}", shader_dir.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "comp").unwrap_or(false))
        .collect();
    files.sort();

    for path in &files {
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let src = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("rlx-vulkan: read {}: {e}", path.display()));

        let words = compile_glsl_to_spirv(&name, &src, &shader_dir);

        // Write SPIR-V words as little-endian bytes.
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        let spv_path = Path::new(&out_dir).join(format!("{name}.spv"));
        fs::write(&spv_path, &bytes)
            .unwrap_or_else(|e| panic!("rlx-vulkan: write {}: {e}", spv_path.display()));
        entries.push(name);
    }

    // The `unary` kernel is assembled, not auto-discovered: its activation
    // dispatch (`rlx_activation_apply`, op 0..28) is @generated from the shared
    // rlxsl manifest using Vulkan's gelu-first opcode ids, inserted right after
    // `#version 450` (before the arena include), then compiled like any other
    // shader. `templates/unary_main.comp` (a subdir, so the loop above skips it)
    // owns the plumbing + cast selectors.
    {
        println!("cargo:rerun-if-changed=shaders/templates/unary_main.comp");
        let main_src = fs::read_to_string(shader_dir.join("templates/unary_main.comp"))
            .expect("rlx-vulkan: read shaders/templates/unary_main.comp");
        let activation = rlxsl::glsl_activation_module(rlxsl::OpcodeScheme::GeluFirst);
        let combined = main_src.replacen(
            "#version 450\n",
            &format!("#version 450\n{activation}\n"),
            1,
        );
        let words = compile_glsl_to_spirv("unary", &combined, &shader_dir);
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        fs::write(Path::new(&out_dir).join("unary.spv"), &bytes)
            .expect("rlx-vulkan: write unary.spv");
        entries.push("unary".to_string());
    }

    // Same assembly for the activation-backward kernel: `rlx_activation_backward`
    // (op 0..17, always relu-first) is auto-differentiated from the forward
    // manifest and prepended to `templates/activation_backward_main.comp`.
    {
        println!("cargo:rerun-if-changed=shaders/templates/activation_backward_main.comp");
        let main_src =
            fs::read_to_string(shader_dir.join("templates/activation_backward_main.comp"))
                .expect("rlx-vulkan: read shaders/templates/activation_backward_main.comp");
        let bwd = rlxsl::glsl_activation_backward_module();
        let combined = main_src.replacen("#version 450\n", &format!("#version 450\n{bwd}\n"), 1);
        let words = compile_glsl_to_spirv("activation_backward", &combined, &shader_dir);
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        fs::write(Path::new(&out_dir).join("activation_backward.spv"), &bytes)
            .expect("rlx-vulkan: write activation_backward.spv");
        entries.push("activation_backward".to_string());
    }

    // Same assembly for the standalone `binary` kernel: `rlx_binary_apply`
    // (op 0..13) is @generated from the shared rlxsl manifest and prepended to
    // `templates/binary_main.comp` (which the auto-discovery loop skips).
    {
        println!("cargo:rerun-if-changed=shaders/templates/binary_main.comp");
        let main_src = fs::read_to_string(shader_dir.join("templates/binary_main.comp"))
            .expect("rlx-vulkan: read shaders/templates/binary_main.comp");
        let bin = rlxsl::binary::glsl_binary_module();
        let combined = main_src.replacen("#version 450\n", &format!("#version 450\n{bin}\n"), 1);
        let words = compile_glsl_to_spirv("binary", &combined, &shader_dir);
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        fs::write(Path::new(&out_dir).join("binary.spv"), &bytes)
            .expect("rlx-vulkan: write binary.spv");
        entries.push("binary".to_string());
    }

    // Same assembly for the standalone `compare` kernel: `rlx_compare_apply`
    // (op 0..5) is @generated from the shared rlxsl manifest and prepended to
    // `templates/compare_main.comp` (which the auto-discovery loop skips).
    {
        println!("cargo:rerun-if-changed=shaders/templates/compare_main.comp");
        let main_src = fs::read_to_string(shader_dir.join("templates/compare_main.comp"))
            .expect("rlx-vulkan: read shaders/templates/compare_main.comp");
        let cmp = rlxsl::compare::glsl_compare_module();
        let combined = main_src.replacen("#version 450\n", &format!("#version 450\n{cmp}\n"), 1);
        let words = compile_glsl_to_spirv("compare", &combined, &shader_dir);
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        fs::write(Path::new(&out_dir).join("compare.spv"), &bytes)
            .expect("rlx-vulkan: write compare.spv");
        entries.push("compare".to_string());
    }

    // Pre-compiled SPIR-V kernels under `shaders/precompiled/*.spv` — kernels
    // naga can't build from GLSL (cooperative matrix / f16). The `.comp` source
    // lives beside each `.spv` for reference, but the committed `.spv` is the
    // build input (no glslang/SDK needed here). Copy into OUT_DIR so they embed
    // with the same `include_bytes!` pattern as the naga-compiled kernels.
    let precompiled_dir = shader_dir.join("precompiled");
    if precompiled_dir.is_dir() {
        let mut spvs: Vec<_> = fs::read_dir(&precompiled_dir)
            .unwrap_or_else(|e| {
                panic!("rlx-vulkan: cannot read {}: {e}", precompiled_dir.display())
            })
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "spv").unwrap_or(false))
            .collect();
        spvs.sort();
        for path in &spvs {
            println!("cargo:rerun-if-changed={}", path.display());
            let name = path.file_stem().unwrap().to_string_lossy().to_string();
            let bytes = fs::read(path)
                .unwrap_or_else(|e| panic!("rlx-vulkan: read {}: {e}", path.display()));
            let spv_path = Path::new(&out_dir).join(format!("{name}.spv"));
            fs::write(&spv_path, &bytes)
                .unwrap_or_else(|e| panic!("rlx-vulkan: write {}: {e}", spv_path.display()));
            entries.push(name);
        }
    }

    // Emit the registry source. Reference the `.spv` blobs RELATIVE to OUT_DIR
    // (resolved by `env!("OUT_DIR")` at the crate's compile time) rather than
    // baking absolute paths — so the embed survives a moved/relocated target
    // dir (e.g. a Docker volume mounted at a different path).
    let mut out_src = String::new();
    out_src.push_str("// @generated by build.rs — GLSL→SPIR-V compute kernels.\n");
    out_src.push_str("/// (kernel name, SPIR-V byte blob) for every shader under `shaders/`.\n");
    out_src.push_str("pub static SHADER_BLOBS: &[(&str, &[u8])] = &[\n");
    for name in &entries {
        writeln!(
            out_src,
            "    ({name:?}, include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{name}.spv\"))),"
        )
        .unwrap();
    }
    out_src.push_str("];\n");
    let gen_path = Path::new(&out_dir).join("shaders_generated.rs");
    fs::write(&gen_path, out_src).unwrap();
}

fn compile_glsl_to_spirv(name: &str, src: &str, shader_dir: &Path) -> Vec<u32> {
    use naga::ShaderStage;
    use naga::back::spv;
    use naga::front::glsl::{Frontend, Options};
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    // Packed-byte / GGUF kernels need the uint+i8 arena helpers (not float-only).
    let u32_arena = name.starts_with("dequant")
        || name.starts_with("quantize")
        || name.starts_with("q_matmul")
        || name.starts_with("q_conv");
    let inc_name = if u32_arena {
        "arena_u32.inc"
    } else {
        "arena_f32.inc"
    };
    let inc = fs::read_to_string(shader_dir.join(inc_name))
        .unwrap_or_else(|e| panic!("rlx-vulkan: read {inc_name}: {e}"));
    let src = inject_arena_include(src, &inc);

    let options = Options::from(ShaderStage::Compute);
    let module = Frontend::default()
        .parse(&options, &src)
        .unwrap_or_else(|e| panic!("rlx-vulkan: GLSL parse error in {name}.comp: {e:?}"));

    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("rlx-vulkan: validation error in {name}.comp: {e:?}"));

    let spv_opts = spv::Options::default();
    let pipe_opts = spv::PipelineOptions {
        shader_stage: ShaderStage::Compute,
        entry_point: "main".to_string(),
    };
    spv::write_vec(&module, &info, &spv_opts, Some(&pipe_opts))
        .unwrap_or_else(|e| panic!("rlx-vulkan: SPIR-V emit error in {name}.comp: {e:?}"))
}

/// Insert dual-buffer arena helpers immediately after `#version 450`.
fn inject_arena_include(src: &str, inc: &str) -> String {
    const VER: &str = "#version 450";
    if let Some(rest) = src.strip_prefix(VER) {
        format!("{VER}\n{inc}{rest}")
    } else {
        format!("{VER}\n{inc}\n{src}")
    }
}
