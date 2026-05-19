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

//! Emit the TinyConv-MNIST hardware tree to a tempdir and assert the
//! basic shape of the output: every expected file is present and
//! non-empty, and the SystemVerilog source contains the module names
//! `top` references.

use std::fs;

use rlx_fpga::codegen::emit_model;
use rlx_fpga::model::tinyconv_mnist_from_cortexm;
use rlx_fpga::verilog::mem_hex_bytes;
use rlx_fpga::weights::TEST_IMAGE;

#[test]
fn emit_writes_complete_tree() {
    let dir = tempfile::tempdir().expect("tempdir");
    let model = tinyconv_mnist_from_cortexm();
    emit_model(&model, dir.path()).expect("emit_model failed");

    // Image .mem for the testbench
    fs::write(dir.path().join("tb_image.mem"), mem_hex_bytes(TEST_IMAGE)).unwrap();

    // Default tune is Precision, which has fuse_conv_relu=true → relu
    // kernels are elided. The remaining layers must each have a .sv.
    let must_exist = [
        "primitives/block_rom.sv",
        "primitives/block_ram.sv",
        "primitives/requant_q31.sv",
        "layers/conv1.sv",
        "layers/pool1.sv",
        "layers/conv2.sv",
        "layers/pool2.sv",
        "layers/fc.sv",
        "layers/argmax.sv",
        "weights/conv1_w.mem",
        "weights/conv1_b.mem",
        "weights/conv1_m0.mem",
        "weights/conv1_sh.mem",
        "weights/conv2_w.mem",
        "weights/fc_w.mem",
        "weights/fc_m0.mem",
        "top.sv",
        "tb.sv",
        "tb_image.mem",
    ];
    for rel in &must_exist {
        let p = dir.path().join(rel);
        let md = fs::metadata(&p).unwrap_or_else(|e| panic!("missing {rel}: {e}"));
        assert!(md.len() > 0, "{rel} is empty");
    }

    // Fused-relu layers must NOT exist
    for elided in ["layers/relu1.sv", "layers/relu2.sv"] {
        assert!(
            !dir.path().join(elided).exists(),
            "{elided} should be elided by fuse_conv_relu"
        );
    }

    // top.sv references every remaining kernel's module and instance.
    let top = fs::read_to_string(dir.path().join("top.sv")).unwrap();
    for module in [
        "conv1_kernel",
        "pool1_kernel",
        "conv2_kernel",
        "pool2_kernel",
        "fc_kernel",
        "argmax_kernel",
    ] {
        assert!(top.contains(module), "top.sv missing {module}");
    }
    for inst in [
        "u_conv1", "u_pool1", "u_conv2", "u_pool2", "u_fc", "u_argmax",
    ] {
        assert!(top.contains(inst), "top.sv missing instance {inst}");
    }
    for elided in ["relu1_kernel", "relu2_kernel", "u_relu1", "u_relu2"] {
        assert!(
            !top.contains(elided),
            "top.sv should not reference elided relu: {elided}"
        );
    }

    // Conv1 kernel mentions its weight ROM init file
    let conv1 = fs::read_to_string(dir.path().join("layers/conv1.sv")).unwrap();
    assert!(
        conv1.contains("weights/conv1_w.mem"),
        "conv1.sv missing $readmemh path"
    );
    assert!(
        conv1.contains("requant_q31"),
        "conv1.sv missing requantize instance"
    );

    // Weight .mem files have one hex byte per *packed* byte (ceil(logical / weights_per_byte)).
    let conv1_w = fs::read_to_string(dir.path().join("weights/conv1_w.mem")).unwrap();
    let lines = conv1_w.lines().count();
    let logical = 8 * 3 * 3; // c_out * kh * kw * c_in for conv1
    let bits = rlx_cortexm::model_weights::WEIGHT_BITS;
    let expected = rlx_fpga::pack::packed_byte_len(logical, bits);
    assert_eq!(
        lines, expected,
        "conv1_w.mem has {lines} lines, expected {expected} (logical={logical}, bits={bits})"
    );

    // Image .mem has 784 lines (28*28)
    let img = fs::read_to_string(dir.path().join("tb_image.mem")).unwrap();
    assert_eq!(img.lines().count(), 784);
}
