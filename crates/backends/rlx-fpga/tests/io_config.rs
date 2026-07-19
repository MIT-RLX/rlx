// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna. GPL-3.0-only.

//! Soft I/O configuration: port renames, memory readout, stream ports, binds.

use rlx_fpga::codegen::{collect_artifacts_io, emit_with_config};
use rlx_fpga::export_config::{
    FpgaExportConfig, InputIface, IoConfig, OutputIface, OutputKind, PortNames, SidebandSpec,
};
use rlx_fpga::model::tinyconv_mnist_from_cortexm;
use rlx_fpga::passes::optimize_default;

#[test]
fn default_io_keeps_classic_port_names() {
    let model = tinyconv_mnist_from_cortexm();
    let opt = optimize_default(&model);
    let arts = collect_artifacts_io(&opt, &IoConfig::default());
    let top = arts
        .iter()
        .find(|a| a.rel_path == "top.sv")
        .expect("top.sv");
    assert!(
        top.content
            .contains("input  logic                       clk")
    );
    assert!(
        top.content
            .contains("input  logic                       in_we")
    );
    assert!(
        top.content
            .contains("output logic signed [7:0]          pred")
    );
    assert!(!top.content.contains("out_dout"));
    assert!(!top.content.contains("in_valid"));
}

#[test]
fn renamed_ports_appear_in_top() {
    let model = tinyconv_mnist_from_cortexm();
    let opt = optimize_default(&model);
    let names = PortNames {
        clk: "sys_clk".into(),
        pred: "class_id".into(),
        ..Default::default()
    };
    let io = IoConfig::default().with_names(names);
    let arts = collect_artifacts_io(&opt, &io);
    let top = arts
        .iter()
        .find(|a| a.rel_path == "top.sv")
        .expect("top.sv");
    assert!(top.content.contains("sys_clk"));
    assert!(top.content.contains("class_id"));
    assert!(top.content.contains("always_ff @(posedge sys_clk)"));
}

#[test]
fn logits_auto_scalar_and_memory_readout() {
    let model = tinyconv_mnist_from_cortexm();
    let cfg = FpgaExportConfig::default().with_output_kind(OutputKind::Logits);
    assert!(matches!(cfg.io.output, OutputIface::ScalarAndMemory));
    let dir = tempfile::tempdir().unwrap();
    // Strip argmax so OUT_LEN is logits width.
    let mut model = model;
    while matches!(model.layers.last(), Some(rlx_fpga::Layer::Argmax { .. })) {
        model.layers.pop();
    }
    emit_with_config(&model, &cfg, dir.path()).unwrap();
    let top = std::fs::read_to_string(dir.path().join("top.sv")).unwrap();
    assert!(top.contains("out_dout"));
    assert!(top.contains("out_addr"));
    assert!(top.contains("out_re"));
}

#[test]
fn stream_input_ports_emitted() {
    let model = tinyconv_mnist_from_cortexm();
    let opt = optimize_default(&model);
    let io = IoConfig::default().with_input(InputIface::Stream { beat_elems: 1 });
    let arts = collect_artifacts_io(&opt, &io);
    let top = arts
        .iter()
        .find(|a| a.rel_path == "top.sv")
        .expect("top.sv");
    assert!(top.content.contains("in_valid"));
    assert!(top.content.contains("in_ready"));
    assert!(top.content.contains("in_data"));
    assert!(
        !top.content
            .contains("input  logic                       in_we")
    );
}

#[test]
fn input_iface_parse() {
    assert!(matches!(
        InputIface::parse("memory").unwrap(),
        InputIface::Memory
    ));
    assert!(matches!(
        InputIface::parse("stream:4").unwrap(),
        InputIface::Stream { beat_elems: 4 }
    ));
    assert!(matches!(
        OutputIface::parse("scalar+memory").unwrap(),
        OutputIface::ScalarAndMemory
    ));
}

#[test]
fn scalar_sideband_ports_emitted() {
    let model = tinyconv_mnist_from_cortexm();
    let opt = optimize_default(&model);
    let io = IoConfig::default()
        .sideband(SidebandSpec::input("temp", 8))
        .sideband(SidebandSpec::signed_input("batch_id", 16).with_echo(true));
    let arts = collect_artifacts_io(&opt, &io);
    let top = arts
        .iter()
        .find(|a| a.rel_path == "top.sv")
        .expect("top.sv");
    assert!(top.content.contains("input  logic [7:0]          temp"));
    assert!(top.content.contains("output logic [7:0]         temp_q"));
    assert!(
        top.content
            .contains("input  logic signed [15:0]          batch_id")
    );
    assert!(top.content.contains("assign temp_q = temp_r;"));
    assert!(top.content.contains("temp_r <= temp;"));
    let tb = arts.iter().find(|a| a.rel_path == "tb.sv").expect("tb.sv");
    assert!(tb.content.contains(".temp(temp)"));
    assert!(tb.content.contains(".temp_q(temp_q)"));
}

#[test]
fn sideband_spec_parse() {
    let s = SidebandSpec::parse("temp:12:signed").unwrap();
    assert_eq!(s.name, "temp");
    assert_eq!(s.bits, 12);
    assert!(s.signed);
    assert!(s.echo);
    let u = SidebandSpec::parse("mode").unwrap();
    assert_eq!(u.bits, 8);
    assert!(!u.signed);
}
