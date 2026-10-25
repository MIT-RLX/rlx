// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::op::{Activation, BinaryOp};
use crate::{DType, NodeOrigin, Op, Shape};

/// Equal down to every byte of every node, origins included.
fn same(a: &Graph, b: &Graph) -> bool {
    bincode::serialize(a).unwrap() == bincode::serialize(b).unwrap()
}

fn round_trip(g: &Graph) -> Graph {
    let text = to_text(g).unwrap();
    let back = from_text(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    assert!(same(g, &back), "differs after a round trip:\n{text}");
    // And the text it prints is the text it read.
    assert_eq!(to_text(&back).unwrap(), text);
    back
}

fn affine() -> Graph {
    let mut g = Graph::new("affine");
    let x = g.input("x", Shape::new(&[1, 2], DType::F32));
    let w = g.param("w", Shape::new(&[2, 2], DType::F32));
    let y = g.add_node(Op::MatMul, vec![x, w], Shape::new(&[1, 2], DType::F32));
    let z = g.add_node(
        Op::Activation(Activation::Gelu),
        vec![y],
        Shape::new(&[1, 2], DType::F32),
    );
    g.set_outputs(vec![z]);
    g
}

#[test]
fn the_documented_example_is_what_prints() {
    let text = to_text(&affine()).unwrap();
    let header = format!("rlxmodel 1\nopset 1\nrlx {RLX_VERSION}\n\n");
    assert_eq!(
        text.strip_prefix(header.as_str()).unwrap_or(&text),
        "\
         graph @affine {\n  \
         %x = input : f32[1, 2]\n  \
         %w = param : f32[2, 2]\n  \
         %2 = MatMul %x, %w : f32[1, 2]\n  \
         %3 = Activation(Gelu) %2 : f32[1, 2]\n  \
         return %3\n\
         }\n"
    );
    round_trip(&affine());
}

#[test]
fn floats_keep_nan_infinities_and_their_exact_value() {
    let mut g = Graph::new("clamp");
    let s = Shape::new(&[4], DType::F32);
    let x = g.input("x", s.clone());
    let lo = g.add_node(
        Op::Clamp {
            min: f32::NEG_INFINITY,
            max: 0.1,
        },
        vec![x],
        s.clone(),
    );
    let hi = g.add_node(
        Op::Clamp {
            min: 1e-30,
            max: f32::INFINITY,
        },
        vec![lo],
        s.clone(),
    );
    let nan = g.add_node(
        Op::Clamp {
            min: f32::NAN,
            max: f32::MAX,
        },
        vec![hi],
        s,
    );
    g.set_outputs(vec![nan]);
    let text = to_text(&g).unwrap();
    assert!(text.contains("Clamp{min: -inf, max: 0.1}"), "{text}");
    assert!(text.contains("min: nan"), "{text}");
    round_trip(&g);
}

#[test]
fn bytes_dynamic_dims_names_and_origins_round_trip() {
    let mut g = Graph::new("odd name");
    let s = Shape::from_dims(&[crate::Dim::Dynamic(0), crate::Dim::Static(3)], DType::F32);
    let a = g.input("a", s.clone());
    let c = g.add_node(
        Op::Constant {
            data: (0..12u8).collect(),
        },
        vec![],
        Shape::new(&[3], DType::F32),
    );
    let sum = g.add_node(Op::Binary(BinaryOp::Add), vec![a, c], s);
    g.node_mut(sum).name = Some("sum \"of\" two\n".into());
    g.node_mut(sum).origin = Some(NodeOrigin {
        hir: None,
        label: Some("block.0".into()),
        pass: Some("fuse".into()),
    });
    g.set_outputs(vec![sum, a]);
    let text = to_text(&g).unwrap();
    assert!(text.contains("graph @\"odd name\""), "{text}");
    // Small enough to print as its values (here, three tiny subnormals).
    assert!(text.contains("%1 = const ["), "{text}");
    assert!(text.contains("f32[?0, 3]"), "{text}");
    round_trip(&g);
}

#[test]
fn nested_bodies_print_as_graphs_and_round_trip() {
    let s = Shape::new(&[4], DType::F32);
    let mut body = Graph::new("body");
    let c = body.input("carry", s.clone());
    let y = body.add_node(Op::Activation(Activation::Gelu), vec![c], s.clone());
    body.set_outputs(vec![y]);
    let mut g = Graph::new("outer");
    let init = g.input("init", s.clone());
    let out = g.add_node(
        Op::Scan {
            body: Box::new(body),
            length: 8,
            save_trajectory: false,
            num_bcast: 0,
            num_xs: 0,
            num_checkpoints: 0,
        },
        vec![init],
        s,
    );
    g.set_outputs(vec![out]);
    let text = to_text(&g).unwrap();
    assert!(text.contains("Scan{body: graph @body {\n"), "{text}");
    round_trip(&g);
}

#[test]
fn several_graphs_in_one_file() {
    let (a, mut b) = (affine(), affine());
    b.name = "second".into();
    let text = write_graphs(&[&a, &b]).unwrap();
    let back = read_graphs(&text).unwrap();
    assert_eq!(back.len(), 2);
    assert!(same(&back[0], &a) && same(&back[1], &b));
    assert!(
        from_text(&text).is_err(),
        "from_text takes exactly one graph"
    );
}

#[test]
fn hand_written_text_reads_with_comments_and_free_layout() {
    let g = from_text(
        "rlxmodel 1 opset 1\n\
         // a comment\n\
         graph @g {\n\
           %0 = Input { name: \"x\" } : F32[ 8 ]   // dtype in any case\n\
           %1 = Reshape{new_shape: [2, -1]} %0 : f32[2, 4]\n\
           return %1\n\
         }",
    )
    .unwrap();
    assert_eq!(g.len(), 2);
    assert_eq!(g.node(g.outputs[0]).shape.dims().len(), 2);
}

#[test]
fn errors_say_where_and_what() {
    let e = from_text("rlxmodel 1\nopset 1\ngraph @g {\n  %0 = NoSuchOp : f32[1]\n  return %0\n}")
        .unwrap_err()
        .to_string();
    assert!(e.contains("NoSuchOp") && e.contains("3:1"), "{e}");

    let e = from_text(
        "rlxmodel 1\nopset 1\ngraph @g {\n  %a = input : f32[1]\n  %1 = MatMul %a f32[1]\n}",
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("5:") && e.contains("expected `:`"), "{e}");

    let e = from_text("rlxmodel 1\nopset 99\n").unwrap_err().to_string();
    assert!(e.contains("opset 99 is newer"), "{e}");
    assert!(
        from_text("graph @g { return }").is_err(),
        "the header is required"
    );
}

#[test]
fn every_nan_keeps_its_bits() {
    let mut g = Graph::new("nans");
    let s = Shape::new(&[1], DType::F32);
    let x = g.input("x", s.clone());
    let neg = f32::from_bits(0xffc0_0000);
    let payload = f32::from_bits(0x7fc0_0001);
    let a = g.add_node(
        Op::Clamp {
            min: neg,
            max: payload,
        },
        vec![x],
        s.clone(),
    );
    let b = g.add_node(
        Op::Clamp {
            min: f32::NAN,
            max: 0.0,
        },
        vec![a],
        s,
    );
    g.set_outputs(vec![b]);
    let text = to_text(&g).unwrap();
    assert!(
        text.contains("Clamp{min: nan(0xffc00000), max: nan(0x7fc00001)}"),
        "{text}"
    );
    assert!(text.contains("Clamp{min: nan, max: 0.0}"), "{text}");
    round_trip(&g);
}

#[test]
fn the_header_versions_rlx_and_the_model() {
    let mut m = Module::new(vec![affine()]);
    m.model = Some(serde_json::json!({
        "name": "affine",
        "version": "1.2.0",
        "task": "graph",
        "labels": ["a", "b c"],
        "two words": 2.5,
    }));
    m.requires = Some("0.2.0".into());
    let text = write_module(&m).unwrap();
    assert!(
        text.contains(&format!("\nrlx {RLX_VERSION}\nrequires 0.2.0\nmodel {{")),
        "{text}"
    );
    assert!(text.contains("\"two words\": 2.5"), "{text}");

    let back = read_module(&text).unwrap();
    assert_eq!(back.rlx.as_deref(), Some(RLX_VERSION));
    assert_eq!(back.requires.as_deref(), Some("0.2.0"));
    assert_eq!(back.model, m.model);
    assert_eq!(
        (back.model_name(), back.model_version()),
        (Some("affine"), Some("1.2.0"))
    );
    assert!(same(&back.graphs[0], &affine()));
    assert_eq!(write_module(&back).unwrap(), text);

    // Written by hand: a bare word reads as a string, versions in any form.
    let m =
        read_module("rlxmodel 1\nopset 1\nrlx \"0.3.0-rc1\"\nmodel {name: e5, version: \"2\"}\n")
            .unwrap();
    assert_eq!(m.model_name(), Some("e5"));
    assert_eq!(m.rlx.as_deref(), Some("0.3.0-rc1"));
    assert!(m.graphs.is_empty());

    // A model that needs a newer rlx says so instead of loading.
    let e = read_module("rlxmodel 1\nopset 1\nrequires 99.0.0\n")
        .unwrap_err()
        .to_string();
    assert!(e.contains("requires rlx 99.0.0 or newer"), "{e}");
    let e = read_module("rlxmodel 1\nopset 1\nrequires banana\n")
        .unwrap_err()
        .to_string();
    assert!(e.contains("not a version"), "{e}");

    // Files from before the version lines read as they always did.
    let old = read_module("rlxmodel 1\nopset 1\n\ngraph @g {\n  return\n}\n").unwrap();
    assert!(old.rlx.is_none() && old.model.is_none() && old.graphs.len() == 1);
}

#[test]
fn the_model_block_carries_authors_licenses_links_and_the_full_card() {
    let card = "# Multilingual-E5-base\n\n\
                Text embeddings by \"weakly-supervised\" contrastive pre-training.\n\n\
                | lang | score |\n|---|---|\n| 日本語 | 0.7 |\n\n\
                ```python\nprint(\"query: \\\\n\")\n```\n";
    let mut m = Module::new(vec![affine()]);
    m.model = Some(serde_json::json!({
        "name": "multilingual-e5-base",
        "version": "1.0.0",
        "authors": ["Liang Wang", "Nan Yang <nan@example.org>"],
        "license": ["MIT", "CC-BY-4.0"],
        "repository": "https://github.com/microsoft/unilm",
        "homepage": "https://huggingface.co/intfloat/multilingual-e5-base",
        "card": card,
        "task": "embedding",
    }));
    let text = write_module(&m).unwrap();
    // Name first, the card last and verbatim.
    let block = text.split_once("model {\n").unwrap().1;
    assert!(
        block.starts_with("  name: \"multilingual-e5-base\",\n  version: \"1.0.0\",\n"),
        "{text}"
    );
    assert!(
        text.contains(&format!("  card: \"\"\"\n{card}\"\"\",\n}}")),
        "{text}"
    );

    let back = read_module(&text).unwrap();
    assert_eq!(back.model, m.model);
    assert_eq!(back.card(), Some(card));
    assert_eq!(back.authors(), ["Liang Wang", "Nan Yang <nan@example.org>"]);
    assert_eq!(back.licenses(), ["MIT", "CC-BY-4.0"]);
    assert_eq!(
        back.repository(),
        Some("https://github.com/microsoft/unilm")
    );
    assert_eq!(
        back.homepage(),
        Some("https://huggingface.co/intfloat/multilingual-e5-base")
    );
    assert_eq!(write_module(&back).unwrap(), text);

    // One license as a plain string; a text holding `"""` stays escaped.
    let m =
        read_module("rlxmodel 1\nopset 1\nmodel {license: \"MIT\", card: \"a\\n\\\"\\\"\\\"b\"}\n")
            .unwrap();
    assert_eq!(m.licenses(), ["MIT"]);
    assert_eq!(m.card(), Some("a\n\"\"\"b"));
    let text = write_module(&m).unwrap();
    assert!(text.contains(r#"card: "a\n\"\"\"b""#), "{text}");
    assert_eq!(read_module(&text).unwrap().card(), m.card());
}

#[test]
fn symbolic_dims_print_by_name_and_read_back() {
    use crate::Dim;
    let mut g = Graph::new("encode");
    let s = Shape::from_dims(
        &[Dim::Static(1), Dim::Dynamic(0), Dim::Static(4)],
        DType::F32,
    );
    let x = g.input("x", s.clone());
    let y = g.add_node(Op::Activation(Activation::Gelu), vec![x], s.clone());
    let z = g.add_node(
        Op::Reshape {
            new_shape: vec![-1, 4],
        },
        vec![y],
        Shape::from_dims(&[Dim::Dynamic(0), Dim::Static(4)], DType::F32),
    );
    g.set_outputs(vec![z]);
    let mut m = Module::new(vec![g]);
    m.dims = vec![("seq".into(), 0)];
    let text = write_module(&m).unwrap();
    assert!(text.contains("\ndims {seq: 0}\n"), "{text}");
    assert!(text.contains(": f32[1, seq, 4]"), "{text}");
    assert!(text.contains(": f32[seq, 4]"), "{text}");
    let back = read_module(&text).unwrap();
    assert_eq!(back.dims, m.dims);
    assert!(same(&back.graphs[0], &m.graphs[0]));
    assert_eq!(write_module(&back).unwrap(), text);
    // Unnamed symbols still print as `?n`.
    assert!(to_text(&m.graphs[0]).unwrap().contains("f32[1, ?0, 4]"));
    // A name the header does not declare.
    let e = read_module(&text.replace("dims {seq: 0}", "dims {len: 0}"))
        .unwrap_err()
        .to_string();
    assert!(e.contains("dim `seq` is not declared"), "{e}");
}

/// A dim that is a multiple of a symbol (a `[seq, 16, 128]` flattened to
/// `[seq*16, 128]`) generalizes to a symbol derived from it, prints as
/// `seq*16`, and specializes with it.
#[test]
fn derived_dims_generalize_print_and_specialize() {
    use crate::{Dim, DimBinding, dynamic};
    let flat = |seq: usize| {
        let mut g = Graph::new("flat");
        let x = g.input("x", Shape::new(&[seq, 16, 128], DType::F32));
        let pad = g.input("pad", Shape::new(&[seq + 1, 2], DType::F32));
        let rows = g.reshape(
            x,
            vec![(seq * 16) as i64, 128],
            Shape::new(&[seq * 16, 128], DType::F32),
        );
        let back = g.reshape(
            rows,
            vec![seq as i64, 2048],
            Shape::new(&[seq, 2048], DType::F32),
        );
        g.set_outputs(vec![back, pad]);
        g
    };
    let g = dynamic::generalize(&flat(37), &flat(53), &[(0, 37, 53)]).unwrap();
    let defs = g.sym_defs().to_vec();
    assert_eq!(defs.len(), 2, "{defs:?}");
    assert!(defs.iter().any(|d| (d.of, d.scale, d.offset) == (0, 16, 0)));
    assert!(defs.iter().any(|d| (d.of, d.scale, d.offset) == (0, 1, 1)));
    let mut m = Module::new(vec![g]);
    m.dims = vec![("seq".into(), 0)];
    let text = write_module(&m).unwrap();
    assert!(text.contains("f32[seq*16, 128]"), "{text}");
    assert!(text.contains("f32[seq+1, 2]"), "{text}");
    let back = read_module(&text).unwrap();
    assert_eq!(write_module(&back).unwrap(), text);
    let mut b = DimBinding::new();
    b.set(0, 10);
    let at10 = dynamic::specialize(&back.graphs[0], &b);
    assert!(at10.sym_defs().is_empty());
    assert_eq!(format!("{at10}"), format!("{}", flat(10)));
    assert_eq!(
        at10.node(crate::NodeId(2)).shape.dims(),
        [Dim::Static(160), Dim::Static(128)]
    );
    // An attention's relative positions flatten to `2·seq²`, `2·seq² +
    // seq − 1` and `2·seq² − seq` (VITS' relative-to-absolute shuffle):
    // each fits a line through two sizes too, with an offset far past its
    // scale. They are the squares, even with the offset widened.
    let square = |seq: usize| {
        let mut g = Graph::new("square");
        let dims = [2 * seq * seq, 2 * seq * seq + seq - 1, 2 * seq * seq - seq];
        let xs: Vec<_> = dims
            .iter()
            .enumerate()
            .map(|(i, d)| g.input(format!("x{i}"), Shape::new(&[*d], DType::F32)))
            .collect();
        g.set_outputs(xs);
        g
    };
    let g =
        dynamic::generalize_with(&square(37), &square(53), &[(0, 37, 53)], Some(1 << 16)).unwrap();
    let defs: Vec<_> = g
        .sym_defs()
        .iter()
        .map(|d| (d.of, d.times, d.scale, d.linear, d.offset))
        .collect();
    assert_eq!(
        defs,
        [
            (0, Some(0), 2, 0, 0),
            (0, Some(0), 2, 1, -1),
            (0, Some(0), 2, -1, 0)
        ]
    );
    let mut m = Module::new(vec![g]);
    m.dims = vec![("seq".into(), 0)];
    let text = write_module(&m).unwrap();
    for d in [
        "f32[seq*seq*2]",
        "f32[seq*seq*2+seq-1]",
        "f32[seq*seq*2-seq]",
    ] {
        assert!(text.contains(d), "{d}: {text}");
    }
    let back = read_module(&text).unwrap();
    assert_eq!(write_module(&back).unwrap(), text);
    let at10 = dynamic::specialize(&back.graphs[0], &b);
    assert_eq!(format!("{at10}"), format!("{}", square(10)));
}

#[test]
fn size_dependent_attributes_generalize_print_and_specialize() {
    use crate::{Dim, DimBinding, dynamic};
    // The last row of a `[seq, 4]` input: `start = seq - 1`.
    let tail = |seq: usize| {
        let mut g = Graph::new("tail");
        let x = g.input("x", Shape::new(&[seq, 4], DType::F32));
        let t = g.add_node(
            Op::Narrow {
                axis: 0,
                start: seq - 1,
                len: 1,
            },
            vec![x],
            Shape::new(&[1, 4], DType::F32),
        );
        g.set_outputs(vec![t]);
        g
    };
    let g = dynamic::generalize(&tail(37), &tail(53), &[(0, 37, 53)]).unwrap();
    assert_eq!(g.sym_attrs().len(), 1);
    assert_eq!(g.sym_attrs()[0].offset, -1);
    let mut m = Module::new(vec![g]);
    m.dims = vec![("seq".into(), 0)];
    let text = write_module(&m).unwrap();
    assert!(
        text.contains("Narrow{axis: 0, start: seq-1, len: 1} %x : f32[1, 4]"),
        "{text}"
    );
    assert!(text.contains("f32[seq, 4]"), "{text}");
    let back = read_module(&text).unwrap();
    assert_eq!(back.graphs[0].sym_attrs(), m.graphs[0].sym_attrs());
    assert_eq!(write_module(&back).unwrap(), text);
    let mut b = DimBinding::new();
    b.set(0, 10);
    let at10 = dynamic::specialize(&back.graphs[0], &b);
    assert!(at10.sym_attrs().is_empty());
    assert_eq!(format!("{at10}"), format!("{}", tail(10)));
    assert_eq!(
        at10.node(at10.outputs[0]).shape.dims(),
        [Dim::Static(1), Dim::Static(4)]
    );
    // Unnamed, the symbol prints as `?0`.
    assert!(to_text(&m.graphs[0]).unwrap().contains("start: ?0-1"));
}

/// An STFT's frames (`⌊n / 256⌋ + 1`), a stride-2 conv's (`⌈t / 2⌉`) and
/// the frames a decoder makes past a prompt's (`d − ⌊n / 256⌋ − 1`): no
/// line through sizes of different remainders. Through a third size each
/// is a quotient, printed `(n)/256+1`, `(t+1)/2`, `(n*-1+255)/256+d*1-1`.
#[test]
fn quotient_dims_generalize_print_and_specialize() {
    use crate::{DimBinding, dynamic};
    let at = |n: usize, t: usize, d: usize| {
        let mut g = Graph::new("frames");
        let a = g.input("a", Shape::new(&[n / 256 + 1, 8], DType::F32));
        let b = g.input("b", Shape::new(&[t.div_ceil(2), 8], DType::F32));
        let c = g.input("c", Shape::new(&[d - n / 256 - 1, 8], DType::F32));
        g.set_outputs(vec![a, b, c]);
        g
    };
    let sizes = [
        (0, 256 * 47, 256 * 59 + 255, 256 * 71 + 100),
        (1, 37, 46, 53),
        (2, 301, 333, 365),
    ];
    let build =
        |k: fn(&(u32, usize, usize, usize)) -> usize| at(k(&sizes[0]), k(&sizes[1]), k(&sizes[2]));
    let g = dynamic::generalize_checked(
        &build(|s| s.1),
        &build(|s| s.2),
        &build(|s| s.3),
        &sizes,
        Some(1 << 16),
    )
    .unwrap();
    let defs: Vec<_> = g
        .sym_defs()
        .iter()
        .map(|d| {
            (
                d.of,
                d.scale,
                d.inner,
                d.div,
                d.plus,
                d.plus_scale,
                d.offset,
            )
        })
        .collect();
    assert_eq!(
        defs,
        [
            (0, 1, 0, 256, None, 0, 1),
            (1, 1, 1, 2, None, 0, 0),
            (0, -1, 255, 256, Some(2), 1, -1)
        ]
    );
    let mut m = Module::new(vec![g]);
    m.dims = vec![("n".into(), 0), ("t".into(), 1), ("d".into(), 2)];
    let text = write_module(&m).unwrap();
    for d in [
        "f32[(n)/256+1, 8]",
        "f32[(t+1)/2, 8]",
        "f32[(n*-1+255)/256+d*1-1, 8]",
    ] {
        assert!(text.contains(d), "{d}: {text}");
    }
    let back = read_module(&text).unwrap();
    assert_eq!(write_module(&back).unwrap(), text);
    for (n, t, d) in [(24_000, 9, 500), (1000, 120, 40), (511, 1, 3)] {
        let mut b = DimBinding::new();
        b.set(0, n);
        b.set(1, t);
        b.set(2, d);
        let s = dynamic::specialize(&back.graphs[0], &b);
        assert_eq!(
            format!("{s}"),
            format!("{}", at(n, t, d)),
            "n {n}, t {t}, d {d}"
        );
    }
}

/// A relative-position table read around its middle: `start = 1000 −
/// seq`, `len = 2·seq − 1` — a start that falls as the size grows.
#[test]
fn a_scaled_attribute_generalizes_prints_and_specializes() {
    use crate::{DimBinding, dynamic};
    let around = |seq: usize| {
        let mut g = Graph::new("rel");
        let table = g.input("table", Shape::new(&[2000, 8], DType::F32));
        let t = g.add_node(
            Op::Narrow {
                axis: 0,
                start: 1000 - seq,
                len: 2 * seq - 1,
            },
            vec![table],
            Shape::new(&[2 * seq - 1, 8], DType::F32),
        );
        g.set_outputs(vec![t]);
        g
    };
    let g = dynamic::generalize(&around(37), &around(53), &[(0, 37, 53)]).unwrap();
    let a: Vec<_> = g
        .sym_attrs()
        .iter()
        .map(|a| (a.path.join("."), a.scale, a.offset))
        .collect();
    assert_eq!(
        a,
        [("start".to_string(), -1, 1000), ("len".to_string(), 2, -1)]
    );
    let mut m = Module::new(vec![g]);
    m.dims = vec![("seq".into(), 0)];
    let text = write_module(&m).unwrap();
    assert!(text.contains("start: seq*-1+1000, len: seq*2-1"), "{text}");
    let back = read_module(&text).unwrap();
    assert_eq!(back.graphs[0].sym_attrs(), m.graphs[0].sym_attrs());
    assert_eq!(write_module(&back).unwrap(), text);
    let mut b = DimBinding::new();
    b.set(0, 10);
    let at10 = dynamic::specialize(&back.graphs[0], &b);
    assert_eq!(format!("{at10}"), format!("{}", around(10)));
}

#[test]
fn values_print_by_name_where_the_name_is_a_unique_label() {
    let mut g = Graph::new("g");
    let s = Shape::new(&[2], DType::F32);
    let x = g.input("x", s.clone());
    let w = g.param("layers.0.w", s.clone());
    let a = g.add_node(Op::Binary(BinaryOp::Mul), vec![x, w], s.clone());
    g.node_mut(a).name = Some("scaled".into());
    // Two nodes with one name, and a name that is no label: numbers.
    let b = g.add_node(Op::Activation(Activation::Gelu), vec![a], s.clone());
    let c = g.add_node(Op::Activation(Activation::Gelu), vec![b], s.clone());
    g.node_mut(b).name = Some("act".into());
    g.node_mut(c).name = Some("act".into());
    let d = g.add_node(Op::Activation(Activation::Gelu), vec![c], s.clone());
    g.node_mut(d).name = Some("out put".into());
    // An input not named like its value keeps its op.
    let y = g.add_node(Op::Input { name: "y".into() }, vec![], s.clone());
    let e = g.add_node(Op::Binary(BinaryOp::Add), vec![d, y], s);
    g.set_outputs(vec![e, a]);
    let text = to_text(&g).unwrap();
    for line in [
        "  %x = input : f32[2]\n",
        "  %layers.0.w = param : f32[2]\n",
        "  %scaled = Binary(Mul) %x, %layers.0.w : f32[2]\n",
        "  %3 = Activation(Gelu) %scaled : f32[2] name \"act\"\n",
        "  %4 = Activation(Gelu) %3 : f32[2] name \"act\"\n",
        "  %5 = Activation(Gelu) %4 : f32[2] name \"out put\"\n",
        "  %6 = Input{name: \"y\"} %",
        "  return %7, %scaled\n",
    ] {
        assert!(
            text.contains(line.trim_end_matches('%')),
            "missing {line:?} in\n{text}"
        );
    }
    round_trip(&g);
}

#[test]
fn labels_are_checked_where_they_are_read() {
    let src = |body: &str| format!("rlxmodel 1\nopset 1\ngraph @g {{\n{body}\n}}");
    let err = |body: &str| from_text(&src(body)).unwrap_err().to_string();
    assert!(err("  %0 = input : f32[1]\n  return %0").contains("`input` needs a named value"));
    assert!(err("  %1 = Input{name: \"a\"} : f32[1]\n  return %1").contains("%1 is value 0 here"));
    assert!(
        err("  %a = input : f32[1]\n  %a = input : f32[1]\n  return %a")
            .contains("%a is defined twice")
    );
    assert!(err("  %a = input : f32[1]\n  return %b").contains("%b is not a value defined above"));
    // Written by hand, names and numbers mix.
    let g = from_text(&src(
        "  %ids = input : f32[4]\n  %1 = Activation(Gelu) %ids : f32[4]\n  return %1",
    ))
    .unwrap();
    assert_eq!(g.node(g.outputs[0]).inputs, [crate::NodeId(0)]);
    assert_eq!(g.node(crate::NodeId(0)).name.as_deref(), Some("ids"));
}

#[test]
fn small_constants_print_as_their_values() {
    let mut g = Graph::new("consts");
    let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let c = |g: &mut Graph, data: Vec<u8>, dims: &[usize], dt: DType| {
        g.add_node(Op::Constant { data }, vec![], Shape::new(dims, dt))
    };
    let a = c(
        &mut g,
        f32s(&[0.0, 1.5, -0.0, f32::NEG_INFINITY, f32::NAN]),
        &[5],
        DType::F32,
    );
    let b = c(
        &mut g,
        (-3i64)
            .to_le_bytes()
            .into_iter()
            .chain(7i64.to_le_bytes())
            .collect(),
        &[2],
        DType::I64,
    );
    let t = c(&mut g, vec![1, 0], &[2], DType::Bool);
    let s = c(&mut g, f32s(&[0.25]), &[], DType::F32);
    let big = c(&mut g, f32s(&[1.0; 300]), &[300], DType::F32);
    let stripped = c(&mut g, Vec::new(), &[4], DType::F32);
    let half = c(&mut g, vec![0, 0x3c], &[1], DType::BF16);
    g.set_outputs(vec![a, b, t, s, big, stripped, half]);
    let text = to_text(&g).unwrap();
    for line in [
        "%0 = const [0.0, 1.5, -0.0, -inf, nan] : f32[5]",
        "%1 = const [-3, 7] : i64[2]",
        "%2 = const [true, false] : bool[2]",
        "%3 = const 0.25 : f32[]",
        "%4 = Constant{data: b64\"",
        "%5 = Constant{data: []} : f32[4]",
        "%6 = Constant{data: b64\"",
    ] {
        assert!(text.contains(line), "missing {line:?} in\n{text}");
    }
    round_trip(&g);

    // Written by hand; the count must fit the shape.
    let src =
        "rlxmodel 1\nopset 1\ngraph @g {\n  %iota = const [0, 1, 2, 3] : f32[4]\n  return %iota\n}";
    let h = from_text(src).unwrap();
    assert_eq!(
        h.node(h.outputs[0]).op,
        Op::Constant {
            data: f32s(&[0.0, 1.0, 2.0, 3.0])
        }
    );
    let e = from_text(&src.replace("[0, 1, 2, 3]", "[0, 1]"))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("a const of 2 values for a shape of 4 elements"),
        "{e}"
    );
}

#[test]
fn variables_are_declared_and_updated_in_the_text() {
    let mut g = Graph::new("counter");
    let s = Shape::new(&[1], DType::F32);
    let n = g.input("n", s.clone());
    let total = g.add_node(Op::Input { name: "t".into() }, vec![], s.clone());
    let one = g.add_node(
        Op::Constant {
            data: 1f32.to_le_bytes().to_vec(),
        },
        vec![],
        s.clone(),
    );
    let next_n = g.add_node(Op::Binary(BinaryOp::Add), vec![n, one], s.clone());
    let next_t = g.add_node(Op::Binary(BinaryOp::Add), vec![total, n], s);
    g.node_mut(next_n).name = Some("n_next".into());
    g.set_outputs(vec![next_t, next_n]);
    g.set_vars(vec![(n, 1), (total, 0)]);
    let text = to_text(&g).unwrap();
    for line in [
        "  %n = var : f32[1]\n",
        "  %1 = Input{name: \"t\"} : f32[1] var\n",
        "  return %1 = %4, %n = %n_next\n",
    ] {
        assert!(text.contains(line), "missing {line:?} in\n{text}");
    }
    let back = from_text(&text).unwrap();
    assert_eq!(back.vars(), g.vars());
    assert_eq!(back.var_output(n), Some(1));
    round_trip(&g);
    let e = from_text(&text.replace("%n = var", "%n = input"))
        .unwrap_err()
        .to_string();
    assert!(e.contains("is not a `var`"), "{e}");

    // A variable's next value can be named too.
    g.name_output(1, "n_out");
    let text = to_text(&g).unwrap();
    assert!(text.contains("%n = %n_next as n_out\n"), "{text}");
    assert_eq!(from_text(&text).unwrap().output_index("n_out"), Some(1));
}

#[test]
fn output_names_round_trip_as_return_names() {
    let mut g = affine();
    let x = g.input_id("x").unwrap();
    let z = g.outputs[0];
    g.set_named_outputs([("logits", z), ("present.0.key", x)]);
    let text = to_text(&g).unwrap();
    assert!(
        text.contains("return %3 as logits, %x as \"present.0.key\"\n"),
        "{text}"
    );
    let back = round_trip(&g);
    assert_eq!(back.output_names(), g.output_names());
    assert_eq!(back.output_index("present.0.key"), Some(1));
    assert_eq!(back.output_name(0), Some("logits"));

    // Unnamed outputs stay unnamed; names past a shorter output list go.
    g.set_outputs(vec![z]);
    assert_eq!(g.output_names(), &[(0, "logits".to_string())]);
    let e = from_text(&text.replace("as logits", "as ,"))
        .unwrap_err()
        .to_string();
    assert!(e.contains("expected an output name after `as`"), "{e}");
}

#[test]
fn long_returns_and_model_blocks_break_across_lines() {
    let s = Shape::new(&[1], DType::F32);
    let mut g = Graph::new("wide");
    let outs: Vec<_> = (0..12)
        .map(|i| g.input(format!("a_rather_long_input_name_{i}"), s.clone()))
        .collect();
    g.set_outputs(outs);
    let mut m = Module::new(vec![g]);
    m.model = Some(serde_json::json!({
        "name": "wide",
        "entries": {"generate": {
            "functions": ["step"], "inputs": {"input_ids": "token", "position": "position"},
            "stop_ids": [151645, 151643], "tokenizer": "sidecar:tokenizer.json",
        }},
    }));
    let text = write_module(&m).unwrap();
    assert!(
        text.contains("  return\n    %a_rather_long_input_name_0,\n"),
        "{text}"
    );
    assert!(
        text.contains("  entries: {\n    generate: {\n      functions: [\"step\"],\n"),
        "{text}"
    );
    assert!(text.lines().all(|l| l.len() < 110), "{text}");
    let back = read_module(&text).unwrap();
    assert_eq!(back.model, m.model);
    assert!(same(&back.graphs[0], &m.graphs[0]));
    assert_eq!(write_module(&back).unwrap(), text);
}

/// An attribute that is a derived dim (`len: seq*4-3`) names the symbol
/// the reader gives that dim, which is past every symbol the header
/// declares — not the number `generalize` gave it, when the header declares
/// one more symbol than the graph uses.
#[test]
fn an_attribute_names_the_derived_symbol_the_reader_numbers() {
    use crate::{DimBinding, GraphExt, dynamic};
    let at = |seq: usize| {
        let mut g = Graph::new("trim");
        let x = g.input("x", Shape::new(&[1, seq * 4], DType::F32));
        let y = g.narrow_(x, 1, 0, seq * 4 - 3);
        g.set_outputs(vec![y]);
        g
    };
    let g = dynamic::generalize(&at(5), &at(7), &[(0, 5, 7)]).unwrap();
    let mut m = Module::new(vec![g]);
    m.dims = vec![("seq".into(), 0), ("batch".into(), 1)];
    let text = write_module(&m).unwrap();
    let back = read_module(&text).unwrap();
    assert_eq!(write_module(&back).unwrap(), text);
    let mut b = DimBinding::new();
    b.set(0, 10);
    let at10 = dynamic::specialize(&back.graphs[0], &b);
    assert_eq!(format!("{at10}"), format!("{}", at(10)), "{text}");
}

/// A grid's dims: an image `[h*32, w*32]`, its `h*w*4` patches, and a
/// reshape between them. `generalize` finds the product of two symbols,
/// the text writes it (`h*w*4`), and binding the image's dims alone binds
/// `h` and `w` (each line inverted), so the patches follow.
#[test]
fn a_dim_may_be_the_product_of_two_symbols() {
    use crate::{DType, Dim, DimBinding, Graph, GraphExt, Shape};
    let at = |h: usize, w: usize| {
        let mut g = Graph::new("grid");
        let img = g.input("image", Shape::new(&[h * 32, w * 32], DType::F32));
        let x = g.reshape_(img, vec![(h * 2) as i64, 16, (w * 2) as i64, 16]);
        let x = g.transpose_(x, vec![0, 2, 1, 3]);
        let patches = g.reshape_(x, vec![(h * w * 4) as i64, 256]);
        g.set_outputs(vec![patches]);
        g
    };
    let g = crate::dynamic::generalize(&at(5, 7), &at(6, 11), &[(0, 5, 6), (1, 7, 11)]).unwrap();
    assert!(
        g.sym_defs()
            .iter()
            .any(|d| (d.of, d.times, d.scale, d.offset) == (0, Some(1), 4, 0)),
        "{:?}",
        g.sym_defs()
    );
    let mut m = Module::new(vec![g]);
    m.dims = vec![("h".into(), 0), ("w".into(), 1)];
    let text = write_module(&m).unwrap();
    assert!(text.contains("f32[h*w*4, 256]"), "{text}");
    assert!(text.contains("f32[h*32, w*32]"), "{text}");
    let back = read_module(&text).unwrap();
    assert_eq!(write_module(&back).unwrap(), text);
    // The image's dims as an input binds them: its derived symbols.
    let g = &back.graphs[0];
    let image = g.node(crate::NodeId(0));
    let mut b = DimBinding::new();
    for (d, n) in image.shape.dims().iter().zip([9 * 32, 13 * 32]) {
        let Dim::Dynamic(s) = d else {
            panic!("{image:?}")
        };
        b.set(*s, n);
    }
    let at_9_13 = crate::dynamic::specialize(g, &b);
    assert_eq!(format!("{at_9_13}"), format!("{}", at(9, 13)));
}

/// Every op the IR has prints and reads back as itself: a graph of each
/// kind's sample (`sample_ops`), its inputs as many as the op takes.
#[cfg(feature = "test-support")]
#[test]
fn every_op_kind_round_trips() {
    let mut failed = Vec::new();
    for (kind, op) in crate::sample_ops::all() {
        let mut g = Graph::new("op");
        let inputs: Vec<_> = (0..op.num_inputs())
            .map(|i| g.input(format!("x{i}"), Shape::new(&[2], DType::F32)))
            .collect();
        let y = g.add_node(op, inputs, Shape::new(&[2], DType::F32));
        g.set_outputs(vec![y]);
        let text = match to_text(&g) {
            Ok(t) => t,
            Err(e) => {
                failed.push(format!("{kind:?}: does not print: {e}"));
                continue;
            }
        };
        match from_text(&text) {
            Ok(back) if same(&g, &back) => {}
            Ok(_) => failed.push(format!("{kind:?}: reads back as another graph:\n{text}")),
            Err(e) => failed.push(format!("{kind:?}: does not read back: {e}\n{text}")),
        }
    }
    assert!(
        failed.is_empty(),
        "{} op kinds:\n{}",
        failed.len(),
        failed.join("\n")
    );
}
