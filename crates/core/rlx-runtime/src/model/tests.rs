// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// A language model whose next token is `(last + 1) % vocab`, sharply.
struct CountLm;

impl LmRunner for CountLm {
    fn family(&self) -> &'static str {
        "count"
    }
    fn vocab_size(&self) -> usize {
        8
    }
    fn predict_logits(&mut self, ids: &[u32]) -> anyhow::Result<Vec<f32>> {
        let next = (ids.last().copied().unwrap_or(0) + 1) % 8;
        Ok((0..8).map(|i| if i == next { 50.0 } else { 0.0 }).collect())
    }
}

/// Held by each test that installs the global tokenizer loader, for as long
/// as it runs: tests run in parallel, and one installing `Chars` while
/// another decodes with `Letters` fails the other.
fn tokenizer_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Characters as ids, `a` = 0.
struct Letters;

impl TextCodec for Letters {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text.bytes().map(|b| (b - b'a') as u32).collect())
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        Ok(ids.iter().map(|&i| (b'a' + i as u8) as char).collect())
    }
    fn stop_ids(&self) -> Vec<u32> {
        vec![7]
    }
}

struct CountProvider;

impl ModelProvider for CountProvider {
    fn name(&self) -> &str {
        "test-count"
    }
    fn tasks(&self) -> Vec<String> {
        vec!["text-generation".into()]
    }
    fn probe(&self, request: &ModelRequest) -> u32 {
        u32::from(request.source == "count") * 50
    }
    fn open(&self, _request: &ModelRequest) -> Result<Box<dyn Model>> {
        Ok(lm_model(
            Box::new(CountLm),
            Some(Box::new(Letters)),
            ModelInfo {
                provider: "test-count".into(),
                ..Default::default()
            },
        ))
    }
}

crate::register_provider!(CountProvider);

fn ids(v: &[i32]) -> Tensor {
    Tensor::new(
        "input_ids",
        DType::I32,
        vec![v.len()],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
}

#[test]
fn open_by_probe_and_run_an_lm() {
    let mut m = open_source("count", json!({})).unwrap();
    let info = m.info();
    assert_eq!(info.task, "text-generation");
    assert_eq!(info.family, "count");
    assert_eq!(info.metadata["vocab_size"], 8);
    let logits = m.run(&[ids(&[2])], &json!({})).unwrap();
    assert_eq!(logits[0].name, "logits");
    assert_eq!(logits[0].to_f32().unwrap()[3], 50.0);
    let toks = m.run(&[ids(&[1])], &json!({"max_tokens": 3})).unwrap();
    assert_eq!(ids_of(&toks[0]).unwrap(), vec![2, 3, 4]);
    // `c` → d e f g, then the stop id 7 (`h`) ends it.
    assert_eq!(
        m.generate_text("c", &json!({"max_tokens": 10})).unwrap(),
        "defg"
    );
    // The provider itself: a plugin test may front it with its vtable copy,
    // which has no in-process `lm()`.
    let own = CountProvider.open(&ModelRequest::new("count", json!({})).unwrap());
    assert!(own.unwrap().lm().is_some());
}

#[test]
fn sampled_generation_follows_a_one_hot_distribution() {
    let mut m = open_source("count", json!({"provider": "test-count"})).unwrap();
    let toks = m
        .run(
            &[ids(&[0])],
            &json!({"max_tokens": 3, "temperature": 0.7, "seed": 5}),
        )
        .unwrap();
    assert_eq!(ids_of(&toks[0]).unwrap(), vec![1, 2, 3]);
}

#[test]
fn unknown_sources_and_providers_say_what_exists() {
    let e = open_source("nothing-claims-this", json!({})).err().unwrap();
    assert!(e.to_string().contains("no model provider claims"), "{e}");
    let e = open_source("count", json!({"provider": "nope"}))
        .err()
        .unwrap();
    assert!(e.to_string().contains("test-count"), "{e}");
    let list = providers();
    assert!(
        list.as_array()
            .unwrap()
            .iter()
            .any(|p| p["name"] == "test-count")
    );
}

#[test]
fn a_plugin_table_round_trips_through_the_c_boundary() {
    static TABLE: plugin::PluginV1 = plugin::vtable(c"test-plugin");
    let names = unsafe { plugin::register_table(&TABLE) }.unwrap();
    assert!(names.contains(&"test-count".to_string()), "{names:?}");
    // The run-time registration now fronts the static one: same answers
    // through the vtable.
    let mut m = open_source("count", json!({})).unwrap();
    assert_eq!(m.info().family, "count");
    let toks = m.run(&[ids(&[4])], &json!({"max_tokens": 2})).unwrap();
    assert_eq!(toks[0].dtype, DType::I32);
    assert_eq!(ids_of(&toks[0]).unwrap(), vec![5, 6]);
    assert_eq!(
        m.generate_text("a", &json!({"max_tokens": 3})).unwrap(),
        "bcd"
    );
    let e = m.run(&[], &json!({})).err().unwrap();
    assert!(e.to_string().contains("input_ids"), "{e}");
}

/// A package with the model but not its weights: the model block names a
/// safetensors file beside it, and each param is read from there.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn an_rlxp_package_reads_weights_it_references() {
    use rlx_ir::{Graph, Shape};
    let mut g = Graph::new("affine");
    let x = g.input("x", Shape::new(&[1, 2], DType::F32));
    let w = g.param("w", Shape::new(&[2, 2], DType::F32));
    let b = g.param("b", Shape::new(&[1, 2], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[1, 2], DType::F32));
    let y = g.binary(
        rlx_ir::op::BinaryOp::Add,
        y,
        b,
        Shape::new(&[1, 2], DType::F32),
    );
    g.set_outputs(vec![y]);
    let dir = std::env::temp_dir().join(format!("rlx-model-ref-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // The weights under the checkpoint's own names.
    let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let (wb, bb) = (bytes(&[2.0, 0.0, 0.0, 3.0]), bytes(&[10.0, 20.0]));
    fn view(shape: Vec<usize>, data: &[u8]) -> safetensors::tensor::TensorView<'_> {
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, data).unwrap()
    }
    let st = safetensors::serialize(
        vec![
            ("lin.weight".to_string(), view(vec![2, 2], &wb)),
            ("lin.b".to_string(), view(vec![1, 2], &bb)),
        ],
        None,
    )
    .unwrap();
    std::fs::write(dir.join("ckpt.safetensors"), st).unwrap();

    let path = dir.join("affine.rlxp");
    let model = |uri: &str| {
        json!({"name": "affine", "version": "1.0.0", "task": "graph", "outputs": ["y"],
               "weights": [{"uri": uri, "prefix": "lin.", "map": {"w": "lin.weight"}}]})
    };
    rlxp::save_package(&path, &g, &[], &model("safetensors://ckpt.safetensors")).unwrap();
    let mut m = open_source(path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    let out = m
        .run(&[Tensor::f32("x", vec![1, 2], &[1.0, 1.0])], &json!({}))
        .unwrap();
    assert_eq!(out[0].to_f32().unwrap(), vec![12.0, 23.0]);

    // A reference that leaves a param out names it rather than running on zeros.
    rlxp::save_package(
        &path,
        &g,
        &[],
        &model("safetensors://elsewhere.safetensors"),
    )
    .unwrap();
    let e = open_source(path.to_str().unwrap(), json!({"device": "cpu"}))
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("no weights for 2 param(s)"), "{e}");
    assert!(e.contains("w, b") || e.contains("b, w"), "{e}");

    // A reference with the file's size and SHA-256: another file of another
    // size is refused on open; one of the same size, on `verify_weights`.
    let ckpt = dir.join("ckpt.safetensors");
    let mut source = rlxp::weight_source(
        "safetensors",
        "ckpt.safetensors",
        &ckpt,
        json!({"w": "lin.weight"}),
    )
    .unwrap();
    source["prefix"] = json!("lin.");
    let checked = json!({"name": "affine", "task": "graph", "weights": [source]});
    rlxp::save_package(&path, &g, &[], &checked).unwrap();
    let verify = json!({"device": "cpu", "verify_weights": true});
    open_source(path.to_str().unwrap(), verify.clone()).unwrap();
    let mut bytes_now = std::fs::read(&ckpt).unwrap();
    let last = bytes_now.len() - 1;
    bytes_now[last] ^= 1;
    std::fs::write(&ckpt, &bytes_now).unwrap();
    open_source(path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    let e = open_source(path.to_str().unwrap(), verify)
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("SHA-256"), "{e}");
    bytes_now.push(0);
    std::fs::write(&ckpt, &bytes_now).unwrap();
    let e = open_source(path.to_str().unwrap(), json!({"device": "cpu"}))
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("changed under it"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A build for a device adds `<function>:<device>` beside the portable
/// function, and a model opened on that device runs it: here Metal's f16
/// weight rounds 1/3 where the CPU's f32 one does not. The build is written
/// elsewhere, so its referenced weights are found from there.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn a_build_for_a_device_runs_on_that_device_only() {
    use rlx_ir::{Graph, Shape};
    let mut g = Graph::new("affine");
    let x = g.input("x", Shape::new(&[1, 1], DType::F32));
    let w = g.param("w", Shape::new(&[1, 1], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[1, 1], DType::F32));
    g.set_outputs(vec![y]);
    let dir = std::env::temp_dir().join(format!("rlx-build-target-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("ckpt")).unwrap();
    std::fs::create_dir_all(dir.join("built")).unwrap();
    let third = (1.0f32 / 3.0).to_le_bytes();
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1, 1], &third).unwrap();
    let st = safetensors::serialize(vec![("w".to_string(), view)], None).unwrap();
    std::fs::write(dir.join("ckpt/model.safetensors"), st).unwrap();
    let portable = dir.join("affine.rlxp");
    let model = json!({"name": "affine", "task": "graph", "outputs": ["y"],
                       "weights": [{"uri": "safetensors://ckpt/model.safetensors"}]});
    rlxp::save_package(&portable, &g, &[], &model).unwrap();

    let built = dir.join("built/affine.rlxp");
    let recipe = rlxp::TargetRecipe::for_target("metal").unwrap();
    let report = rlxp::build_for_target(&portable, &built, "metal", &recipe).unwrap();
    assert_eq!(report.built, ["affine:metal"]);
    assert_eq!(report.weights_narrowed, 1);
    let module = rlx_pkg::Package::open(&built).unwrap().module().unwrap();
    let names: Vec<&str> = module.graphs.iter().map(|g| g.name.as_str()).collect();
    assert_eq!(names, ["affine", "affine:metal"]);
    assert_eq!(
        module.model.unwrap()["weights"][0]["uri"],
        "safetensors://../ckpt/model.safetensors"
    );

    let run = |device: &str| {
        let mut m = open_source(built.to_str().unwrap(), json!({"device": device})).unwrap();
        m.run(&[Tensor::f32("x", vec![1, 1], &[1.0])], &json!({}))
            .unwrap()[0]
            .to_f32()
            .unwrap()[0]
    };
    assert_eq!(run("cpu"), 1.0 / 3.0);
    if cfg!(feature = "metal") && crate::device_ext::is_available(crate::Device::Metal) {
        let f16_third = half::f16::from_f32(1.0 / 3.0).to_f32();
        assert_eq!(run("metal"), f16_third);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A param declared f16 binds from a bf16 checkpoint tensor in its own
/// dtype: converted once, two bytes an element, on every backend.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn a_referenced_weight_binds_in_the_dtype_its_param_declares() {
    use rlx_ir::{Graph, Shape};
    let mut g = Graph::new("affine");
    let x = g.input("x", Shape::new(&[1, 2], DType::F32));
    let w = g.param("w", Shape::new(&[2, 2], DType::F16));
    let y = g.matmul(x, w, Shape::new(&[1, 2], DType::F32));
    g.set_outputs(vec![y]);
    let dir = std::env::temp_dir().join(format!("rlx-model-ref-f16-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // bf16 is an f32's top half; these values are exact in both.
    let bf16: Vec<u8> = [1.5f32, -0.25, 0.0, 3.0]
        .iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2, 2], &bf16).unwrap();
    let st = safetensors::serialize(vec![("w".to_string(), view)], None).unwrap();
    std::fs::write(dir.join("ckpt.safetensors"), st).unwrap();
    let path = dir.join("affine.rlxp");
    let model = json!({"name": "affine", "task": "graph", "outputs": ["y"],
                       "weights": [{"uri": "safetensors://ckpt.safetensors"}]});
    rlxp::save_package(&path, &g, &[], &model).unwrap();
    let mut devices = vec!["cpu"];
    if cfg!(feature = "metal") && crate::device_ext::is_available(crate::Device::Metal) {
        devices.push("metal");
    }
    for device in devices {
        let mut m = open_source(path.to_str().unwrap(), json!({"device": device})).unwrap();
        let out = m
            .run(&[Tensor::f32("x", vec![1, 2], &[1.0, 1.0])], &json!({}))
            .unwrap();
        assert_eq!(out[0].to_f32().unwrap(), vec![1.5, 2.75], "{device}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn an_rlxp_package_runs_as_a_model() {
    use rlx_ir::{Graph, Shape};
    let mut g = Graph::new("affine");
    let x = g.input("x", Shape::new(&[1, 2], DType::F32));
    let w = g.param("w", Shape::new(&[2, 2], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[1, 2], DType::F32));
    g.set_outputs(vec![y]);
    let dir = std::env::temp_dir().join(format!("rlx-model-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("affine.rlxp");
    rlxp::save_package(
        &path,
        &g,
        &[Tensor::f32("w", vec![2, 2], &[2.0, 0.0, 0.0, 3.0])],
        &json!({"name": "affine", "version": "1.0.0", "task": "graph", "family": "affine",
                "outputs": ["y"], "authors": ["A. Author"], "license": "MIT",
                "repository": "https://github.com/example/affine", "card": "# Affine\n\nx·w\n"}),
    )
    .unwrap();
    let mut m = open_source(path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    let info = m.info();
    assert_eq!(
        (info.provider.as_str(), info.family.as_str()),
        ("rlxp", "affine")
    );
    assert_eq!(info.inputs[0].name, "x");
    assert_eq!(info.inputs[0].shape, vec![Some(1), Some(2)]);
    // The model block of the stored `.rlxmodel` header.
    assert_eq!(info.metadata["model_name"], "affine");
    assert_eq!(info.metadata["model_version"], "1.0.0");
    assert_eq!(info.metadata["rlx_version"], rlx_ir::rlxmodel::RLX_VERSION);
    assert_eq!(info.metadata["functions"], json!(["affine"]));
    assert_eq!(info.metadata["authors"], json!(["A. Author"]));
    assert_eq!(info.metadata["license"], "MIT");
    assert_eq!(
        info.metadata["repository"],
        "https://github.com/example/affine"
    );
    assert_eq!(info.metadata["card"], "# Affine\n\nx·w\n");
    let out = m
        .run(&[Tensor::f32("x", vec![1, 2], &[1.0, 1.0])], &json!({}))
        .unwrap();
    assert_eq!(out[0].name, "y");
    assert_eq!(out[0].to_f32().unwrap(), vec![2.0, 3.0]);
    let e = m.run(&[], &json!({})).err().unwrap();
    assert!(e.to_string().contains("missing input 'x'"), "{e}");
    std::fs::remove_dir_all(&dir).ok();
}

// ── tasks ───────────────────────────────────────────────────

/// "Transcribes" audio as the count of positive samples.
struct Counter;

impl SpeechToText for Counter {
    fn family(&self) -> &str {
        "counter"
    }
    fn transcribe(&mut self, audio: &[f32], sample_rate: u32, _: &Value) -> Result<String> {
        let n = audio.iter().filter(|&&v| v > 0.0).count();
        Ok(format!("{n} at {sample_rate}"))
    }
}

/// "Speaks" one sample per character: its byte value / 100.
struct Beeper;

impl TextToSpeech for Beeper {
    fn family(&self) -> &str {
        "beeper"
    }
    fn sample_rate(&self) -> u32 {
        8000
    }
    fn synthesize(&mut self, text: &str, _: &Value) -> Result<Vec<f32>> {
        Ok(text.bytes().map(|b| b as f32 / 100.0).collect())
    }
}

/// Embeds a text as `[len, vowels]`.
struct Shape;

impl TextEmbedding for Shape {
    fn family(&self) -> &str {
        "shape"
    }
    fn dim(&self) -> usize {
        2
    }
    fn embed(&mut self, texts: &[&str], _: &Value) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let vowels = t.chars().filter(|c| "aeiou".contains(*c)).count();
                vec![t.len() as f32, vowels as f32]
            })
            .collect())
    }
}

/// Scores a passage by the words it shares with the query.
struct Overlap;

impl TextRanking for Overlap {
    fn family(&self) -> &str {
        "overlap"
    }
    fn score(&mut self, query: &str, passages: &[&str], _: &Value) -> Result<Vec<f32>> {
        Ok(passages
            .iter()
            .map(|p| {
                p.split_whitespace()
                    .filter(|w| query.split_whitespace().any(|q| q == *w))
                    .count() as f32
            })
            .collect())
    }
}

static RANK: Provider = Provider::new("test-rank", tasks::TEXT_RANKING, |req| {
    Ok(text_ranking_model(Overlap, req.model_info()))
})
.claims(&[Claim::Id("overlap")]);
crate::register_provider!(RANK);

#[test]
fn a_ranking_model_scores_through_its_method_and_through_run() {
    let mut m = open_source("overlap", json!({})).unwrap();
    assert_eq!(m.info().task, "text-ranking");
    let passages = ["the cat sat", "a dog ran", "the cat and the hat"];
    let scores = m.score("the cat", &passages, &json!({})).unwrap();
    assert_eq!(scores, [2.0, 0.0, 3.0]);
    let mut inputs = vec![Tensor::text("query", "the cat")];
    inputs.extend(passages.iter().map(|p| Tensor::text("passage", p)));
    let outs = m.run(&inputs, &json!({})).unwrap();
    assert_eq!(
        (outs[0].name.as_str(), outs[0].to_f32().unwrap()),
        ("scores", scores)
    );
    // Other tasks' methods say what this one takes.
    let e = m.embed(&["x"], &json!({})).unwrap_err().to_string();
    assert!(e.contains("use score"), "{e}");
    // And `score` on another task's model says what that one takes.
    let mut embed = open_source("shape", json!({})).unwrap();
    let e = embed
        .score("q", &["p"], &json!({}))
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("needs a text-ranking model") && e.contains("use embed"),
        "{e}"
    );
}

static STT: Provider = Provider::new("test-stt", tasks::SPEECH_RECOGNITION, |req| {
    Ok(speech_to_text_model(Counter, req.model_info()))
})
.claims(&[Claim::Id("counter"), Claim::Extension("wavmodel")]);
crate::register_provider!(STT);

static TTS: Provider = Provider::new("test-tts", tasks::TEXT_TO_SPEECH, |req| {
    Ok(text_to_speech_model(Beeper, req.model_info()))
})
.claims(&[Claim::Prefix("beeper:")]);
crate::register_provider!(TTS);

static EMBED: Provider = Provider::new("test-embed", tasks::EMBEDDING, |req| {
    Ok(text_embedding_model(Shape, req.model_info()))
})
.claims(&[Claim::Id("shape")]);
crate::register_provider!(EMBED);

#[test]
fn each_task_runs_through_its_method_and_through_run() {
    let mut stt = open_source("counter", json!({})).unwrap();
    let info = stt.info();
    assert_eq!(
        (info.provider.as_str(), info.task.as_str()),
        ("test-stt", "speech-recognition")
    );
    assert_eq!(info.metadata["sample_rate"], 16_000);
    assert_eq!(
        stt.transcribe(&[1.0, -1.0, 2.0], 22_050, &json!({}))
            .unwrap(),
        "2 at 22050"
    );
    // The same through `run`, as a plugin's model is driven.
    let outs = stt
        .run(&[Tensor::f32("audio", vec![2], &[1.0, 1.0])], &json!({}))
        .unwrap();
    assert_eq!(outs[0].to_text().unwrap(), "2 at 16000");

    let mut tts = open_source("beeper:any", json!({})).unwrap();
    let audio = tts.synthesize("dd", &json!({})).unwrap();
    assert_eq!((audio.samples, audio.sample_rate), (vec![1.0, 1.0], 8000));

    let mut e = open_source("shape", json!({})).unwrap();
    assert_eq!(
        e.embed(&["hello", "sky"], &json!({})).unwrap(),
        vec![vec![5.0, 2.0], vec![3.0, 0.0]]
    );
    assert!(e.embed(&[], &json!({})).unwrap().is_empty());
}

#[test]
fn a_method_on_the_wrong_kind_of_model_says_what_to_use() {
    let mut stt = open_source("counter", json!({})).unwrap();
    let err = stt.synthesize("hi", &json!({})).err().unwrap().to_string();
    assert!(
        err.contains("synthesize needs a text-to-speech model"),
        "{err}"
    );
    assert!(err.contains("use transcribe instead"), "{err}");
    let err = stt
        .chat(&[ChatMessage::user("hi")], &json!({}))
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("does not generate text"), "{err}");
    let mut lm = open_source("count", json!({})).unwrap();
    let err = lm
        .transcribe(&[0.0], 16_000, &json!({}))
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("use chat / generate_text / generate"), "{err}");
}

#[test]
fn task_models_cross_the_plugin_c_boundary() {
    static TABLE: plugin::PluginV1 = plugin::vtable(c"test-task-plugin");
    unsafe { plugin::register_table(&TABLE) }.unwrap();
    // Now served through the vtable: the task methods go over `run`.
    let mut stt = open_source("counter", json!({})).unwrap();
    assert_eq!(
        stt.transcribe(&[3.0], 8000, &json!({})).unwrap(),
        "1 at 8000"
    );
    let mut tts = open_source("beeper:x", json!({})).unwrap();
    assert_eq!(tts.synthesize("d", &json!({})).unwrap().sample_rate, 8000);
    let mut e = open_source("shape", json!({})).unwrap();
    assert_eq!(e.embed(&["aa"], &json!({})).unwrap(), vec![vec![2.0, 2.0]]);
}

/// Letters with a chat template: `role:content;` per turn.
struct ChatLetters;

impl TextCodec for ChatLetters {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Letters.encode(&text.replace([':', ';'], ""))
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        Letters.decode(ids)
    }
    fn stop_ids(&self) -> Vec<u32> {
        vec![7]
    }
    fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        Ok(messages
            .iter()
            .map(|m| format!("{}:{};", &m.role[..1], m.content))
            .collect())
    }
}

#[test]
fn chat_applies_the_tokenizer_template() {
    // `u:c;` encodes as `u c`: the reply counts on from `c`.
    let mut m = lm_model(
        Box::new(CountLm),
        Some(Box::new(ChatLetters)),
        ModelInfo::default(),
    );
    assert_eq!(
        m.chat(&[ChatMessage::user("c")], &json!({"max_tokens": 9}))
            .unwrap(),
        "defg"
    );
    // A codec without a template refuses rather than guessing a format.
    let mut plain = open_source("count", json!({})).unwrap();
    let err = plain
        .chat(&[ChatMessage::user("c")], &json!({}))
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("no chat template"), "{err}");
}

#[test]
fn claims_read_gguf_architecture_and_hf_model_type() {
    let dir = std::env::temp_dir().join(format!("rlx-claims-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // A GGUF header: two keys, the architecture second (after a string array).
    let mut g = Vec::new();
    g.extend(b"GGUF");
    g.extend(3u32.to_le_bytes());
    g.extend(0u64.to_le_bytes());
    g.extend(2u64.to_le_bytes());
    let key = |g: &mut Vec<u8>, k: &str| {
        g.extend((k.len() as u64).to_le_bytes());
        g.extend(k.as_bytes());
    };
    key(&mut g, "tokenizer.ggml.tokens");
    g.extend(9u32.to_le_bytes());
    g.extend(8u32.to_le_bytes());
    g.extend(2u64.to_le_bytes());
    for t in ["a", "bc"] {
        g.extend((t.len() as u64).to_le_bytes());
        g.extend(t.as_bytes());
    }
    key(&mut g, "general.architecture");
    g.extend(8u32.to_le_bytes());
    g.extend(7u64.to_le_bytes());
    g.extend(b"whisper");
    let gguf = dir.join("m.gguf");
    std::fs::write(&gguf, g).unwrap();
    assert_eq!(tasks::gguf_architecture(&gguf).as_deref(), Some("whisper"));
    let req = ModelRequest::new(gguf.to_str().unwrap(), json!({})).unwrap();
    assert_eq!(Claim::GgufArch("whisper").score(&req), 70);
    assert_eq!(Claim::GgufArch("qwen3").score(&req), 0);

    std::fs::write(dir.join("config.json"), r#"{"model_type": "kokoro"}"#).unwrap();
    let req = ModelRequest::new(dir.to_str().unwrap(), json!({})).unwrap();
    assert_eq!(Claim::HfModelType("kokoro").score(&req), 70);
    std::fs::remove_dir_all(&dir).ok();
}

/// A package's `embed` entry, run by rlx alone: tokenize with the package's
/// tokenizer, run the smallest function that fits, pool.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn a_package_embed_entry_tokenizes_buckets_and_pools() {
    use rlx_ir::{Graph, Shape};
    // `hidden[s] = id[s] · w`, at one [batch, seq] shape.
    fn encoder(batch: usize, seq: usize) -> Graph {
        let mut g = Graph::new(format!("encode_{seq}"));
        let f = |d: &[usize]| Shape::new(d, DType::F32);
        let ids = g.input("input_ids", f(&[batch, seq]));
        let w = g.param("w", f(&[1, 2]));
        let col = g.reshape(ids, vec![(batch * seq) as i64, 1], f(&[batch * seq, 1]));
        let h = g.matmul(col, w, f(&[batch * seq, 2]));
        let h = g.reshape(h, vec![batch as i64, seq as i64, 2], f(&[batch, seq, 2]));
        g.set_outputs(vec![h]);
        g
    }
    // Package tokenizers load through the installed loader; here, letters.
    let _loader = tokenizer_lock();
    set_tokenizer_loader(|_bytes, _chat| Ok(Box::new(Letters)));
    let dir = std::env::temp_dir().join(format!("rlx-embed-entry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let path = dir.join("enc.rlxp");
    let model = json!({
        "name": "letters-encoder", "version": "0.1.0", "family": "letters",
        "entries": {"embed": {
            "functions": ["encode_8", "encode_4"],
            "tokenizer": "tokenizer.json",
            "inputs": {"input_ids": "token_ids"},
            "pooling": "mean",
            "normalize": false,
        }},
    });
    rlxp::save_module(
        &path,
        &[encoder(2, 4), encoder(2, 8)],
        &[Tensor::f32("w", vec![1, 2], &[1.0, 10.0])],
        &model,
    )
    .unwrap();

    let mut m = open_source(path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    let info = m.info();
    assert_eq!(
        (info.task.as_str(), info.family.as_str()),
        ("embedding", "letters")
    );
    assert_eq!(info.metadata["functions"], json!(["encode_4", "encode_8"]));
    // "bd" = ids 1, 3 → mean 2 → [2, 20]; "cccccc" fits only seq 8;
    // "bbbbbbbbbd" (10 ids) is cut to 8, keeping the last: mean (7·1+3)/8.
    let v = m
        .embed(&["bd", "cccccc", "e", "bbbbbbbbbd"], &json!({}))
        .unwrap();
    let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5);
    assert!(close(&v[0], &[2.0, 20.0]), "{v:?}");
    assert!(close(&v[1], &[2.0, 20.0]), "{v:?}");
    assert!(close(&v[2], &[4.0, 40.0]), "{v:?}");
    assert!(close(&v[3], &[1.25, 12.5]), "{v:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A package's `generate` entry: decode functions, a cache carried from
/// step to step, a table row per position, the package's stop id.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn a_package_generate_entry_decodes_with_its_cache() {
    use rlx_ir::{Graph, Shape};
    // Next id = id + 1 (a one-hot table); the cache gains the position's
    // table row each step.
    fn decode(upper: usize) -> Graph {
        let mut g = Graph::new(format!("decode_{upper}"));
        let f = |d: &[usize]| Shape::new(d, DType::F32);
        let ids = g.input("input_ids", Shape::new(&[1], DType::I64));
        let mask = g.input("mask", f(&[1, upper + 1]));
        let pos = g.input("pos", f(&[1, 1]));
        let past = g.input("past_k_0", f(&[upper, 1]));
        let table = g.param("next", f(&[8, 8]));
        let logits = g.gather(table, ids, 0, f(&[1, 8]));
        let cache = g.concat(vec![past, pos], 0, f(&[upper + 1, 1]));
        g.set_outputs(vec![logits, cache, mask]);
        g
    }
    let _loader = tokenizer_lock();
    set_tokenizer_loader(|_bytes, _chat| Ok(Box::new(Letters)));
    let dir = std::env::temp_dir().join(format!("rlx-generate-entry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let path = dir.join("lm.rlxp");
    let next: Vec<f32> = (0..8)
        .flat_map(|i| (0..8).map(move |j| if j == (i + 1) % 8 { 1.0 } else { 0.0 }))
        .collect();
    let positions: Vec<f32> = (0..32).map(|p| p as f32).collect();
    let model = json!({
        "name": "counting-lm", "version": "0.1.0", "family": "count",
        "entries": {"generate": {
            "functions": ["decode_16", "decode_4"],
            "tokenizer": "tokenizer.json",
            "stop_ids": [7],
            "vocab_size": 8,
            "inputs": {"input_ids": "token", "mask": "decode_mask", "pos": {"table": "positions"}},
            "state": [["past_k_0", 1]],
            "logits": 0,
        }},
    });
    rlxp::save_module(
        &path,
        &[decode(4), decode(16)],
        &[
            Tensor::f32("next", vec![8, 8], &next),
            Tensor::f32("positions", vec![32, 1], &positions),
        ],
        &model,
    )
    .unwrap();

    let mut m = open_source(path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    assert_eq!(m.info().task, "text-generation");
    // `c` → d e f g, then the package's stop id 7 (`h`) ends it.
    assert_eq!(
        m.generate_text("c", &json!({"max_tokens": 10})).unwrap(),
        "defg"
    );
    // Five prompt tokens outgrow the 4-row function: the 16-row one runs on.
    assert_eq!(
        m.generate_text("abcab", &json!({"max_tokens": 1})).unwrap(),
        "c"
    );
    let e = m.chat(&[ChatMessage::user("hi")], &json!({})).unwrap_err();
    assert!(e.to_string().contains("no chat template"), "{e}");
    let long = "a".repeat(20);
    let e = m
        .generate_text(&long, &json!({"max_tokens": 1}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("the context is full"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A symbolic step listed at one short size grows with the context: past
/// 4 tokens it is specialized at 9, then 19, up to the entry's
/// `max_context`.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn a_symbolic_step_grows_with_the_context_to_max_context() {
    use rlx_ir::{Graph, Shape};
    fn decode(upper: usize) -> Graph {
        let mut g = Graph::new("decode");
        let f = |d: &[usize]| Shape::new(d, DType::F32);
        let ids = g.input("input_ids", Shape::new(&[1], DType::I64));
        let mask = g.input("mask", f(&[1, upper + 1]));
        let pos = g.input("pos", f(&[1, 1]));
        let past = g.input("past_k_0", f(&[upper, 1]));
        let table = g.param("next", f(&[8, 8]));
        let logits = g.gather(table, ids, 0, f(&[1, 8]));
        let cache = g.concat(vec![past, pos], 0, f(&[upper + 1, 1]));
        g.set_outputs(vec![logits, cache, mask]);
        g
    }
    let _loader = tokenizer_lock();
    set_tokenizer_loader(|_bytes, _chat| Ok(Box::new(Letters)));
    let g = rlx_ir::dynamic::generalize(&decode(36), &decode(52), &[(0, 36, 52)]).unwrap();
    let dir = std::env::temp_dir().join(format!("rlx-grow-entry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let path = dir.join("lm.rlxp");
    let next: Vec<f32> = (0..8)
        .flat_map(|i| (0..8).map(move |j| if j == (i + 1) % 8 { 1.0 } else { 0.0 }))
        .collect();
    let positions: Vec<f32> = (0..32).map(|p| p as f32).collect();
    let mut module = rlx_ir::rlxmodel::Module::new(vec![g]);
    module.dims = vec![("upper".into(), 0)];
    module.model = Some(json!({
        "name": "counting-lm", "family": "count",
        "entries": {"generate": {
            "functions": ["decode"],
            "sizes": [{"upper": 4}],
            "max_context": 20,
            "tokenizer": "tokenizer.json",
            "vocab_size": 8,
            "inputs": {"input_ids": "token", "mask": "decode_mask", "pos": {"table": "positions"}},
            "state": [["past_k_0", 1]],
            "logits": 0,
        }},
    }));
    rlxp::save_package_module(
        &path,
        &module,
        &[
            Tensor::f32("next", vec![8, 8], &next),
            Tensor::f32("positions", vec![32, 1], &positions),
        ],
        &[],
    )
    .unwrap();
    let mut m = open_source(path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    // 12 prompt tokens then 3: past 4, the step at 9, then at 19.
    let prompt = "a".repeat(11) + "c";
    assert_eq!(
        m.generate_text(&prompt, &json!({"max_tokens": 3})).unwrap(),
        "def"
    );
    let e = m
        .generate_text(&"a".repeat(25), &json!({"max_tokens": 1}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("decodes up to 20"), "{e}");
    // Open options: a lower `max_context`, and steps grown 4× at a time.
    let mut m = open_source(
        path.to_str().unwrap(),
        json!({"device": "cpu", "max_context": 10}),
    )
    .unwrap();
    let e = m
        .generate_text(&prompt, &json!({"max_tokens": 1}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("decodes up to 10"), "{e}");
    let mut m = open_source(
        path.to_str().unwrap(),
        json!({"device": "cpu", "context_growth": 4}),
    )
    .unwrap();
    assert_eq!(
        m.generate_text(&prompt, &json!({"max_tokens": 3})).unwrap(),
        "def"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Prompts of different lengths decode together through one step sized
/// for them all: next id = (the sum of the sequence's ids so far + 1) mod
/// 8, so every cached row of its own sequence matters. The batch gives what
/// each prompt gives alone, each stopping at the tokenizer's stop id.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn prompts_decode_together_as_each_does_alone() {
    use rlx_ir::{Graph, Op, Shape};
    fn decode(upper: usize, batch: usize) -> Graph {
        let mut g = Graph::new("decode");
        let f = |d: &[usize]| Shape::new(d, DType::F32);
        let ids = g.input("input_ids", Shape::new(&[batch], DType::I64));
        let mask = g.input("mask", f(&[batch, upper + 1]));
        let past = g.input("past_k_0", f(&[batch, upper, 1]));
        let table = g.param("next", f(&[256, 8]));
        let id = g.add_node(Op::Cast { to: DType::F32 }, vec![ids], f(&[batch]));
        let row = g.reshape(id, vec![batch as i64, 1, 1], f(&[batch, 1, 1]));
        let cache = g.concat(vec![past, row], 1, f(&[batch, upper + 1, 1]));
        let sum = g.reduce(
            cache,
            rlx_ir::op::ReduceOp::Sum,
            vec![1, 2],
            false,
            f(&[batch]),
        );
        let at = g.add_node(
            Op::Cast { to: DType::I64 },
            vec![sum],
            Shape::new(&[batch], DType::I64),
        );
        let logits = g.gather(table, at, 0, f(&[batch, 8]));
        g.set_outputs(vec![logits, cache, mask]);
        g
    }
    let _loader = tokenizer_lock();
    set_tokenizer_loader(|_bytes, _chat| Ok(Box::new(Letters)));
    let g = rlx_ir::dynamic::generalize(&decode(36, 5), &decode(52, 7), &[(0, 36, 52), (1, 5, 7)])
        .unwrap();
    let dir = std::env::temp_dir().join(format!("rlx-batch-entry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let path = dir.join("lm.rlxp");
    let next: Vec<f32> = (0..256)
        .flat_map(|i| (0..8).map(move |j| if j == (i + 1) % 8 { 1.0 } else { 0.0 }))
        .collect();
    let mut module = rlx_ir::rlxmodel::Module::new(vec![g]);
    module.dims = vec![("upper".into(), 0), ("batch".into(), 1)];
    module.model = Some(json!({
        "name": "summing-lm", "family": "sum",
        "entries": {"generate": {
            "functions": ["decode"],
            "sizes": [{"upper": 8, "batch": 1}, {"upper": 32, "batch": 1}],
            "batch": "batch",
            "tokenizer": "tokenizer.json",
            "vocab_size": 8,
            "inputs": {"input_ids": "token", "mask": "decode_mask"},
            "state": [["past_k_0", 1]],
            "logits": 0,
        }},
    }));
    rlxp::save_package_module(
        &path,
        &module,
        &[Tensor::f32("next", vec![256, 8], &next)],
        &[],
    )
    .unwrap();
    let mut m = open_source(path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    let prompts = ["a", "cab", "bbbbbbb", "ggd", "g"];
    let opts = json!({"max_tokens": 6});
    let alone: Vec<String> = prompts
        .iter()
        .map(|p| m.generate_text(p, &opts).unwrap())
        .collect();
    let together = m.generate_batch(&prompts, &opts).unwrap();
    assert_eq!(together, alone);
    // The batch ran in the batched step: past its context it says so in
    // its own words (one by one would say "decodes up to").
    let long = "a".repeat(30);
    let e = m
        .generate_batch(&[long.as_str(), "b"], &json!({"max_tokens": 8}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("for the longest sequence"), "{e}");
    // Different lengths, some stopping early at `h`.
    assert!(alone.iter().any(|t| t.len() < 6), "{alone:?}");
    assert!(alone.iter().any(|t| t.len() == 6), "{alone:?}");

    // Beam search: the hypotheses as the batch's sequences, against the
    // same package without its `batch` symbol (each hypothesis extended on
    // its own). The batched path ran: past its context it says so.
    let mut entry = module.model.clone().unwrap();
    entry["entries"]["generate"]
        .as_object_mut()
        .unwrap()
        .remove("batch");
    let mut plain = module.clone();
    plain.model = Some(entry);
    let plain_path = dir.join("plain.rlxp");
    rlxp::save_package_module(
        &plain_path,
        &plain,
        &[Tensor::f32("next", vec![256, 8], &next)],
        &[],
    )
    .unwrap();
    let mut one_by_one =
        open_source(plain_path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    for beams in [2, 3, 4] {
        let opts = json!({"max_tokens": 6, "num_beams": beams});
        for p in ["a", "cab", "bbd"] {
            assert_eq!(
                m.generate_text(p, &opts).unwrap(),
                one_by_one.generate_text(p, &opts).unwrap(),
                "{beams} beams after {p:?}"
            );
        }
    }
    let e = m
        .generate_text(&"a".repeat(30), &json!({"max_tokens": 8, "num_beams": 2}))
        .unwrap_err()
        .to_string();
    assert!(e.contains("for the longest sequence"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A stateful step (its cache a variable) keeps the cache in the
/// executable, and carries it into a longer step when the context outgrows
/// the short one.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn a_stateful_step_keeps_its_cache_across_steps_and_functions() {
    use rlx_ir::op::{CmpOp, ReduceOp, ScatterNdReduction};
    use rlx_ir::{Graph, Op, Shape};
    // Next id = 1 + the sum of the ids at positions `0..=position`, mod 8:
    // every cache row matters, and rows past the position must not.
    fn step(cap: usize) -> Graph {
        let mut g = Graph::new(format!("step_{cap}"));
        let f = |d: &[usize]| Shape::new(d, DType::F32);
        let ids = g.input("input_ids", Shape::new(&[1], DType::I64));
        let pos = g.input("position", f(&[1]));
        let cache = g.input("cache", f(&[1, cap, 1]));
        let table = g.param("next", f(&[128, 8]));
        let iota = g.param(format!("iota_{cap}"), f(&[cap]));
        let value = g.add_node(Op::Cast { to: DType::F32 }, vec![ids], f(&[1]));
        let update = g.reshape(value, vec![1, 1], f(&[1, 1]));
        let zero = Op::Constant {
            data: 0f32.to_le_bytes().to_vec(),
        };
        let zero = g.add_node(zero, vec![], f(&[1]));
        let pair = g.concat(vec![zero, pos], 0, f(&[2]));
        let index = g.reshape(pair, vec![1, 2], f(&[1, 2]));
        let next = g.scatter_nd(cache, index, update, ScatterNdReduction::None);
        g.node_mut(next).name = Some("cache.next".into());
        let seen = g.add_node(
            Op::Compare(CmpOp::Le),
            vec![iota, pos],
            Shape::new(&[cap], DType::Bool),
        );
        let seen = g.add_node(Op::Cast { to: DType::F32 }, vec![seen], f(&[cap]));
        let rows = g.reshape(next, vec![cap as i64], f(&[cap]));
        let kept = g.binary(rlx_ir::op::BinaryOp::Mul, rows, seen, f(&[cap]));
        let sum = g.reduce(kept, ReduceOp::Sum, vec![0], true, f(&[1]));
        let at = g.add_node(
            Op::Cast { to: DType::I64 },
            vec![sum],
            Shape::new(&[1], DType::I64),
        );
        let logits = g.gather(table, at, 0, f(&[1, 8]));
        g.set_outputs(vec![logits, next]);
        g.set_vars(vec![(cache, 1)]);
        g
    }
    // The prompt in one run: the same sum up to `last`, and the ids as the
    // cache's rows.
    fn prefill(seq: usize) -> Graph {
        let mut g = Graph::new(format!("prefill_{seq}"));
        let f = |d: &[usize]| Shape::new(d, DType::F32);
        let ids = g.input("input_ids", Shape::new(&[1, seq], DType::I64));
        let last = g.input("last", Shape::new(&[1], DType::I64));
        let table = g.param("next", f(&[128, 8]));
        let iota = g.param(format!("iota_{seq}"), f(&[seq]));
        let values = g.add_node(Op::Cast { to: DType::F32 }, vec![ids], f(&[1, seq]));
        let rows = g.reshape(values, vec![1, seq as i64, 1], f(&[1, seq, 1]));
        let last = g.add_node(Op::Cast { to: DType::F32 }, vec![last], f(&[1]));
        let seen = g.add_node(
            Op::Compare(CmpOp::Le),
            vec![iota, last],
            Shape::new(&[seq], DType::Bool),
        );
        let seen = g.add_node(Op::Cast { to: DType::F32 }, vec![seen], f(&[seq]));
        let flat = g.reshape(values, vec![seq as i64], f(&[seq]));
        let kept = g.binary(rlx_ir::op::BinaryOp::Mul, flat, seen, f(&[seq]));
        let sum = g.reduce(kept, ReduceOp::Sum, vec![0], true, f(&[1]));
        let at = g.add_node(
            Op::Cast { to: DType::I64 },
            vec![sum],
            Shape::new(&[1], DType::I64),
        );
        let logits = g.gather(table, at, 0, f(&[1, 8]));
        g.set_outputs(vec![logits, rows]);
        g
    }
    let _loader = tokenizer_lock();
    set_tokenizer_loader(|_bytes, _chat| Ok(Box::new(Letters)));
    let dir = std::env::temp_dir().join(format!("rlx-stateful-entry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let path = dir.join("lm.rlxp");
    let next: Vec<f32> = (0..128)
        .flat_map(|i| (0..8).map(move |j| if j == (i + 1) % 8 { 1.0 } else { 0.0 }))
        .collect();
    let model = json!({
        "name": "summing-lm", "family": "sum",
        "entries": {"generate": {
            "functions": ["step_4", "step_16"],
            "tokenizer": "tokenizer.json",
            "vocab_size": 8,
            "inputs": {"input_ids": "token", "position": "position"},
            "state_update": "replace",
            "logits": 0,
            "prefill": {
                "function": "prefill_8",
                "inputs": {"input_ids": "tokens", "last": "last"},
                "logits": 0,
                "state": [1],
            },
        }},
    });
    let iota = |n: usize| (0..n).map(|i| i as f32).collect::<Vec<f32>>();
    let mut module = rlx_ir::rlxmodel::Module::new(vec![step(4), step(16), prefill(8)]);
    module.model = Some(model);
    rlxp::save_package_module(
        &path,
        &module,
        &[
            Tensor::f32("next", vec![128, 8], &next),
            Tensor::f32("iota_4", vec![4], &iota(4)),
            Tensor::f32("iota_16", vec![16], &iota(16)),
            Tensor::f32("iota_8", vec![8], &iota(8)),
        ],
        &[],
    )
    .unwrap();

    let mut devices = vec!["cpu"];
    if cfg!(feature = "metal") && crate::device_ext::is_available(crate::Device::Metal) {
        devices.push("metal");
    }
    for device in devices {
        let mut m = open_source(path.to_str().unwrap(), json!({"device": device})).unwrap();
        // c=2 → d (3), then 2+3 → g (6), 11 → e (4), 15 → a (0); position
        // 4 outgrows the 4-row step, and the 16-row one sums the same 15.
        for _ in 0..2 {
            assert_eq!(
                m.generate_text("c", &json!({"max_tokens": 6})).unwrap(),
                "dgeaaa",
                "{device}"
            );
        }
        // A second generation in the 16-row step: the rows the first left
        // past its position are masked, not summed.
        assert_eq!(
            m.generate_text("b", &json!({"max_tokens": 3})).unwrap(),
            "cea",
            "{device}"
        );
        // A prompt of more than one token goes through the prefill (padded
        // to 8), whose rows the step then sums on: `dc` is 3+2 = 5 → g (6),
        // then 11 → e (4), 15 → a; rows the last generation left (`b`, `c`)
        // would read 9 → c instead. One longer than the prefill's 8 is stepped
        // token by token: `abcabcabd` sums to 10 → d (3), then 13 → g.
        assert_eq!(
            m.generate_text("dc", &json!({"max_tokens": 3})).unwrap(),
            "gea",
            "{device}"
        );
        assert_eq!(
            m.generate_text("abcabcabd", &json!({"max_tokens": 2}))
                .unwrap(),
            "dg",
            "{device}"
        );

        // A session snapshot resumes where it was taken, after other
        // prompts filled the caches: one taken after a prefill (its caches
        // on the host), one after a step (its caches in the executable).
        // `decode_logits` steps from the restored cache alone.
        let lm = m.lm().unwrap();
        let want = lm.predict_logits(&[3, 2, 4]).unwrap();
        let want_next = lm.predict_logits(&[3, 2, 4, 1]).unwrap();
        lm.prefill_logits(&[3, 2]).unwrap();
        let after_prefill = lm.export_session().unwrap();
        lm.decode_logits(4).unwrap();
        let after_step = lm.export_session().unwrap();
        assert_eq!(after_step.tokens, [3, 2, 4]);
        lm.predict_logits(&[7, 7, 7, 7, 7, 7, 7, 7, 7, 7]).unwrap();
        assert!(lm.restore_session(&after_prefill), "{device}");
        assert_eq!(lm.decode_logits(4).unwrap(), want, "{device}");
        lm.predict_logits(&[6, 6, 6, 6, 6]).unwrap();
        assert!(lm.restore_session(&after_step), "{device}");
        assert_eq!(lm.decode_logits(1).unwrap(), want_next, "{device}");
        // Only the prompt past the snapshot runs.
        lm.predict_logits(&[5, 5]).unwrap();
        let got = lm
            .prefill_logits_reusing(&[3, 2, 4, 1], &after_step, 3)
            .unwrap();
        assert_eq!(got, want_next, "{device}");
        // Another package's snapshot (other caches) is refused.
        let mut alien = after_step.clone();
        alien.kv.layers_k.push(Vec::new());
        assert!(!lm.restore_session(&alien));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// A package with no decode step, only full-sequence prompt functions
/// (`prefill`, no `functions`): each token runs the shortest that holds the
/// context over all of it, as a runner without a cache does. The next id is
/// 1 + the sum of every id so far, mod 8, so each run must see the whole
/// context. One function per length (`prefill.functions`), 4 and 8.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn a_package_with_no_step_recomputes_the_context_each_token() {
    use rlx_ir::op::{CmpOp, ReduceOp};
    use rlx_ir::{Graph, Op, Shape};
    fn forward(seq: usize) -> Graph {
        let mut g = Graph::new(format!("forward_{seq}"));
        let f = |d: &[usize]| Shape::new(d, DType::F32);
        let ids = g.input("input_ids", Shape::new(&[1, seq], DType::I64));
        let last = g.input("last", Shape::new(&[1], DType::I64));
        let table = g.param("next", f(&[128, 8]));
        let iota = g.param(format!("iota_{seq}"), f(&[seq]));
        let values = g.add_node(Op::Cast { to: DType::F32 }, vec![ids], f(&[1, seq]));
        let last = g.add_node(Op::Cast { to: DType::F32 }, vec![last], f(&[1]));
        let seen = g.add_node(
            Op::Compare(CmpOp::Le),
            vec![iota, last],
            Shape::new(&[seq], DType::Bool),
        );
        let seen = g.add_node(Op::Cast { to: DType::F32 }, vec![seen], f(&[seq]));
        let flat = g.reshape(values, vec![seq as i64], f(&[seq]));
        let kept = g.binary(rlx_ir::op::BinaryOp::Mul, flat, seen, f(&[seq]));
        let sum = g.reduce(kept, ReduceOp::Sum, vec![0], true, f(&[1]));
        let at = g.add_node(
            Op::Cast { to: DType::I64 },
            vec![sum],
            Shape::new(&[1], DType::I64),
        );
        let logits = g.gather(table, at, 0, f(&[1, 8]));
        g.set_outputs(vec![logits]);
        g
    }
    let _loader = tokenizer_lock();
    set_tokenizer_loader(|_bytes, _chat| Ok(Box::new(Letters)));
    let dir = std::env::temp_dir().join(format!("rlx-recompute-entry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let path = dir.join("lm.rlxp");
    let next: Vec<f32> = (0..128)
        .flat_map(|i| (0..8).map(move |j| if j == (i + 1) % 8 { 1.0 } else { 0.0 }))
        .collect();
    let model = json!({
        "name": "summing-lm", "family": "sum",
        "entries": {"generate": {
            "tokenizer": "tokenizer.json",
            "vocab_size": 8,
            "inputs": {},
            "prefill": {
                "functions": ["forward_4", "forward_8"],
                "inputs": {"input_ids": "tokens", "last": "last"},
                "logits": 0,
                "state": [],
            },
        }},
    });
    let iota = |n: usize| (0..n).map(|i| i as f32).collect::<Vec<f32>>();
    let mut module = rlx_ir::rlxmodel::Module::new(vec![forward(4), forward(8)]);
    module.model = Some(model);
    rlxp::save_package_module(
        &path,
        &module,
        &[
            Tensor::f32("next", vec![128, 8], &next),
            Tensor::f32("iota_4", vec![4], &iota(4)),
            Tensor::f32("iota_8", vec![8], &iota(8)),
        ],
        &[],
    )
    .unwrap();

    let mut m = open_source(path.to_str().unwrap(), json!({})).unwrap();
    // c=2 → d (3), 2+3 → g (6), 11 → e (4), 15 → a (0), 15 → a: past four
    // tokens the 8-token function runs the context.
    assert_eq!(
        m.generate_text("c", &json!({"max_tokens": 5})).unwrap(),
        "dgeaa"
    );
    let lm = m.lm().unwrap();
    // A continuation and a fresh prompt both see their whole context.
    assert_eq!(
        crate::lm::argmax_u32(&lm.predict_logits(&[3, 2]).unwrap()),
        6
    );
    assert_eq!(
        crate::lm::argmax_u32(&lm.predict_logits(&[3, 2, 1]).unwrap()),
        7
    );
    assert_eq!(crate::lm::argmax_u32(&lm.predict_logits(&[1]).unwrap()), 2);
    assert_eq!(
        crate::lm::argmax_u32(&lm.predict_logits(&[1; 6]).unwrap()),
        7
    );
    let err = lm.predict_logits(&[1; 9]).unwrap_err().to_string();
    assert!(err.contains("the context is full: 9 tokens"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Recurrent state (`kind: "replace"`) becomes its step's whole output, with
/// no length bounding it: here a running sum of the ids, the next id
/// `sum + 1` mod 8.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn a_recurrent_state_is_replaced_whole() {
    use rlx_ir::{Graph, Op, Shape};
    let f = |d: &[usize]| Shape::new(d, DType::F32);
    let mut g = Graph::new("step");
    let ids = g.input("input_ids", Shape::new(&[1], DType::I64));
    let h = g.input("h", f(&[1]));
    let table = g.param("next", f(&[128, 8]));
    let id = g.add_node(Op::Cast { to: DType::F32 }, vec![ids], f(&[1]));
    let sum = g.binary(rlx_ir::op::BinaryOp::Add, h, id, f(&[1]));
    let at = g.add_node(
        Op::Cast { to: DType::I64 },
        vec![sum],
        Shape::new(&[1], DType::I64),
    );
    let logits = g.gather(table, at, 0, f(&[1, 8]));
    g.set_outputs(vec![logits, sum]);
    let _loader = tokenizer_lock();
    set_tokenizer_loader(|_bytes, _chat| Ok(Box::new(Letters)));
    let dir = std::env::temp_dir().join(format!("rlx-recurrent-entry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let path = dir.join("lm.rlxp");
    let next: Vec<f32> = (0..128)
        .flat_map(|i| (0..8).map(move |j| if j == (i + 1) % 8 { 1.0 } else { 0.0 }))
        .collect();
    let model = json!({
        "name": "summing-rnn", "family": "sum",
        "entries": {"generate": {
            "functions": ["step"],
            "tokenizer": "tokenizer.json",
            "vocab_size": 8,
            "inputs": {"input_ids": "token"},
            "state": [["h", 1, "replace"]],
            "logits": 0,
        }},
    });
    rlxp::save_module(
        &path,
        &[g],
        &[Tensor::f32("next", vec![128, 8], &next)],
        &model,
    )
    .unwrap();
    let mut m = open_source(path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    // c=2 → d (3); 5 → g (6); 11 → e (4); 15 → a (0); 15 → a.
    assert_eq!(
        m.generate_text("c", &json!({"max_tokens": 5})).unwrap(),
        "dgeaa"
    );
    // A new prompt starts from zero state.
    assert_eq!(
        m.generate_text("b", &json!({"max_tokens": 2})).unwrap(),
        "ce"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The same encoder as one symbolic graph: generalized from two builds,
/// stored once, specialized at the sizes its entry lists.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn a_symbolic_function_runs_at_the_sizes_its_entry_lists() {
    use rlx_ir::{Graph, Shape};
    fn encoder(batch: usize, seq: usize) -> Graph {
        let mut g = Graph::new("encode");
        let f = |d: &[usize]| Shape::new(d, DType::F32);
        let ids = g.input("input_ids", f(&[batch, seq]));
        let w = g.param("w", f(&[1, 2]));
        let col = g.reshape(ids, vec![(batch * seq) as i64, 1], f(&[batch * seq, 1]));
        let h = g.matmul(col, w, f(&[batch * seq, 2]));
        let h = g.reshape(h, vec![batch as i64, seq as i64, 2], f(&[batch, seq, 2]));
        g.set_outputs(vec![h]);
        g
    }
    let _loader = tokenizer_lock();
    set_tokenizer_loader(|_bytes, _chat| Ok(Box::new(Letters)));
    // batch 1, so the flattened `batch * seq` is `seq` too.
    let g = rlx_ir::dynamic::generalize(&encoder(1, 37), &encoder(1, 53), &[(0, 37, 53)]).unwrap();
    let dir = std::env::temp_dir().join(format!("rlx-symbolic-entry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let path = dir.join("enc.rlxp");
    let mut module = rlx_ir::rlxmodel::Module::new(vec![g]);
    module.dims = vec![("seq".into(), 0)];
    module.model = Some(json!({
        "name": "letters-encoder", "family": "letters",
        "entries": {"embed": {
            "functions": ["encode"],
            "sizes": [{"seq": 4}, {"seq": 8}],
            "tokenizer": "tokenizer.json",
            "inputs": {"input_ids": "token_ids"},
            "pooling": "mean",
            "normalize": false,
        }},
    }));
    rlxp::save_package_module(
        &path,
        &module,
        &[Tensor::f32("w", vec![1, 2], &[1.0, 10.0])],
        &[],
    )
    .unwrap();
    let mut m = open_source(path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    assert_eq!(m.info().metadata["functions"], json!(["encode"]));
    let v = m
        .embed(&["bd", "cccccc", "bbbbbbbbbd"], &json!({}))
        .unwrap();
    let close = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5);
    assert!(close(&v[0], &[2.0, 20.0]), "{v:?}");
    assert!(close(&v[1], &[2.0, 20.0]), "{v:?}");
    assert!(close(&v[2], &[1.25, 12.5]), "{v:?}");
    // The stored text says `seq`.
    let pack = rlx_pkg::Package::open(&path).unwrap();
    let text = rlx_ir::rlxmodel::write_module(&pack.module().unwrap()).unwrap();
    assert!(text.contains("f32[1, seq]"), "{text}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Entries named apart from their kind (`"double": {"kind": "pipeline"}`):
/// two pipelines in one package, opened as one model. `run` goes to the
/// entry whose inputs are the tensors given; `options.entry`, or the open
/// option, picks one by name; the package says it needs `entry.kind`.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn named_entries_route_by_their_inputs() {
    use rlx_ir::{Graph, GraphExt, Shape};
    let scale = |name: &str, input: &str, k: f32| {
        let mut g = Graph::new(name);
        let x = g.input(input, Shape::new(&[3], DType::F32));
        let c = g.constant(f64::from(k), DType::F32);
        let y = g.mul(x, c);
        g.set_outputs(vec![y]);
        g
    };
    let dir = std::env::temp_dir().join(format!("rlx-named-entries-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut m =
        rlx_ir::rlxmodel::Module::new(vec![scale("twice", "a", 2.0), scale("thrice", "b", 3.0)]);
    let pipe = |f: &str, input: &str| {
        json!({"kind": "pipeline", "task": "feature-extraction", "inputs": [input],
               "stages": [{"function": f, "inputs": {input: input}, "outputs": ["y"]}],
               "outputs": ["y"]})
    };
    m.model = Some(json!({
        "name": "two-pipelines", "family": "toy",
        "entries": {"double": pipe("twice", "a"), "triple": pipe("thrice", "b")},
    }));
    let features = rlxp::features_of(m.model.as_ref().unwrap());
    assert!(features.contains(&"entry.kind".to_string()), "{features:?}");
    assert!(
        !features.contains(&"pipeline.kind".to_string()),
        "{features:?}"
    );
    let path = dir.join("two.rlxp");
    rlxp::save_package_module(&path, &m, &[], &[]).unwrap();
    let source = path.to_str().unwrap();

    let mut model = open_source(source, json!({"device": "cpu"})).unwrap();
    let info = model.info();
    let names: Vec<&str> = info.metadata["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["double", "triple"]);
    let x = |name: &str| Tensor::f32(name, vec![3], &[1.0, 2.0, 3.0]);
    let y = |out: Vec<Tensor>| out[0].to_f32().unwrap();
    assert_eq!(
        y(model.run(&[x("a")], &json!({})).unwrap()),
        [2.0, 4.0, 6.0]
    );
    assert_eq!(
        y(model.run(&[x("b")], &json!({})).unwrap()),
        [3.0, 6.0, 9.0]
    );
    let picked = model.run(&[x("b")], &json!({"entry": "triple"})).unwrap();
    assert_eq!(y(picked), [3.0, 6.0, 9.0]);

    let mut one = open_source(source, json!({"device": "cpu", "entry": "triple"})).unwrap();
    assert_eq!(y(one.run(&[x("b")], &json!({})).unwrap()), [3.0, 6.0, 9.0]);
    assert!(one.info().metadata.get("entries").is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A package with several entries opens the one the option `entry` names:
/// here its `pipeline`, where it would open its `embed` (which needs a
/// tokenizer it lacks) by default; an entry it does not have is refused.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn the_entry_option_picks_one_of_several_entries() {
    use rlx_ir::{Graph, GraphExt, Shape};
    let mut g = Graph::new("double");
    let x = g.input("x", Shape::new(&[3], DType::F32));
    let two = g.constant(2.0, DType::F32);
    let y = g.mul(x, two);
    g.set_outputs(vec![y]);
    let dir = std::env::temp_dir().join(format!("rlx-entry-option-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut m = rlx_ir::rlxmodel::Module::new(vec![g]);
    m.model = Some(json!({
        "name": "two-entries", "family": "toy",
        "entries": {
            "embed": {"functions": ["double"], "tokenizer": "sidecar:tokenizer.json"},
            "pipeline": {
                "task": "feature-extraction",
                "inputs": ["x"],
                "stages": [{"function": "double", "inputs": {"x": "x"}, "outputs": ["y"]}],
                "outputs": ["y"],
            },
        },
    }));
    let path = dir.join("two.rlxp");
    rlxp::save_package_module(&path, &m, &[], &[]).unwrap();
    let source = path.to_str().unwrap();
    assert!(open_source(source, json!({"device": "cpu"})).is_err());
    let mut model = open_source(source, json!({"device": "cpu", "entry": "pipeline"})).unwrap();
    let out = model
        .run(&[Tensor::f32("x", vec![3], &[1.0, 2.0, 3.0])], &json!({}))
        .unwrap();
    assert_eq!(out[0].to_f32().unwrap(), [2.0, 4.0, 6.0]);
    let e = open_source(source, json!({"device": "cpu", "entry": "score"}))
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("no entry 'score'"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A pipeline entry: a front end from its own package, symbolic in its
/// input's length and reading that length (`x#shape`), then a sum; run at
/// two lengths, from a u8 tensor.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn a_pipeline_runs_its_stages_at_each_input_shape() {
    use rlx_ir::{Graph, GraphExt, Shape};
    let f = |d: &[usize]| Shape::new(d, DType::F32);
    let front = |n: usize| {
        let mut g = Graph::new("front");
        let x = g.input("x", f(&[n]));
        let dims = g.input("dims", f(&[1]));
        let w = g.param("w", f(&[1]));
        let y = g.mul(x, w);
        let y = g.add(y, dims);
        g.set_outputs(vec![y]);
        g
    };
    let front = rlx_ir::dynamic::generalize(&front(37), &front(53), &[(0, 37, 53)]).unwrap();
    let total = |n: usize| {
        let mut g = Graph::new("total");
        let y = g.input("y", f(&[n]));
        let t = g.reduce(y, rlx_ir::op::ReduceOp::Sum, vec![0], true, f(&[1]));
        g.set_outputs(vec![t]);
        g
    };
    let total = rlx_ir::dynamic::generalize(&total(37), &total(53), &[(0, 37, 53)]).unwrap();
    let dir = std::env::temp_dir().join(format!("rlx-pipeline-entry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut m = rlx_ir::rlxmodel::Module::new(vec![front]);
    m.dims = vec![("n".into(), 0)];
    rlxp::save_package_module(
        &dir.join("front.rlxp"),
        &m,
        &[Tensor::f32("w", vec![1], &[10.0])],
        &[],
    )
    .unwrap();
    let mut m = rlx_ir::rlxmodel::Module::new(vec![total]);
    m.dims = vec![("n".into(), 0)];
    m.model = Some(json!({
        "name": "scaled-sum", "family": "sum",
        "entries": {"pipeline": {
            "task": "feature-extraction",
            "inputs": [{"name": "x", "kind": "signal", "dtype": "u8", "shape": [null], "rate": 8}],
            "stages": [
                {"function": "front", "package": "front.rlxp",
                 "inputs": {"x": "x", "dims": "x#shape"}, "outputs": ["y"]},
                {"function": "total", "inputs": {"y": "y"}, "outputs": ["total"]},
            ],
            "outputs": ["y", "total"],
        }},
    }));
    let path = dir.join("sum.rlxp");
    rlxp::save_package_module(&path, &m, &[], &[]).unwrap();
    let mut model = open_source(path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    let info = model.info();
    assert_eq!(info.task, "feature-extraction");
    // The contract: its dtype and shape, and what else it says.
    assert_eq!(info.inputs[0].shape, vec![None]);
    assert_eq!(info.inputs[0].dtype, DType::U8);
    assert_eq!(info.metadata["inputs"][0]["kind"], "signal");
    assert_eq!(info.metadata["inputs"][0]["rate"], 8);
    assert_eq!(info.metadata["outputs"][1]["name"], "total");
    let e = model
        .run(
            &[Tensor::new("x", DType::U8, vec![1, 3], vec![1, 2, 3])],
            &json!({}),
        )
        .unwrap_err()
        .to_string();
    assert!(e.contains("this model takes [None]"), "{e}");
    for (x, y, t) in [
        (vec![1u8, 2, 3], vec![13.0, 23.0, 33.0], 69.0),
        (
            vec![1, 2, 3, 4, 5],
            vec![15.0, 25.0, 35.0, 45.0, 55.0],
            175.0,
        ),
        (vec![3, 2, 1], vec![33.0, 23.0, 13.0], 69.0),
    ] {
        let input = Tensor::new("pixels", DType::U8, vec![x.len()], x.clone());
        let out = model.run(&[input], &json!({})).unwrap();
        assert_eq!(out[0].shape, vec![x.len()]);
        assert_eq!(out[0].to_f32().unwrap(), y);
        assert_eq!(out[1].to_f32().unwrap(), vec![t]);
    }
    // `stage_devices` picks a stage's device over the package's.
    let e = open_source(
        path.to_str().unwrap(),
        json!({"device": "cpu", "stage_devices": {"total": "no-such-device"}}),
    )
    .err()
    .expect("an unknown stage device fails")
    .to_string();
    assert!(e.contains("device of stage 'total'"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A package's model block as a Hugging Face `README.md`: `card_data` as
/// the front matter, the block's own fields where it has none, then the body.
#[cfg(feature = "model-io")]
#[test]
fn a_model_block_renders_as_a_huggingface_model_card() {
    let model = json!({
        "name": "e5", "task": "embedding", "license": "mit",
        "card_data": {"language": ["en", "de"], "base_model": "intfloat/e5-base",
                      "model-index": [{"name": "e5", "results": []}]},
        "card": "\n# e5\n\nText embeddings.\n",
    });
    let readme = rlxp::huggingface_readme(&model);
    assert!(readme.starts_with("---\n"), "{readme}");
    for line in [
        "language: [\"en\",\"de\"]\n",
        "base_model: \"intfloat/e5-base\"\n",
        "model-index: [{\"name\":\"e5\",\"results\":[]}]\n",
        "license: \"mit\"\n",
        "pipeline_tag: \"feature-extraction\"\n",
        "library_name: \"rlx\"\n",
    ] {
        assert!(readme.contains(line), "missing {line:?} in\n{readme}");
    }
    assert!(
        readme.ends_with("---\n\n# e5\n\nText embeddings.\n"),
        "{readme}"
    );
    // A card's own pipeline_tag / library_name win.
    let own = json!({"task": "embedding", "card_data": {"pipeline_tag": "sentence-similarity"}});
    assert!(rlxp::huggingface_readme(&own).contains("pipeline_tag: \"sentence-similarity\""));
}

/// A package lists the entry features it uses; an rlx that lacks one
/// refuses it by name rather than run without it.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn a_package_using_an_unknown_feature_is_refused_by_name() {
    use rlx_ir::{Graph, Shape};
    let mut g = Graph::new("main");
    let x = g.input("x", Shape::new(&[1], DType::F32));
    g.set_outputs(vec![x]);
    let dir = std::env::temp_dir().join(format!("rlx-features-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("future.rlxp");
    let model = json!({"name": "future", "entries": {"generate": {"teleport": true}}});
    rlxp::save_module(&path, &[g], &[], &model).unwrap();
    let pack = rlx_pkg::Package::open(&path).unwrap();
    let written = pack.module().unwrap().model.unwrap();
    assert_eq!(
        written["features"],
        json!(["generate", "generate.teleport"])
    );
    let e = open_source(path.to_str().unwrap(), json!({"device": "cpu"}))
        .err()
        .expect("refused")
        .to_string();
    assert!(
        e.contains("generate.teleport") && e.contains("update rlx"),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A model's generation defaults apply where the caller says nothing:
/// `max_new_tokens` caps the text, the caller's `max_tokens` wins.
#[test]
fn generation_defaults_fill_what_the_caller_leaves_out() {
    let info = ModelInfo {
        metadata: json!({"generation": {"max_new_tokens": 3, "do_sample": false, "temperature": 0.7}}),
        ..Default::default()
    };
    let mut m = lm_model(Box::new(CountLm), Some(Box::new(Letters)), info);
    assert_eq!(m.generate_text("a", &json!({})).unwrap(), "bcd");
    assert_eq!(
        m.generate_text("a", &json!({"max_tokens": 5})).unwrap(),
        "bcdef"
    );
}

/// `response_format: "json"`: the likeliest tokens that do not continue a
/// JSON value are passed over, and the value ends once whole. This model
/// always likes `x` best, then `}`, then `{`: nothing opens with `x` or
/// `}`, so it writes `{`, then `}` (an empty object, whole), and stops.
#[test]
fn json_output_keeps_to_a_json_prefix() {
    const VOCAB: &[u8] = b"x}{\"a:1 ";
    struct Prefers;
    impl LmRunner for Prefers {
        fn family(&self) -> &'static str {
            "prefers"
        }
        fn vocab_size(&self) -> usize {
            VOCAB.len()
        }
        fn predict_logits(&mut self, _ids: &[u32]) -> anyhow::Result<Vec<f32>> {
            Ok((0..VOCAB.len()).map(|i| -(i as f32)).collect())
        }
    }
    struct Chars;
    impl TextCodec for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text
                .bytes()
                .map(|b| VOCAB.iter().position(|&v| v == b).unwrap() as u32)
                .collect())
        }
        fn decode(&self, ids: &[u32]) -> Result<String> {
            Ok(ids.iter().map(|&i| VOCAB[i as usize] as char).collect())
        }
    }
    let mut m = lm_model(
        Box::new(Prefers),
        Some(Box::new(Chars)),
        ModelInfo::default(),
    );
    let out = m
        .generate_text(" ", &json!({"max_tokens": 6, "response_format": "json"}))
        .unwrap();
    assert_eq!(out, "{}");
    let plain = m.generate_text(" ", &json!({"max_tokens": 3})).unwrap();
    assert_eq!(plain, "xxx");
}

/// A pipeline's loop: Newton's step for √a (`x ← (x + a/x) / 2`) until
/// `x²` is within 1e-4 of `a`, each step scaled by its `schedule` value (1),
/// counting its steps; then a finish stage doubles it. It stops well before
/// its 20 steps, and the `steps` option caps it.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn a_pipeline_loops_until_its_state_says_done() {
    use rlx_ir::op::{BinaryOp, CmpOp};
    use rlx_ir::{Graph, GraphExt, Op, Shape};
    let f = Shape::new(&[1], DType::F32);
    let mut newton = Graph::new("newton");
    let x = newton.input("x", f.clone());
    let a = newton.input("a", f.clone());
    let step = newton.input("step", f.clone());
    let scale = newton.input("scale", f.clone());
    let q = newton.div(a, x);
    let sum = newton.add(x, q);
    let half = newton.constant(0.5, DType::F32);
    let next = newton.mul(sum, half);
    let next = newton.mul(next, scale);
    let sq = newton.mul(next, next);
    let err = newton.sub(sq, a);
    let err = newton.abs(err);
    let tol = newton.constant(1e-4, DType::F32);
    let ok = newton.add_node(
        Op::Compare(CmpOp::Lt),
        vec![err, tol],
        Shape::new(&[1], DType::Bool),
    );
    let done = newton.add_node(Op::Cast { to: DType::F32 }, vec![ok], f.clone());
    let one = newton.constant(1.0, DType::F32);
    let count = newton.binary(BinaryOp::Add, step, one, f.clone());
    newton.set_outputs(vec![next, done, count]);
    let mut start = Graph::new("start");
    let a0 = start.input("a", f.clone());
    let one0 = start.constant(1.0, DType::F32);
    let x0 = start.mul(a0, one0);
    start.set_outputs(vec![x0]);
    let mut twice = Graph::new("twice");
    let xi = twice.input("x", f.clone());
    let two = twice.constant(2.0, DType::F32);
    let y = twice.mul(xi, two);
    twice.set_outputs(vec![y]);
    let dir = std::env::temp_dir().join(format!("rlx-loop-entry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("newton.rlxp");
    let model = json!({
        "name": "newton", "family": "loop",
        "entries": {"pipeline": {
            "inputs": [{"name": "a", "kind": "tensor", "dtype": "f32", "shape": [1]}],
            "stages": [{"function": "start", "inputs": {"a": "a"}, "outputs": ["x"]}],
            "loop": {
                "stages": [{"function": "newton",
                            "inputs": {"x": "x", "a": "a", "step": "step", "scale": "scale"},
                            "outputs": ["x", "done", "count"]}],
                "steps": 20,
                "schedule": {"scale": vec![1.0; 20]},
                "until": "done",
            },
            "finish": [{"function": "twice", "inputs": {"x": "x"}, "outputs": ["y"]}],
            "outputs": ["y", "count"],
        }},
    });
    rlxp::save_module(&path, &[start, newton, twice], &[], &model).unwrap();
    let mut m = open_source(path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    let out = m
        .run(&[Tensor::f32("a", vec![1], &[2.0])], &json!({}))
        .unwrap();
    let y = out[0].to_f32().unwrap()[0];
    let count = out[1].to_f32().unwrap()[0];
    assert!((y - 2.0 * 2f32.sqrt()).abs() < 1e-4, "{y}");
    assert!(count < 10.0, "ran {count} steps");
    // `steps: 1`: one Newton step from 2 is 1.5.
    let out = m
        .run(&[Tensor::f32("a", vec![1], &[2.0])], &json!({"steps": 1}))
        .unwrap();
    assert_eq!(out[0].to_f32().unwrap()[0], 3.0);
    let _ = std::fs::remove_dir_all(&dir);
}

/// An image in the prompt: its stage's embedding rows stand where its
/// placeholder was, then decoding goes on by tokens. Next id = (the sum
/// of every embedding value so far + 1) mod 8: `abPc` with image rows 3
/// and 4 sums to 0 + 1 + 3 + 4 + 2 = 10, so `d`, then `g`, then `e`.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn an_image_in_the_prompt_stands_where_its_placeholder_was() {
    use rlx_ir::{Graph, GraphExt, Op, Shape};
    let f = |d: &[usize]| Shape::new(d, DType::F32);
    let generalized = |build: &dyn Fn(usize) -> Graph, sym: u32| {
        rlx_ir::dynamic::generalize(&build(37), &build(53), &[(sym, 37, 53)]).unwrap()
    };
    let embed = generalized(
        &|n| {
            let mut g = Graph::new("embed");
            let ids = g.input("input_ids", Shape::new(&[n], DType::I32));
            let v = g.add_node(Op::Cast { to: DType::F32 }, vec![ids], f(&[n]));
            let out = g.reshape_(v, vec![n as i64, 1]);
            g.set_outputs(vec![out]);
            g
        },
        0,
    );
    let image = generalized(
        &|k| {
            let mut g = Graph::new("image");
            let px = g.input("pixels", f(&[k]));
            let out = g.reshape_(px, vec![k as i64, 1]);
            g.set_outputs(vec![out]);
            g
        },
        0,
    );
    let sum_logits = |g: &mut Graph, rows: rlx_ir::NodeId, n: usize| {
        let table = g.param("next", f(&[256, 8]));
        let sum = g.reduce(rows, rlx_ir::op::ReduceOp::Sum, vec![1, 2], false, f(&[1]));
        let at = g.add_node(
            Op::Cast { to: DType::I64 },
            vec![sum],
            Shape::new(&[1], DType::I64),
        );
        let _ = n;
        g.gather(table, at, 0, f(&[1, 8]))
    };
    let prefill = generalized(
        &|seq| {
            let mut g = Graph::new("prefill_embeds");
            let e = g.input("inputs_embeds", f(&[1, seq, 1]));
            let _last = g.input("last", f(&[1]));
            let logits = sum_logits(&mut g, e, seq);
            g.set_outputs(vec![logits, e]);
            g
        },
        0,
    );
    let step = generalized(
        &|upper| {
            let mut g = Graph::new("step");
            let ids = g.input("input_ids", Shape::new(&[1], DType::I64));
            let past = g.input("past_k_0", f(&[1, upper, 1]));
            let v = g.add_node(Op::Cast { to: DType::F32 }, vec![ids], f(&[1]));
            let row = g.reshape_(v, vec![1, 1, 1]);
            let cache = g.concat(vec![past, row], 1, f(&[1, upper + 1, 1]));
            let logits = sum_logits(&mut g, cache, upper + 1);
            g.set_outputs(vec![logits, cache]);
            g
        },
        0,
    );
    struct Chars;
    impl TextCodec for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text
                .bytes()
                .map(|b| if b == b'P' { 8 } else { (b - b'a') as u32 })
                .collect())
        }
        fn decode(&self, ids: &[u32]) -> Result<String> {
            Ok(ids.iter().map(|&i| (b'a' + i as u8) as char).collect())
        }
    }
    let _loader = tokenizer_lock();
    set_tokenizer_loader(|_bytes, _chat| Ok(Box::new(Chars)));
    let dir = std::env::temp_dir().join(format!("rlx-media-entry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let next: Vec<f32> = (0..256)
        .flat_map(|i| (0..8).map(move |j| if j == (i + 1) % 8 { 1.0 } else { 0.0 }))
        .collect();
    let mut module = rlx_ir::rlxmodel::Module::new(vec![step, prefill, embed, image]);
    module.dims = vec![("n".into(), 0)];
    module.model = Some(json!({
        "name": "summing-vlm", "family": "sum",
        "entries": {"generate": {
            "functions": ["step"],
            "sizes": [{"n": 16}],
            "tokenizer": "tokenizer.json",
            "vocab_size": 8,
            "inputs": {"input_ids": "token"},
            "state": [["past_k_0", 1]],
            "logits": 0,
            "media": {
                "embed": {"function": "embed", "inputs": {"input_ids": "ids"}, "outputs": ["embeds"]},
                "prefill": {"function": "prefill_embeds",
                            "inputs": {"inputs_embeds": "prompt_embeds", "last": "last"},
                            "outputs": ["logits", "past_k_0"], "sizes": [8]},
                "kinds": {"image": {"token": 8, "input": "image",
                    "stages": [{"function": "image", "inputs": {"pixels": "image"}, "outputs": ["image_embeds"]}],
                    "output": "image_embeds"}},
            },
        }},
    }));
    let path = dir.join("vlm.rlxp");
    rlxp::save_package_module(
        &path,
        &module,
        &[Tensor::f32("next", vec![256, 8], &next)],
        &[],
    )
    .unwrap();
    let mut m = open_source(path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    let image = Tensor::f32("image", vec![2], &[3.0, 4.0]);
    let opts = json!({"max_tokens": 3});
    assert_eq!(
        m.generate_with_media("abPc", std::slice::from_ref(&image), &opts)
            .unwrap(),
        "dge"
    );
    // A run of placeholders is one image; a prompt with none refuses it.
    assert_eq!(
        m.generate_with_media("abPPPc", std::slice::from_ref(&image), &opts)
            .unwrap(),
        "dge"
    );
    let e = m
        .generate_with_media("abc", &[image], &opts)
        .unwrap_err()
        .to_string();
    assert!(e.contains("more media"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Multimodal RoPE: each token's RoPE rows come from its 3-D position, and
/// decoding goes on from one past the image's largest. The table's row `p`
/// is `[p, p, p]`, one frequency per axis, so a token's row is its `(t, h,
/// w)`, and the next id is (the sum of the rows + 1) mod 8. `abPc` with a
/// 2×3 image: `a (0,0,0)`, `b (1,1,1)`, the image from 2 (`t` 2, `h` 2-3,
/// `w` 2-4: 45), `c (5,5,5)`: 63, so `a`. The step's next row (9) is
/// position 6 (`[6,6,6]`: `d`), then 7 (`g`). The image's further rows
/// (`kinds.image.rows`, 1 each) reach the prefill at its six positions: 63
/// + 6, so `g` first.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn an_image_in_the_prompt_takes_its_multimodal_rope_positions() {
    use rlx_ir::{Graph, GraphExt, Op, Shape};
    let f = |d: &[usize]| Shape::new(d, DType::F32);
    let generalized = |build: &dyn Fn(usize) -> Graph, sym: u32| {
        rlx_ir::dynamic::generalize(&build(37), &build(53), &[(sym, 37, 53)]).unwrap()
    };
    let next_of = |g: &mut Graph, x: rlx_ir::NodeId, axes: Vec<usize>| {
        let table = g.param("next", f(&[256, 8]));
        let sum = g.reduce(x, rlx_ir::op::ReduceOp::Sum, axes, false, f(&[1]));
        let at = g.add_node(
            Op::Cast { to: DType::I64 },
            vec![sum],
            Shape::new(&[1], DType::I64),
        );
        g.gather(table, at, 0, f(&[1, 8]))
    };
    let embed = generalized(
        &|n| {
            let mut g = Graph::new("embed");
            let ids = g.input("input_ids", Shape::new(&[n], DType::I32));
            let v = g.add_node(Op::Cast { to: DType::F32 }, vec![ids], f(&[n]));
            let out = g.reshape_(v, vec![n as i64, 1]);
            g.set_outputs(vec![out]);
            g
        },
        0,
    );
    // A 4×6 image of 2-pixel rows: 6 embedding rows.
    let mut image = Graph::new("image");
    let px = image.input("pixels", f(&[4, 6]));
    let flat = image.reshape_(px, vec![24]);
    let six = image.add_node(
        Op::Narrow {
            axis: 0,
            start: 0,
            len: 6,
        },
        vec![flat],
        f(&[6]),
    );
    let rows = image.reshape_(six, vec![6, 1]);
    // A further output laid out like the rows (`kinds.image.rows`): 1 each.
    let one = image.constant(1.0, DType::F32);
    let extra = image.add(rows, one);
    image.set_outputs(vec![rows, extra]);
    let prefill = generalized(
        &|seq| {
            let mut g = Graph::new("prefill_embeds");
            let e = g.input("inputs_embeds", f(&[1, seq, 1]));
            let _last = g.input("last", f(&[1]));
            let cos = g.input("rope_cos", f(&[seq, 3]));
            let _sin = g.input("rope_sin", f(&[seq, 3]));
            // The extra rows, zeros but at the image's positions.
            let extra = g.input("extra", f(&[1, seq, 1]));
            let flat = g.reshape_(extra, vec![seq as i64, 1]);
            let three = g.add_node(
                Op::Expand {
                    target_shape: vec![-1, 3],
                },
                vec![flat],
                f(&[seq, 3]),
            );
            let third = g.constant(1.0 / 3.0, DType::F32);
            let spread = g.mul(three, third);
            let both = g.add(cos, spread);
            let logits = next_of(&mut g, both, vec![0, 1]);
            g.set_outputs(vec![logits, e]);
            g
        },
        0,
    );
    let step = generalized(
        &|upper| {
            let mut g = Graph::new("step");
            let ids = g.input("input_ids", Shape::new(&[1], DType::I64));
            let past = g.input("past_k_0", f(&[1, upper, 1]));
            let cos = g.input("rope_cos", f(&[1, 3]));
            let v = g.add_node(Op::Cast { to: DType::F32 }, vec![ids], f(&[1]));
            let row = g.reshape_(v, vec![1, 1, 1]);
            let cache = g.concat(vec![past, row], 1, f(&[1, upper + 1, 1]));
            let logits = next_of(&mut g, cos, vec![0, 1]);
            g.set_outputs(vec![logits, cache]);
            g
        },
        0,
    );
    struct Chars;
    impl TextCodec for Chars {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            Ok(text
                .bytes()
                .map(|b| if b == b'P' { 8 } else { (b - b'a') as u32 })
                .collect())
        }
        fn decode(&self, ids: &[u32]) -> Result<String> {
            Ok(ids.iter().map(|&i| (b'a' + i as u8) as char).collect())
        }
    }
    let _loader = tokenizer_lock();
    set_tokenizer_loader(|_bytes, _chat| Ok(Box::new(Chars)));
    let dir = std::env::temp_dir().join(format!("rlx-mrope-entry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    let next: Vec<f32> = (0..256)
        .flat_map(|i| (0..8).map(move |j| if j == (i + 1) % 8 { 1.0 } else { 0.0 }))
        .collect();
    let table: Vec<f32> = (0..32).flat_map(|p| [p as f32; 3]).collect();
    let mut module = rlx_ir::rlxmodel::Module::new(vec![step, prefill, embed, image]);
    module.dims = vec![("n".into(), 0)];
    module.model = Some(json!({
        "name": "mrope-vlm", "family": "sum",
        "entries": {"generate": {
            "functions": ["step"],
            "sizes": [{"n": 16}],
            "tokenizer": "tokenizer.json",
            "vocab_size": 8,
            "inputs": {"input_ids": "token", "rope_cos": {"table": "rope_cos.table"}},
            "state": [["past_k_0", 1]],
            "logits": 0,
            "media": {
                "embed": {"function": "embed", "inputs": {"input_ids": "ids"}, "outputs": ["embeds"]},
                "prefill": {"function": "prefill_embeds",
                            "inputs": {"inputs_embeds": "prompt_embeds", "last": "last",
                                       "rope_cos": "rope_cos", "rope_sin": "rope_sin",
                                       "extra": "extra"},
                            "outputs": ["logits", "past_k_0"], "sizes": [12]},
                "mrope": {"sections": [1, 1, 1], "cos": "rope_cos.table", "sin": "rope_sin.table"},
                "kinds": {"image": {"token": 8, "input": "image", "grid": {"patch": 2},
                    "stages": [{"function": "image", "inputs": {"pixels": "image"},
                                "outputs": ["image_embeds", "extra"]}],
                    "output": "image_embeds", "rows": ["extra"]}},
            },
        }},
    }));
    let path = dir.join("vlm.rlxp");
    rlxp::save_package_module(
        &path,
        &module,
        &[
            Tensor::f32("next", vec![256, 8], &next),
            Tensor::f32("rope_cos.table", vec![32, 3], &table),
            Tensor::f32("rope_sin.table", vec![32, 3], &table),
        ],
        &[],
    )
    .unwrap();
    let mut m = open_source(path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    let image = Tensor::f32("image", vec![4, 6], &[0.0; 24]);
    let opts = json!({"max_tokens": 3});
    assert_eq!(
        m.generate_with_media("abPc", std::slice::from_ref(&image), &opts)
            .unwrap(),
        "gdg"
    );
    // Text alone after it: positions are rows again (each step's next id
    // from the row it fed: `c` at 2 gives 7, `h`; then rows 3 and 4 give
    // `c`, `f`).
    assert_eq!(m.generate_text("abc", &opts).unwrap(), "hcf");
    // An image of another size than the stage takes is refused.
    let e = m
        .generate_with_media(
            "abPc",
            &[Tensor::f32("image", vec![4, 4], &[0.0; 16])],
            &opts,
        )
        .unwrap_err()
        .to_string();
    assert!(e.contains("[4, 4]"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `synthesize`: text to audio by stages (each id two samples, plus the
/// voice's offset), and by a decoder that writes codec tokens first (next
/// id = the sum so far + 1: after `a`, `b c e a`; less the offset 1, within
/// a 4-code book, codes 0 1 3; a stage halves them).
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn synthesize_speaks_by_stages_and_by_a_codec_decoder() {
    use rlx_ir::{Graph, GraphExt, Op, Shape};
    let f = |d: &[usize]| Shape::new(d, DType::F32);
    let generalized = |b: &dyn Fn(usize) -> Graph| {
        rlx_ir::dynamic::generalize(&b(37), &b(53), &[(0, 37, 53)]).unwrap()
    };
    let _loader = tokenizer_lock();
    set_tokenizer_loader(|_bytes, _chat| Ok(Box::new(Letters)));
    let dir = std::env::temp_dir().join(format!("rlx-tts-entry-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();

    // By stages, with voices.
    let speak = generalized(&|n| {
        let mut g = Graph::new("speak");
        let ids = g.input("input_ids", Shape::new(&[n], DType::I32));
        let voice = g.input("voice", f(&[1]));
        let v = g.add_node(Op::Cast { to: DType::F32 }, vec![ids], f(&[n]));
        let col = g.reshape_(v, vec![n as i64, 1]);
        let two = g.concat_(vec![col, col], 1);
        let flat = g.reshape_(two, vec![(2 * n) as i64]);
        let out = g.add(flat, voice);
        g.set_outputs(vec![out]);
        g
    });
    let mut m1 = rlx_ir::rlxmodel::Module::new(vec![speak]);
    m1.dims = vec![("n".into(), 0)];
    m1.model = Some(
        json!({"name": "speaker", "family": "toy-tts", "entries": {"synthesize": {
            "tokenizer": "tokenizer.json", "sample_rate": 8000,
            "voices": {"low": "voice.low", "high": "voice.high"},
            "stages": [{"function": "speak", "inputs": {"input_ids": "ids", "voice": "voice"}, "outputs": ["audio"]}],
        }}}),
    );
    let p1 = dir.join("speak.rlxp");
    rlxp::save_package_module(
        &p1,
        &m1,
        &[
            Tensor::f32("voice.low", vec![1], &[0.5]),
            Tensor::f32("voice.high", vec![1], &[10.0]),
        ],
        &[],
    )
    .unwrap();
    let mut tts = open_source(p1.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    // By name; with none asked for, the first.
    assert_eq!(tts.info().metadata["voices"], json!(["high", "low"]));
    let a = tts.synthesize("abc", &json!({"voice": "high"})).unwrap();
    assert_eq!(
        (a.sample_rate, a.samples),
        (8000, vec![10.0, 10.0, 11.0, 11.0, 12.0, 12.0])
    );

    // By a codec decoder.
    let step = generalized(&|upper| {
        let mut g = Graph::new("step");
        let ids = g.input("input_ids", Shape::new(&[1], DType::I64));
        let past = g.input("past_k_0", f(&[1, upper, 1]));
        let v = g.add_node(Op::Cast { to: DType::F32 }, vec![ids], f(&[1]));
        let row = g.reshape_(v, vec![1, 1, 1]);
        let cache = g.concat(vec![past, row], 1, f(&[1, upper + 1, 1]));
        let table = g.param("next", f(&[256, 8]));
        let sum = g.reduce(cache, rlx_ir::op::ReduceOp::Sum, vec![1, 2], false, f(&[1]));
        let at = g.add_node(
            Op::Cast { to: DType::I64 },
            vec![sum],
            Shape::new(&[1], DType::I64),
        );
        let logits = g.gather(table, at, 0, f(&[1, 8]));
        g.set_outputs(vec![logits, cache]);
        g
    });
    let decode = generalized(&|n| {
        let mut g = Graph::new("decode_codes");
        let codes = g.input("codes", Shape::new(&[n], DType::I32));
        let v = g.add_node(Op::Cast { to: DType::F32 }, vec![codes], f(&[n]));
        let half = g.constant(0.5, DType::F32);
        let out = g.mul(v, half);
        g.set_outputs(vec![out]);
        g
    });
    let next: Vec<f32> = (0..256)
        .flat_map(|i| (0..8).map(move |j| if j == (i + 1) % 8 { 1.0 } else { 0.0 }))
        .collect();
    let mut m2 = rlx_ir::rlxmodel::Module::new(vec![step, decode]);
    m2.dims = vec![("n".into(), 0)];
    m2.model = Some(
        json!({"name": "codec-speaker", "family": "toy-tts", "entries": {"synthesize": {
            "tokenizer": "tokenizer.json", "sample_rate": 16000,
            "decoder": {"functions": ["step"], "sizes": [{"n": 16}], "tokenizer": "tokenizer.json",
                        "vocab_size": 8, "inputs": {"input_ids": "token"},
                        "state": [["past_k_0", 1]], "logits": 0},
            "token_offset": 1, "codebook": 4, "max_tokens": 4,
            "stages": [{"function": "decode_codes", "inputs": {"codes": "codes"}, "outputs": ["audio"]}],
        }}}),
    );
    let p2 = dir.join("codec.rlxp");
    rlxp::save_package_module(&p2, &m2, &[Tensor::f32("next", vec![256, 8], &next)], &[]).unwrap();
    let mut tts = open_source(p2.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
    assert_eq!(
        tts.synthesize("a", &json!({})).unwrap().samples,
        vec![0.0, 0.5, 1.5]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// An adapter package: its base with `W + scale · B·A` merged in as the
/// weights bind, for a param the base holds and for one it references.
/// `x = [1, 1]` through `x·W` with `W = I` gives `[1, 1]`; the update
/// `2 · [0, 1]ᵀ·[1, 0]` adds 2 at row 1, column 0: `[3, 1]`.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn an_adapter_package_merges_its_updates_into_its_base() {
    use rlx_ir::{Graph, Shape};
    let mut g = Graph::new("affine");
    let x = g.input("x", Shape::new(&[1, 2], DType::F32));
    let w = g.param("w", Shape::new(&[2, 2], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[1, 2], DType::F32));
    g.set_outputs(vec![y]);
    let dir = std::env::temp_dir().join(format!("rlx-adapter-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let identity = [1.0f32, 0.0, 0.0, 1.0];
    let update = |tensor: &str| rlxp::LoraUpdate {
        tensor: tensor.into(),
        rank: 1,
        a: vec![1.0, 0.0],
        b: vec![0.0, 1.0],
        scale: 2.0,
    };
    let run = |path: &std::path::Path| {
        let mut m = open_source(path.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
        m.run(&[Tensor::f32("x", vec![1, 2], &[1.0, 1.0])], &json!({}))
            .unwrap()[0]
            .to_f32()
            .unwrap()
    };

    // Held by the base.
    let base = dir.join("held.rlxp");
    rlxp::save_package(
        &base,
        &g,
        &[Tensor::f32("w", vec![2, 2], &identity)],
        &json!({"name": "held"}),
    )
    .unwrap();
    let adapter = dir.join("held-lora.rlxp");
    rlxp::save_adapter(&adapter, "held-lora", "held.rlxp", &[update("w")]).unwrap();
    assert_eq!(run(&base), vec![1.0, 1.0]);
    assert_eq!(run(&adapter), vec![3.0, 1.0]);
    // Two adapters on one base add up.
    let twice = dir.join("twice.rlxp");
    rlxp::save_adapter(&twice, "twice", "held-lora.rlxp", &[update("w")]).unwrap();
    assert_eq!(run(&twice), vec![5.0, 1.0]);

    // Referenced from a checkpoint, under the checkpoint's name.
    let bytes: Vec<u8> = identity.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &bytes).unwrap();
    let st = safetensors::serialize(vec![("lin.weight".to_string(), view)], None).unwrap();
    std::fs::write(dir.join("ckpt.safetensors"), st).unwrap();
    let base = dir.join("ref.rlxp");
    let model = json!({"name": "ref", "weights": [{"uri": "safetensors://ckpt.safetensors", "map": {"w": "lin.weight"}}]});
    rlxp::save_package(&base, &g, &[], &model).unwrap();
    let adapter = dir.join("ref-lora.rlxp");
    rlxp::save_adapter(&adapter, "ref-lora", "ref.rlxp", &[update("lin.weight")]).unwrap();
    assert_eq!(run(&adapter), vec![3.0, 1.0]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A package's f32 matmul weights quantized into it (GGUF Q8_0): the graph
/// reads them through `DequantMatMul`, the package holds a quarter-and-a-bit
/// of the bytes, and the output stays within Q8_0's error.
#[cfg(all(feature = "model-io", feature = "cpu"))]
#[test]
fn a_package_quantizes_its_matmul_weights_into_itself() {
    use rlx_ir::{Graph, Shape};
    let (k, n) = (64usize, 8usize);
    let mut g = Graph::new("proj");
    let x = g.input("x", Shape::new(&[1, 3, k], DType::F32));
    let w = g.param("w", Shape::new(&[k, n], DType::F32));
    let y = g.matmul(x, w, Shape::new(&[1, 3, n], DType::F32));
    g.set_outputs(vec![y]);
    let dir = std::env::temp_dir().join(format!("rlx-quantize-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let wv: Vec<f32> = (0..k * n)
        .map(|i| ((i * 37 % 97) as f32 / 97.0 - 0.5) * 0.2)
        .collect();
    let xv: Vec<f32> = (0..3 * k)
        .map(|i| (i * 11 % 29) as f32 / 29.0 - 0.5)
        .collect();
    let base = dir.join("proj.rlxp");
    rlxp::save_package(
        &base,
        &g,
        &[Tensor::f32("w", vec![k, n], &wv)],
        &json!({"name": "proj"}),
    )
    .unwrap();
    let q8 = dir.join("proj.q8_0.rlxp");
    let report = rlxp::quantize_package(&base, &q8, rlx_ir::quant::QuantScheme::GgufQ8_0).unwrap();
    assert_eq!(report.weights, 1);
    assert_eq!(report.f32_bytes, 4 * k * n);
    assert_eq!(report.quantized_bytes, k * n / 32 * 34);
    let text =
        rlx_ir::rlxmodel::write_module(&rlx_pkg::Package::open(&q8).unwrap().module().unwrap())
            .unwrap();
    assert!(
        text.contains("DequantMatMul") && !text.contains("MatMul %"),
        "{text}"
    );
    let run = |p: &std::path::Path| {
        let mut m = open_source(p.to_str().unwrap(), json!({"device": "cpu"})).unwrap();
        m.run(&[Tensor::f32("x", vec![1, 3, k], &xv)], &json!({}))
            .unwrap()[0]
            .to_f32()
            .unwrap()
    };
    let (want, got) = (run(&base), run(&q8));
    let err = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    assert!(
        err < 0.01 * scale,
        "Q8_0 off by {err} (outputs up to {scale})"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
