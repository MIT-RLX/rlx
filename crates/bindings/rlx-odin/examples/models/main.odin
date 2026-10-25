// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Models: any provider's model, opened by path or id and run by name. A graph
// saved as an .rlxp package runs anywhere; a plugin's models (a downstream
// collection's LLMs, or the reference plugin's demo:count) load through
// RLX_MODEL_PLUGINS.
//   RLX_MODEL_PLUGINS=…/librlx_model_plugin.so odin run examples/models -collection:rlx=.
#+feature dynamic-literals
package main

import "core:encoding/json"
import "core:fmt"
import "core:os"
import "core:strings"
import rlx "rlx:rlx"

tmp_dir :: proc() -> string {
	t := os.get_env("TMPDIR", context.allocator)
	return t if t != "" else "/tmp"
}

main :: proc() {
	g := rlx.must(rlx.graph_new(name = "affine"))
	x := rlx.must(rlx.input(g, "x", {1, 2}))
	rlx.set_outputs(g, {rlx.must(rlx.matmul(g, x, rlx.must(rlx.param(g, "w", {2, 2}))))})
	path := strings.concatenate({tmp_dir(), "/rlx-example-odin.rlxp"})
	meta := `{"task": "graph", "family": "affine", "outputs": ["y"]}`
	if err := rlx.save_model(g, path, map[string][]f32{"w" = {2, 0, 0, 3}}, model = meta); err != .None {
		panic(rlx.last_error())
	}

	m := rlx.must(rlx.model_open(path, options = `{"device": "cpu"}`))
	info := rlx.must(rlx.model_info(m)).(json.Object)
	fmt.printfln("model: %s (%s) on %s", info["family"].(json.String), info["task"].(json.String), info["device"].(json.String))
	y := rlx.must(rlx.model_run(m, map[string][]f32{"x" = {1, 1}}))[0]
	fmt.printfln("model run: %.3f %.3f", y[0], y[1])
	rlx.model_close(m)
	os.remove(path)

	lm := rlx.must(rlx.model_open("demo:count"))
	lm_info := rlx.must(rlx.model_info(lm)).(json.Object)
	fmt.printfln("plugin model: %s (%s)", lm_info["family"].(json.String), lm_info["task"].(json.String))
	fmt.printfln("model text: %s", rlx.must(rlx.model_generate_text(lm, "c", options = `{"max_tokens": 4}`)))
	ids := make([]rlx.Node, 1)
	ids[0] = 2
	fmt.printf("model ids:")
	for id in rlx.must(rlx.model_generate(lm, ids, max_tokens = 3)) do fmt.printf(" %d", id)
	fmt.println()

	// By task: a chat, speech out and back in, embeddings.
	reply := rlx.must(rlx.model_chat(lm, `[{"role": "user", "content": "c"}]`, options = `{"max_tokens": 4}`))
	fmt.printfln("chat: %s", reply)
	rlx.model_close(lm)
	voice := rlx.must(rlx.model_open("demo:voice"))
	speech := rlx.must(rlx.model_synthesize(voice, "hello"))
	fmt.printfln("voice: %d samples at %d Hz", len(speech.audio), speech.sample_rate)
	ear := rlx.must(rlx.model_open("demo:speech"))
	fmt.printfln("heard: %s", rlx.must(rlx.model_transcribe(ear, speech.audio, sample_rate = int(speech.sample_rate))))
	embedder := rlx.must(rlx.model_open("demo:embed"))
	texts := make([]string, 2)
	texts[0], texts[1] = "hello", "sky"
	fmt.printf("embed:")
	for row in rlx.must(rlx.model_embed(embedder, texts)).(json.Array) {
		for v in row.(json.Array) do fmt.printf(" %d", int(v.(json.Float) or_else f64(v.(json.Integer) or_else 0)))
	}
	fmt.println()
	rlx.model_close(voice)
	rlx.model_close(ear)
	rlx.model_close(embedder)
}
