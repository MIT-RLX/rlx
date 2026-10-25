// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Models: any provider's model, opened by path or id and run by name. A graph
// saved as an .rlxp package runs anywhere; a plugin's models (a downstream
// collection's LLMs, or the reference plugin's demo:count) load through RLX_MODEL_PLUGINS.
//
//	RLX_MODEL_PLUGINS=target/debug/librlx_model_plugin.dylib go run ./examples/models
package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	rlx "github.com/MIT-RLX/rlx/crates/bindings/rlx-go"
)

func must[T any](v T, err error) T {
	if err != nil {
		log.Fatal(err)
	}
	return v
}

func main() {
	g := must(rlx.NewGraph(&rlx.GraphOpts{Name: rlx.Ptr("affine")}))
	x := must(g.Input("x", []int{1, 2}, nil))
	must(0, g.SetOutputs([]rlx.Node{must(g.Matmul(x, must(g.Param("w", []int{2, 2}, nil))))}))
	path := filepath.Join(os.TempDir(), "rlx-example-go.rlxp")
	must(0, rlx.SaveModel(g, path, map[string][]float32{"w": {2, 0, 0, 3}}, &rlx.SaveModelOpts{
		Model: map[string]any{"task": "graph", "family": "affine", "outputs": []string{"y"}},
	}))
	defer os.Remove(path)

	m := must(rlx.ModelOpen(path, &rlx.ModelOpenOpts{Options: map[string]any{"device": "cpu"}}))
	defer m.Close()
	info := must(m.Info())
	fmt.Printf("model: %v (%v) on %v\n", info["family"], info["task"], info["device"])
	y := must(m.Run(map[string][]float32{"x": {1, 1}}, nil))[0]
	fmt.Printf("model run: %.3f %.3f\n", y[0], y[1])

	lm := must(rlx.ModelOpen("demo:count", nil))
	defer lm.Close()
	lmInfo := must(lm.Info())
	fmt.Printf("plugin model: %v (%v)\n", lmInfo["family"], lmInfo["task"])
	fmt.Println("model text:", must(lm.GenerateText("c", &rlx.ModelGenerateTextOpts{Options: map[string]any{"max_tokens": 4}})))
	ids := must(lm.Generate([]uint32{2}, &rlx.ModelGenerateOpts{MaxTokens: rlx.Ptr(3)}))
	parts := []string{}
	for _, id := range ids {
		parts = append(parts, fmt.Sprint(id))
	}
	fmt.Println("model ids:", strings.Join(parts, " "))

	// By task: a chat, speech out and back in, embeddings.
	msgs := []map[string]string{{"role": "user", "content": "c"}}
	fmt.Println("chat:", must(lm.Chat(msgs, &rlx.ModelChatOpts{Options: map[string]any{"max_tokens": 4}})))
	voice := must(rlx.ModelOpen("demo:voice", nil))
	defer voice.Close()
	speech := must(voice.Synthesize("hello", nil))
	fmt.Printf("voice: %d samples at %d Hz\n", len(speech.Audio), speech.SampleRate)
	ear := must(rlx.ModelOpen("demo:speech", nil))
	defer ear.Close()
	fmt.Println("heard:", must(ear.Transcribe(speech.Audio, &rlx.ModelTranscribeOpts{SampleRate: rlx.Ptr(speech.SampleRate)})))
	embedder := must(rlx.ModelOpen("demo:embed", nil))
	defer embedder.Close()
	values := []string{}
	for _, row := range must(embedder.Embed([]string{"hello", "sky"}, nil)).([]any) {
		for _, v := range row.([]any) {
			values = append(values, fmt.Sprint(int(v.(float64))))
		}
	}
	fmt.Println("embed:", strings.Join(values, " "))
}
