// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Inference: build a graph, compile it for a device, set its weights, run it.
//   odin run examples/inference -collection:rlx=. -- [device]
// A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].
#+feature dynamic-literals
package main

import "core:fmt"
import "core:os"
import rlx "rlx:rlx"

weights :: proc(n, seed: int) -> []f32 {
	w := make([]f32, n)
	for i in 0 ..< n do w[i] = f32((i * 7919 + seed * 104729) % 1000) / 1000 - 0.5
	return w
}

main :: proc() {
	device := len(os.args) > 1 ? os.args[1] : "cpu"
	g := rlx.must(rlx.graph_new(name = "classifier"))
	x := rlx.must(rlx.input(g, "x", {1, 4}))
	hidden := rlx.must(rlx.relu(g, rlx.must(rlx.matmul(g, x, rlx.must(rlx.param(g, "w1", {4, 8}))))))
	out := rlx.must(rlx.softmax(g, rlx.must(rlx.matmul(g, hidden, rlx.must(rlx.param(g, "w2", {8, 3}))))))
	rlx.set_outputs(g, {out})

	model := rlx.must(rlx.session_compile(rlx.must(rlx.session_new(device = device)), g)) // consumes g
	fmt.printfln("compiled on: %s", rlx.must(rlx.compiled_device(model)))
	rlx.must(rlx.compiled_set_params(model, map[string][]f32{"w1" = weights(32, 1), "w2" = weights(24, 2)}))
	probs := rlx.must(rlx.compiled_run(model, map[string][]f32{"x" = {0.5, -1, 2, 0.25}}))[0]
	fmt.printf("probs:")
	sum: f32
	for p in probs {
		fmt.printf(" %.4f", p)
		sum += p
	}
	fmt.printfln("\nprobs sum: %.4f", sum)
	fmt.printfln("output shapes: %v dtypes: %v", rlx.must(rlx.compiled_output_shapes(model)), rlx.must(rlx.compiled_output_dtypes(model)))
}
