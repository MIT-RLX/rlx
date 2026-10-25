// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Picking a backend: what this build has, what suits a graph, and the ways to
// choose — by name, by cost model, by policy, by benchmark, per run.
//   odin run examples/backends -collection:rlx=.
#+feature dynamic-literals
package main

import "core:encoding/json"
import "core:fmt"
import "core:strings"
import rlx "rlx:rlx"

model :: proc() -> rlx.Graph {
	g := rlx.must(rlx.graph_new(name = "mlp"))
	x := rlx.must(rlx.input(g, "x", {4, 16}))
	rlx.set_outputs(g, {rlx.must(rlx.tanh(g, rlx.must(rlx.matmul(g, x, rlx.must(rlx.param(g, "w", {16, 16}))))))})
	return g
}

main :: proc() {
	w := make([]f32, 256)
	for i in 0 ..< 256 do w[i] = f32(i % 5) / 10 - 0.2
	xs := make([]f32, 64)
	for i in 0 ..< 64 do xs[i] = f32(i % 7) / 7
	x := map[string][]f32{"x" = xs}

	available := rlx.must(rlx.devices())
	fmt.printfln("devices: %s", strings.join(available, " "))
	fmt.printfln("metal available: %v", rlx.is_available("metal"))
	fmt.printfln("fastest for graph: %s", rlx.must(rlx.fastest_device_for(model())))
	reference: []f32
	for device in available {
		c := rlx.must(rlx.session_compile(rlx.must(rlx.session_new(device = device)), model()))
		rlx.compiled_set_param(c, "w", w)
		y := rlx.must(rlx.compiled_run(c, x))[0]
		if reference == nil do reference = y
		d: f32
		for v, i in y do d = max(d, abs(v - reference[i]))
		fmt.printfln("session %s: max|Δ| vs first = %.2e", device, d)
	}
	flex := rlx.must(rlx.flexible_session_new(policy = `{"prefer": ["metal", "cpu"]}`))
	fmt.printfln("flexible session picked: %s", rlx.must(rlx.compiled_device(rlx.must(rlx.flexible_session_compile(flex, model())))))
	runner := rlx.must(rlx.runner_new(model()))
	rlx.runner_set_param(runner, "w", w)
	fmt.printfln("runner warmed: %s", strings.join(rlx.must(rlx.runner_warm_all(runner)), " "))
	for row in rlx.must(rlx.runner_benchmark(runner, x, runs = 3)).(json.Array) {
		o := row.(json.Object)
		fmt.printfln("  bench %s: exec_ms=%v", o["device"].(json.String), o["exec_ms"])
	}
	fmt.printfln("runner ran on: %s", rlx.must(rlx.runner_run(runner, x)).device)
	fmt.printfln("run_try chain: %s", rlx.must(rlx.runner_run_try(runner, {"metal", "cpu"}, x)).device)
	router := rlx.must(rlx.router_new(model()))
	rlx.router_set_param(router, "w", w)
	fmt.printfln("router ran on: %s", rlx.must(rlx.router_run_chain(router, x)).device)
}
