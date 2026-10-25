// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The Odin binding against the real library; exits non-zero on the first
// failure.
//   odin run tests -collection:rlx=.
#+feature dynamic-literals
package main

import "core:encoding/json"
import "core:fmt"
import "core:os"
import "core:slice"
import "core:strings"
import rlx "rlx:rlx"

checks := 0

check :: proc(ok: bool, what: string) {
	checks += 1
	if !ok {
		fmt.eprintln("failed:", what)
		os.exit(1)
	}
}

loss_graph :: proc() -> (rlx.Graph, rlx.Node) {
	g := rlx.must(rlx.graph_new(name = "mse"))
	x := rlx.must(rlx.input(g, "x", {1, 2}))
	y := rlx.must(rlx.input(g, "y", {1, 2}))
	w := rlx.must(rlx.param(g, "w", {2, 2}))
	d := rlx.must(rlx.sub(g, rlx.must(rlx.matmul(g, x, w)), y))
	rlx.set_outputs(g, {rlx.must(rlx.mean(g, rlx.must(rlx.mul(g, d, d)), {0, 1}))})
	return g, w
}

main :: proc() {
	feed := map[string][]f32{"x" = {1, 2}, "y" = {0, 0}}
	{
		g, _ := loss_graph()
		m := rlx.must(rlx.session_compile(rlx.must(rlx.session_new()), g))
		check(rlx.is_consumed(g), "compile consumes")
		rlx.compiled_set_param(m, "w", {1, 0, 0, 1})
		check(rlx.must(rlx.compiled_run(m, feed))[0][0] == 2.5, "loss 2.5")
		check(rlx.must(rlx.compiled_device(m)) == "cpu", "device")
	}
	{
		_, err := rlx.relu(rlx.must(rlx.graph_new()), 42)
		check(err == .Range && strings.contains(rlx.last_error(), "node id 42"), "range error")
	}
	{
		g, w := loss_graph()
		bwd := rlx.must(rlx.session_compile(rlx.must(rlx.session_new()), rlx.must(rlx.grad(g, {w}))))
		rlx.compiled_set_param(bwd, "w", {1, 0, 0, 1})
		check(rlx.must(rlx.compiled_run(bwd, map[string][]f32{"x" = {1, 2}, "y" = {0, 0}, "d_output" = {1}}))[1][3] == 4, "dw")
		t := rlx.must(rlx.trainer_new(rlx.must(rlx.clone(g)), {"w"}, map[string][]f32{"w" = {0, 0, 0, 0}}, options = `{"optimizer": {"kind": "adamw", "lr": 0.1}}`))
		r := rlx.must(rlx.trainer_run(t, 300,
			// The batch must outlive the callback: slice literals live on its stack.
			proc(s: int) -> (map[string][]f32, bool) {
				m := make(map[string][]f32)
				m["x"] = slice.clone([]f32{1, 1})
				m["y"] = slice.clone([]f32{1, -1})
				return m, true
			},
			on_step = proc(s: int, l: f32) -> bool {return l > 1e-6}))
		check((r.last_loss.? or_else 1) < 1e-4, "trainer converges")
		opt := rlx.must(rlx.optimizer_new(kind = "sgd", options = `{"lr": 0.5}`))
		p := map[string][]f32{"p" = {1, -1}}
		rlx.optimizer_step(opt, p, map[string][]f32{"p" = {2, -2}})
		check(p["p"][0] == 0 && p["p"][1] == 0, "optimizer in place")
	}
	{
		path := strings.concatenate({tmp_dir(), "/rlx-odin-test.gguf"})
		gw := rlx.must(rlx.gguf_writer_new())
		rlx.gguf_writer_add_f32(gw, "w", {4}, "F32", {0.5, -1, 2, 0.25})
		check(rlx.must(rlx.gguf_writer_write(gw, path, architecture = "demo")) == 1, "write")
		f := rlx.must(rlx.gguf_open(path))
		check((rlx.must(rlx.gguf_architecture(f)).? or_else "") == "demo", "architecture")
		check(rlx.must(rlx.gguf_tensor(f, "w"))[2] == 2, "tensor")
		check(rlx.must(rlx.gguf_info(f, "w")).(json.Object)["dtype"].(json.String) == "F32", "info")
		os.remove(path)
	}
	{
		check(rlx.must(rlx.devices())[0] == "cpu" && rlx.is_available("cpu"), "cpu")
		g, _ := loss_graph()
		runner := rlx.must(rlx.runner_new(g, policy = `{"only": ["cpu"]}`))
		rlx.runner_set_param(runner, "w", {1, 0, 0, 1})
		res := rlx.must(rlx.runner_run(runner, feed))
		check(res.device == "cpu" && res.outputs[0][0] == 2.5, "runner")
		check(rlx.must(rlx.group_all_reduce(rlx.must(rlx.group_connect(0, 1)), {2, 3}, op = "max"))[1] == 3, "all_reduce")
	}
	fmt.printfln("test_rlx: %d checks passed", checks)
}

tmp_dir :: proc() -> string {
	t := os.get_env("TMPDIR", context.allocator)
	return t == "" ? "/tmp" : t
}
