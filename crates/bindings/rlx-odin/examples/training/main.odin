// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
// the same fit by hand with grad and an Optimizer.
//   odin run examples/training -collection:rlx=. -- [device]
#+feature dynamic-literals
package main

import "core:encoding/json"
import "core:fmt"
import "core:math"
import "core:os"
import "core:strings"
import rlx "rlx:rlx"

W_TRUE :: [3]f32{2, -1, 0.5}
BATCH :: 8

data :: proc(step: int) -> map[string][]f32 {
	x := make([]f32, BATCH * 3)
	y := make([]f32, BATCH)
	w := W_TRUE
	for i in 0 ..< BATCH {
		for k in 0 ..< 3 {
			v := f32((step * 31 + i * 7 + k * 3) % 17) / 8.5 - 1
			x[i * 3 + k] = v
			y[i] += v * w[k]
		}
	}
	m := make(map[string][]f32)
	m["x"] = x
	m["y"] = y
	return m
}

fmt3 :: proc(v: []f32) -> string {
	return fmt.tprintf("%.3f %.3f %.3f", v[0], v[1], v[2])
}

main :: proc() {
	device := len(os.args) > 1 ? os.args[1] : "cpu"
	g := rlx.must(rlx.graph_new(name = "regression"))
	x := rlx.must(rlx.input(g, "x", {BATCH, 3}))
	y := rlx.must(rlx.input(g, "y", {BATCH, 1}))
	w := rlx.must(rlx.param(g, "w", {3, 1}))
	d := rlx.must(rlx.sub(g, rlx.must(rlx.matmul(g, x, w)), y))
	rlx.set_outputs(g, {rlx.must(rlx.mean(g, rlx.must(rlx.mul(g, d, d)), {0, 1}))})

	opts := fmt.tprintf(`{{"device": "%s", "optimizer": {{"kind": "adamw", "lr": 0.1}}}}`, device)
	trainer := rlx.must(rlx.trainer_new(rlx.must(rlx.clone(g)), {"w"}, map[string][]f32{"w" = {0, 0, 0}}, options = opts))
	first := rlx.must(rlx.trainer_evaluate(trainer, data(0)))
	rlx.must(rlx.trainer_run(trainer, 400,
		proc(s: int) -> (map[string][]f32, bool) {return data(s), true},
		lr = proc(s: int) -> f32 {return 0.1 * 0.5 * (1 + math.cos(math.PI * f32(s) / 400))},
		on_step = proc(s: int, loss: f32) -> bool {return loss > 1e-6}))
	fmt.printfln("trainer loss: %.4f -> %.6f in %d steps", first, rlx.must(rlx.trainer_evaluate(trainer, data(0))), rlx.must(rlx.trainer_steps(trainer)))
	fmt.printfln("trainer w: %s", fmt3(rlx.must(rlx.trainer_params(trainer))["w"]))

	path := strings.concatenate({tmp_dir(), "/rlx-example-train-odin.gguf"})
	written := rlx.must(rlx.trainer_save(trainer, path))
	restored := rlx.must(rlx.trainer_new(rlx.must(rlx.clone(g)), {"w"}, map[string][]f32{"w" = {0, 0, 0}}, options = fmt.tprintf(`{{"device": "%s"}}`, device)))
	info := rlx.must(rlx.trainer_load(restored, path))
	fmt.printfln("checkpoint: %d tensors, restored step %v", written, info.(json.Object)["step"])
	os.remove(path)
	rlx.must(rlx.trainer_freeze(restored, names = {"w"}))
	before := fmt3(rlx.must(rlx.trainer_params(restored))["w"])
	rlx.must(rlx.trainer_step(restored, data(1)))
	fmt.printfln("frozen: %v unchanged: %v", rlx.must(rlx.trainer_frozen(restored)), fmt3(rlx.must(rlx.trainer_params(restored))["w"]) == before)
	rlx.must(rlx.trainer_unfreeze(restored))

	backward := rlx.must(rlx.session_compile(rlx.must(rlx.session_new(device = device)), rlx.must(rlx.grad(g, {w})))) // outputs [loss, dw]
	opt := rlx.must(rlx.optimizer_new(kind = "sgd", options = `{"lr": 0.2}`))
	params := map[string][]f32{"w" = {0, 0, 0}}
	loss: f32
	for step in 0 ..< 300 {
		rlx.compiled_set_param(backward, "w", params["w"])
		feed := data(step)
		feed["d_output"] = {1}
		outs := rlx.must(rlx.compiled_run(backward, feed))
		loss = outs[0][0]
		rlx.optimizer_step(opt, params, map[string][]f32{"w" = outs[1]})
	}
	fmt.printfln("manual loss: %.6f w: %s", loss, fmt3(params["w"]))
}

tmp_dir :: proc() -> string {
	t := os.get_env("TMPDIR", context.allocator)
	return t == "" ? "/tmp" : t
}
