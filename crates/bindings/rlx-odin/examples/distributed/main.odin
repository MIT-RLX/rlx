// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Distributed: a three-rank mesh on loopback, one thread per rank (across
// machines, one process per host with the same peers). Rank 0 coordinates:
// collectives, pipelined inference (a stage shipped per worker),
// data-parallel training where every replica ends identical.
//   odin run examples/distributed -collection:rlx=.
#+feature dynamic-literals
package main

import "core:encoding/json"
import "core:fmt"
import "core:os"
import "core:slice"
import "core:strings"
import "core:thread"
import rlx "rlx:rlx"

WORLD :: 3
PER_RANK :: 16

peers_json: string

connect :: proc(rank: int) -> rlx.Group {
	return rlx.must(rlx.group_connect(rank, WORLD, options = fmt.tprintf(`{{"peers": %s, "device": "cpu"}}`, peers_json)))
}

collectives :: proc(g: rlx.Group, rank: int) {
	total := rlx.must(rlx.group_all_reduce(g, {f32(rank + 1)}, op = "sum"))
	gathered := rlx.must(rlx.group_all_gather(g, {f32(rank + 1)}))
	shared := rlx.must(rlx.group_broadcast(g, 0, {rank == 0 ? 42 : 0})) // the root's length
	if rank == 0 {
		fmt.printfln("mesh: rank %d of %d, leader: %v", rlx.must(rlx.group_rank(g)), rlx.must(rlx.group_world(g)), rlx.must(rlx.group_is_leader(g)))
		fmt.printfln("all_reduce sum: %v\nall_gather: %v\nbroadcast: %v", total[0], gathered, shared[0])
	}
}

worker :: proc(t: ^thread.Thread) {
	rank := t.user_index
	g := connect(rank)
	collectives(g, rank)
	rlx.must(rlx.group_serve_worker(g, options = `{"steps": 2}`))
	r := rlx.must(rlx.group_serve_trainer(g))
	ws := r.(json.Object)["params"].(json.Object)["w"].(json.Array)
	w := make([]f32, len(ws))
	for v, i in ws do w[i] = f32(v.(json.Float) or_else f64(v.(json.Integer) or_else 0))
	rlx.group_send(g, 0, 1, w) // rank 0 compares
}

blob :: proc(dir, name: string, values: []f32) -> string {
	path := strings.concatenate({dir, "/", name})
	_ = os.write_entire_file(path, slice.to_bytes(values))
	return strings.concatenate({"file://", path})
}

main :: proc() {
	peers := rlx.must(rlx.loopback_peers(WORLD))
	peers_json = fmt.aprintf("[%s]", strings.join(slice.mapper(peers, proc(p: string) -> string {return fmt.aprintf("%q", p)}), ","))
	workers: [WORLD - 1]^thread.Thread
	for r in 1 ..< WORLD {
		workers[r - 1] = thread.create(worker)
		workers[r - 1].user_index = r
		thread.start(workers[r - 1])
	}
	g := connect(0)
	collectives(g, 0)
	tmp := strings.concatenate({tmp_dir(), "/rlx-dist-odin"})
	os.make_directory(tmp)

	stage_w: [2][16]f32
	for i in 0 ..< 16 {
		stage_w[0][i] = f32(i * 5 % 7) / 7 - 0.3
		stage_w[1][i] = f32(i * 3 % 5) / 5 - 0.2
	}
	caps := rlx.must(rlx.group_collect_caps(g)).(json.Array)
	fmt.printf("workers: ")
	for c, i in caps {
		o := c.(json.Object)
		fmt.printf("%srank %v on %s", i > 0 ? ", " : "", o["rank"], o["platform"].(json.String))
	}
	fmt.println()
	stage := rlx.must(rlx.graph_new(name = "stage"))
	rlx.set_outputs(stage, {rlx.must(rlx.relu(stage, rlx.must(rlx.matmul(stage, rlx.must(rlx.input(stage, "x", {1, 4})), rlx.must(rlx.param(stage, "w", {4, 4}))))))})
	for r in 1 ..< WORLD {
		uri := blob(tmp, fmt.tprintf("w%d.bin", r), stage_w[(r - 1) % 2][:])
		rlx.group_ship_stage(g, r, stage, fmt.tprintf(`{{"device": "cpu", "weights": {{"w": "%s"}}}}`, uri))
	}
	inputs := [2][4]f32{{1, -1, 0.5, 2}, {0.25, 0.5, -0.75, 1}}
	for &xv in inputs {
		rlx.group_send_activation(g, 1, xv[:])
		y := rlx.must(rlx.group_recv_activation(g, WORLD - 1))
		fmt.printfln("pipeline: %.3f %.3f %.3f %.3f", y[0], y[1], y[2], y[3])
	}

	rlx.must(rlx.group_collect_caps(g))
	lr := rlx.must(rlx.graph_new(name = "lr"))
	bx := rlx.must(rlx.input(lr, "X", {4, 3}))
	by := rlx.must(rlx.input(lr, "y", {4, 1}))
	wv := rlx.must(rlx.param(lr, "w", {3, 1}))
	d := rlx.must(rlx.sub(lr, rlx.must(rlx.matmul(lr, bx, wv)), by))
	rlx.set_outputs(lr, {rlx.must(rlx.mean(lr, rlx.must(rlx.mul(lr, d, d)), {0, 1}))})
	bwd := rlx.must(rlx.grad(lr, {wv}))
	w_true := [3]f32{1.5, -2, 0.5}
	N :: PER_RANK * WORLD
	x_all := make([]f32, N * 3)
	y_all := make([]f32, N)
	for i in 0 ..< N * 3 do x_all[i] = f32(i * 37 % 23) / 11.5 - 1
	for i in 0 ..< N do for k in 0 ..< 3 do y_all[i] += x_all[i * 3 + k] * w_true[k]
	ux, uy, uw := blob(tmp, "X.bin", x_all), blob(tmp, "y.bin", y_all), blob(tmp, "w0.bin", {0, 0, 0})
	lrs := strings.repeat("0.2,", 39)
	spec :: proc(r: int, ux, uy, uw, lrs: string) -> string {
		return fmt.tprintf(
			`{{"params": [{{"name": "w", "uri": "%s"}}], "data": [{{"input": "X", "uri": "%s", "elem": 3, "shard_start": %d, "shard_len": %d}}, {{"input": "y", "uri": "%s", "elem": 1, "shard_start": %d, "shard_len": %d}}], "batch": 4, "lr_per_epoch": [%s0.2], "device": "cpu"}}`,
			uw, ux, r * PER_RANK, PER_RANK, uy, r * PER_RANK, PER_RANK, lrs)
	}
	for r in 1 ..< WORLD do rlx.group_ship_train(g, r, bwd, spec(r, ux, uy, uw, lrs))
	mine := rlx.must(rlx.group_run_train(g, bwd, spec(0, ux, uy, uw, lrs))).(json.Object)
	metrics := mine["metrics"].(json.Object)
	fmt.printfln("training loss: %.4f -> %.2e", metrics["first_loss"].(json.Float), metrics["last_loss"].(json.Float))
	ws := mine["params"].(json.Object)["w"].(json.Array)
	w0: [3]f32
	for v, i in ws do w0[i] = f32(v.(json.Float) or_else f64(v.(json.Integer) or_else 0))
	fmt.printfln("rank 0 w: %.3f %.3f %.3f (true: 1.5 -2 0.5)", w0[0], w0[1], w0[2])
	for r in 1 ..< WORLD {
		theirs := rlx.must(rlx.group_recv(g, r, 1))
		agree := true
		for k in 0 ..< 3 do agree = agree && abs(theirs[k] - w0[k]) < 1e-5
		fmt.printfln("rank %d w agrees: %v", r, agree)
	}
	for t in workers do thread.join(t)
	os.remove_all(tmp)
}

tmp_dir :: proc() -> string {
	t := os.get_env("TMPDIR", context.allocator)
	return t == "" ? "/tmp" : t
}
