// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Distributed: each rank is a process; start one per rank with the same peers
// (here three on loopback; across machines, one per host):
//   ./distributed 1 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
//   ./distributed 2 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
//   ./distributed 0 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502
// (build: v -o distributed examples/distributed.v). Rank 0 coordinates:
// collectives, pipelined inference (a stage shipped per worker), data-parallel
// training. With no arguments it is a mesh of one.
import math
import os
import rlx

const per_rank = 16

fn blob(rank int, name string, values []f64) !string {
	path := os.join_path(os.temp_dir(), 'rlx-dist-v-${rank}-${name}')
	mut bytes := []u8{}
	for v in values {
		bits := math.f32_bits(f32(v))
		bytes << [u8(bits), u8(bits >> 8), u8(bits >> 16), u8(bits >> 24)]
	}
	os.write_file_array(path, bytes)!
	return 'file://${path}'
}

fn main() {
	rank := if os.args.len > 1 { os.args[1].int() } else { 0 }
	world := if os.args.len > 2 { os.args[2].int() } else { 1 }
	peers := if os.args.len > 3 {
		', "peers": [' + os.args[3].split(',').map('"${it}"').join(',') + ']'
	} else {
		''
	}
	g := rlx.Group.connect(rank, world, options: '{"device": "cpu"${peers}}')!
	total := g.all_reduce([f64(rank + 1)], op: 'sum')!
	gathered := g.all_gather([f64(rank + 1)])!
	shared_ := g.broadcast(0, [if rank == 0 { 42.0 } else { 0.0 }])! // the root's length
	if rank == 0 {
		println('mesh: rank ${g.rank()!} of ${g.world()!}, leader: ${g.is_leader()!}')
		println('all_reduce sum: ${total.map(it.str()).join(',')}')
		println('all_gather: ${gathered.map(it.str()).join(',')}')
		println('broadcast: ${shared_.map(it.str()).join(',')}')
	}
	if world == 1 {
		println('a mesh of one: pass rank world peers to start more ranks')
		return
	}
	if rank != 0 {
		g.serve_worker(options: '{"steps": 2}')!
		r := g.serve_trainer()!
		w := r.as_map()['params']!.as_map()['w']!.as_array().map(it.f64())
		g.send(0, 1, w)! // rank 0 compares
		return
	}

	stage_w := [[]f64{len: 16, init: f64(index * 5 % 7) / 7 - 0.3},
		[]f64{len: 16, init: f64(index * 3 % 5) / 5 - 0.2}]
	mut names := []string{}
	for c in g.collect_caps()!.as_array() {
		m := c.as_map()
		names << 'rank ${m['rank']!.int()} on ${m['platform']!.str()}'
	}
	println('workers: ${names.join(', ')}')
	stage := rlx.Graph.new(name: 'stage')!
	stage.set_outputs([stage.relu(stage.matmul(stage.input('x', [1, 4])!, stage.param('w', [4, 4])!)!)!])!
	for r in 1 .. world {
		uri := blob(rank, 'w${r}.bin', stage_w[(r - 1) % 2])!
		g.ship_stage(r, stage, '{"device": "cpu", "weights": {"w": "${uri}"}}')!
	}
	for xv in [[1.0, -1, 0.5, 2], [0.25, 0.5, -0.75, 1]] {
		g.send_activation(1, xv)!
		println('pipeline: ${g.recv_activation(world - 1)!.map('${it:.3f}').join(' ')}')
	}

	g.collect_caps()!
	lr := rlx.Graph.new(name: 'lr')!
	big_x := lr.input('X', [4, 3])!
	y := lr.input('y', [4, 1])!
	wv := lr.param('w', [3, 1])!
	d := lr.sub(lr.matmul(big_x, wv)!, y)!
	lr.set_outputs([lr.mean(lr.mul(d, d)!, [0, 1])!])!
	bwd := rlx.grad(lr, [wv])!
	w_true := [1.5, -2, 0.5]
	n := per_rank * world
	x_all := []f64{len: n * 3, init: f64(index * 37 % 23) / 11.5 - 1}
	y_all := []f64{len: n, init: x_all[index * 3] * w_true[0] + x_all[index * 3 + 1] * w_true[1] +
		x_all[index * 3 + 2] * w_true[2]}
	ux := blob(rank, 'X.bin', x_all)!
	uy := blob(rank, 'y.bin', y_all)!
	uw := blob(rank, 'w0.bin', [0.0, 0, 0])!
	lrs := []string{len: 40, init: '0.2'}.join(',')
	spec := fn [ux, uy, uw, lrs] (r int) string {
		return '{"params": [{"name": "w", "uri": "${uw}"}], "data": [' +
			'{"input": "X", "uri": "${ux}", "elem": 3, "shard_start": ${r * per_rank}, "shard_len": ${per_rank}},' +
			'{"input": "y", "uri": "${uy}", "elem": 1, "shard_start": ${r * per_rank}, "shard_len": ${per_rank}}],' +
			'"batch": 4, "lr_per_epoch": [${lrs}], "device": "cpu"}'
	}
	for r in 1 .. world {
		g.ship_train(r, bwd, spec(r))!
	}
	mine := g.run_train(bwd, spec(0))!.as_map()
	metrics := mine['metrics']!.as_map()
	println('training loss: ${metrics['first_loss']!.f64():.4f} -> ${metrics['last_loss']!.f64():.2e}')
	w0 := mine['params']!.as_map()['w']!.as_array().map(it.f64())
	println('rank 0 w: ${w0.map('${it:.3f}').join(' ')} (true: 1.5 -2 0.5)')
	for r in 1 .. world {
		theirs := g.recv(r, 1)!
		mut agree := true
		for k in 0 .. w0.len {
			agree = agree && math.abs(theirs[k] - w0[k]) < 1e-5
		}
		println('rank ${r} w agrees: ${agree}')
	}
	for name in ['w1.bin', 'w2.bin', 'X.bin', 'y.bin', 'w0.bin'] {
		os.rm(os.join_path(os.temp_dir(), 'rlx-dist-v-${rank}-${name}')) or {}
	}
}
