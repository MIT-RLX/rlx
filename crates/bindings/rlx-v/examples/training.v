// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
// the same fit by hand with grad and an Optimizer.
//   v run examples/training.v [device]
import math
import os
import rlx

const w_true = [2.0, -1, 0.5]
const batch = 8

fn data(step int) map[string][]f64 {
	mut x := []f64{}
	mut y := []f64{}
	for i in 0 .. batch {
		mut s := 0.0
		for k in 0 .. 3 {
			v := f64((step * 31 + i * 7 + k * 3) % 17) / 8.5 - 1
			x << v
			s += v * w_true[k]
		}
		y << s
	}
	return {
		'x': x
		'y': y
	}
}

fn fmt(v []f64) string {
	return v.map('${it:.3f}').join(' ')
}

fn main() {
	device := if os.args.len > 1 { os.args[1] } else { 'cpu' }
	g := rlx.Graph.new(name: 'regression')!
	x := g.input('x', [batch, 3])!
	y := g.input('y', [batch, 1])!
	w := g.param('w', [3, 1])!
	d := g.sub(g.matmul(x, w)!, y)!
	g.set_outputs([g.mean(g.mul(d, d)!, [0, 1])!])!

	init := {
		'w': [0.0, 0, 0]
	}
	trainer := rlx.Trainer.new(g.clone()!, ['w'], init,
		options: '{"device": "${device}", "optimizer": {"kind": "adamw", "lr": 0.1}}'
	)!
	first := trainer.evaluate(data(0))!
	trainer.run(400, fn (s int) ?map[string][]f64 {
		return data(s)
	},
		lr:      fn (s int) f64 {
			return 0.1 * 0.5 * (1 + math.cos(math.pi * s / 400))
		}
		on_step: fn (s int, loss f64) bool {
			return loss > 1e-6
		}
	)!
	println('trainer loss: ${first:.4f} -> ${trainer.evaluate(data(0))!:.6f} in ${trainer.steps()!} steps')
	println('trainer w: ${fmt(trainer.params()!['w'])}')

	path := os.join_path(os.temp_dir(), 'rlx-example-train-v.gguf')
	written := trainer.save(path)!
	restored := rlx.Trainer.new(g.clone()!, ['w'], init, options: '{"device": "${device}"}')!
	info := restored.load(path)!
	println('checkpoint: ${written} tensors, restored step ${info.as_map()['step']!.int()}')
	os.rm(path)!
	restored.freeze(names: ['w'])!
	before := fmt(restored.params()!['w'])
	restored.step(data(1))!
	println('frozen: ${restored.frozen()!} unchanged: ${fmt(restored.params()!['w']) == before}')
	restored.unfreeze()!

	backward := rlx.Session.new(device: device)!.compile(rlx.grad(g, [w])!)! // outputs [loss, dw]
	opt := rlx.Optimizer.new(kind: 'sgd', options: '{"lr": 0.2}')!
	mut params := {
		'w': [0.0, 0, 0]
	}
	mut loss := 0.0
	for step in 0 .. 300 {
		backward.set_param('w', params['w'])!
		mut feed := data(step)
		feed['d_output'] = [1.0]
		outs := backward.run(feed)!
		loss = outs[0][0]
		opt.step(mut params, {
			'w': outs[1]
		})!
	}
	println('manual loss: ${loss:.6f} w: ${fmt(params['w'])}')
}
