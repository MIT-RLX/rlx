// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The V binding against the real library; exits non-zero on the first failure.
import os
import rlx

struct Checks {
mut:
	n int
}

fn (mut c Checks) check(ok bool, what string) {
	c.n++
	if !ok {
		eprintln('failed: ${what}')
		exit(1)
	}
}

fn loss_graph() !(&rlx.Graph, int) {
	g := rlx.Graph.new(name: 'mse')!
	x := g.input('x', [1, 2])!
	y := g.input('y', [1, 2])!
	w := g.param('w', [2, 2])!
	d := g.sub(g.matmul(x, w)!, y)!
	g.set_outputs([g.mean(g.mul(d, d)!, [0, 1])!])!
	return g, w
}

fn main() {
	mut c := Checks{}
	feed := {
		'x': [1.0, 2]
		'y': [0.0, 0]
	}
	{
		g, _ := loss_graph()!
		m := rlx.Session.new()!.compile(g)!
		c.check(g.is_consumed(), 'compile consumes')
		m.set_param('w', [1.0, 0, 0, 1])!
		c.check(m.run(feed)![0][0] == 2.5, 'loss 2.5')
		c.check(m.device()! == 'cpu', 'device')
	}
	{
		g := rlx.Graph.new()!
		if _ := g.relu(42) {
			c.check(false, 'range error raised')
		} else {
			c.check(err is rlx.RlxError && (err as rlx.RlxError).kind == 'range', 'range error')
		}
	}
	{
		g, w := loss_graph()!
		bwd := rlx.Session.new()!.compile(rlx.grad(g, [w])!)!
		bwd.set_param('w', [1.0, 0, 0, 1])!
		c.check(bwd.run({
			'x':        [1.0, 2]
			'y':        [0.0, 0]
			'd_output': [1.0]
		})![1][3] == 4, 'dw')
		t := rlx.Trainer.new(g.clone()!, ['w'], {
			'w': [0.0, 0, 0, 0]
		}, options: '{"optimizer": {"kind": "adamw", "lr": 0.1}}')!
		r := t.run(300, fn (s int) ?map[string][]f64 {
			return {
				'x': [1.0, 1]
				'y': [1.0, -1]
			}
		}, on_step: fn (s int, l f64) bool {
			return l > 1e-6
		})!
		c.check((r.last_loss or { 1.0 }) < 1e-4, 'trainer converges')
		opt := rlx.Optimizer.new(kind: 'sgd', options: '{"lr": 0.5}')!
		mut p := {
			'p': [1.0, -1]
		}
		opt.step(mut p, {
			'p': [2.0, -2]
		})!
		c.check(p['p'] == [0.0, 0], 'optimizer in place')
	}
	{
		path := os.join_path(os.temp_dir(), 'rlx-v-test.gguf')
		gw := rlx.GgufWriter.new()!
		gw.add_f32('w', [4], 'F32', [0.5, -1, 2, 0.25])!
		c.check(gw.write(path, architecture: 'demo')! == 1, 'write')
		f := rlx.Gguf.open(path)!
		c.check(f.architecture()! == 'demo', 'architecture')
		c.check(f.tensor('w')![2] == 2, 'tensor')
		c.check(f.info('w')!.as_map()['dtype']!.str() == 'F32', 'info')
		os.rm(path)!
	}
	{
		c.check(rlx.devices()![0] == 'cpu' && rlx.is_available('cpu'), 'cpu')
		g, _ := loss_graph()!
		r := rlx.Runner.new(g, policy: '{"only": ["cpu"]}')!
		r.set_param('w', [1.0, 0, 0, 1])!
		res := r.run(feed)!
		c.check(res.device == 'cpu' && res.outputs[0][0] == 2.5, 'runner')
		c.check(rlx.Group.connect(0, 1)!.all_reduce([2.0, 3], op: 'max')![1] == 3, 'all_reduce')
	}
	println('test_rlx: ${c.n} checks passed')
}
