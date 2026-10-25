// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Picking a backend: what this build has, what suits a graph, and the ways to
// choose — by name, by cost model, by policy, by benchmark, per run.
//   v run examples/backends.v
import math
import rlx

fn model() !&rlx.Graph {
	g := rlx.Graph.new(name: 'mlp')!
	x := g.input('x', [4, 16])!
	g.set_outputs([g.tanh(g.matmul(x, g.param('w', [16, 16])!)!)!])!
	return g
}

fn main() {
	w := []f64{len: 256, init: f64(index % 5) / 10 - 0.2}
	x := {
		'x': []f64{len: 64, init: f64(index % 7) / 7}
	}

	available := rlx.devices()!
	println('devices: ${available.join(' ')}')
	println('metal available: ${rlx.is_available('metal')}')
	println('fastest for graph: ${rlx.fastest_device_for(model()!)!}')
	mut reference := []f64{}
	for device in available {
		c := rlx.Session.new(device: device)!.compile(model()!)!
		c.set_param('w', w)!
		y := c.run(x)![0]
		if reference.len == 0 {
			reference = y.clone()
		}
		mut d := 0.0
		for i in 0 .. y.len {
			d = math.max(d, math.abs(y[i] - reference[i]))
		}
		println('session ${device}: max|Δ| vs first = ${d:.2e}')
	}
	flex := rlx.FlexibleSession.new(policy: '{"prefer": ["metal", "cpu"]}')!
	println('flexible session picked: ${flex.compile(model()!)!.device()!}')
	runner := rlx.Runner.new(model()!)!
	runner.set_param('w', w)!
	println('runner warmed: ${runner.warm_all()!.join(' ')}')
	for row in runner.benchmark(x, runs: 3)!.as_array() {
		m := row.as_map()
		println('  bench ${m['device']!.str()}: exec_ms=${m['exec_ms']!.f64()}')
	}
	println('runner ran on: ${runner.run(x)!.device}')
	println('run_try chain: ${runner.run_try(['metal', 'cpu'], x)!.device}')
	router := rlx.Router.new(model()!)!
	router.set_param('w', w)!
	println('router ran on: ${router.run_chain(x)!.device}')
}
