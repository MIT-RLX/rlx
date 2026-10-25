// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Inference: build a graph, compile it for a device, set its weights, run it.
//   v run examples/inference.v [device]
// A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].
import os
import rlx

fn weights(n int, seed int) []f64 {
	return []f64{len: n, init: f64((index * 7919 + seed * 104729) % 1000) / 1000 - 0.5}
}

fn main() {
	device := if os.args.len > 1 { os.args[1] } else { 'cpu' }
	g := rlx.Graph.new(name: 'classifier')!
	x := g.input('x', [1, 4])!
	hidden := g.relu(g.matmul(x, g.param('w1', [4, 8])!)!)!
	g.set_outputs([g.softmax(g.matmul(hidden, g.param('w2', [8, 3])!)!)!])!

	model := rlx.Session.new(device: device)!.compile(g)! // consumes g
	println('compiled on: ${model.device()!}')
	model.set_params({
		'w1': weights(32, 1)
		'w2': weights(24, 2)
	})!
	probs := model.run({
		'x': [0.5, -1, 2, 0.25]
	})![0]
	println('probs: ${probs.map('${it:.4f}').join(' ')}')
	mut sum := 0.0
	for p in probs {
		sum += p
	}
	println('probs sum: ${sum:.4f}')
	println('output shapes: ${model.output_shapes()!} dtypes: ${model.output_dtypes()!}')
}
