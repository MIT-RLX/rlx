// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Models: any provider's model, opened by path or id and run by name. A graph
// saved as an .rlxp package runs anywhere; a plugin's models (a downstream
// collection's LLMs, or the reference plugin's demo:count) load through
// RLX_MODEL_PLUGINS.
//   RLX_MODEL_PLUGINS=…/librlx_model_plugin.so v run examples/models.v
import os
import rlx

fn main() {
	g := rlx.Graph.new(name: 'affine')!
	x := g.input('x', [1, 2])!
	g.set_outputs([g.matmul(x, g.param('w', [2, 2])!)!])!
	path := os.join_path(os.temp_dir(), 'rlx-example-v.rlxp')
	rlx.save_model(g, path, {
		'w': [2.0, 0, 0, 3]
	}, model: '{"task": "graph", "family": "affine", "outputs": ["y"]}')!

	m := rlx.Model.open(path, options: '{"device": "cpu"}')!
	info := m.info()!.as_map()
	println('model: ${info['family']!.str()} (${info['task']!.str()}) on ${info['device']!.str()}')
	y := m.run({
		'x': [1.0, 1]
	})![0]
	println('model run: ${y[0]:.3f} ${y[1]:.3f}')
	os.rm(path)!

	lm := rlx.Model.open('demo:count')!
	lm_info := lm.info()!.as_map()
	println('plugin model: ${lm_info['family']!.str()} (${lm_info['task']!.str()})')
	println('model text: ${lm.generate_text('c', options: '{"max_tokens": 4}')!}')
	println('model ids: ${lm.generate([2], max_tokens: 3)!.map(it.str()).join(' ')}')

	// By task: a chat, speech out and back in, embeddings.
	println('chat: ${lm.chat('[{"role": "user", "content": "c"}]', options: '{"max_tokens": 4}')!}')
	speech := rlx.Model.open('demo:voice')!.synthesize('hello')!
	println('voice: ${speech.audio.len} samples at ${speech.sample_rate} Hz')
	ear := rlx.Model.open('demo:speech')!
	println('heard: ${ear.transcribe(speech.audio, sample_rate: speech.sample_rate)!}')
	mut values := []string{}
	for row in rlx.Model.open('demo:embed')!.embed(['hello', 'sky'])!.as_array() {
		for v in row.as_array() {
			values << int(v.f64()).str()
		}
	}
	println('embed: ${values.join(' ')}')
}
