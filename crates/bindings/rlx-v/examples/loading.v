// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
// into a compiled graph, and quantizing in memory.
//   v run examples/loading.v
import encoding.binary
import math
import os
import rlx

fn main() {
	tmp := os.temp_dir()
	w := []f64{len: 64, init: f64(index % 9) / 4 - 1}

	gguf := os.join_path(tmp, 'rlx-example-v.gguf')
	writer := rlx.GgufWriter.new()!
	writer.add_f32('proj.weight', [8, 8], 'F32', w)!
	writer.add_f32('proj.weight_q8', [8, 8], 'Q8_0', w)!
	writer.write(gguf, architecture: 'demo', metadata: '{"demo.layers": 1}')!
	f := rlx.Gguf.open(gguf)!
	println('gguf architecture: ${f.architecture()!}')
	println('gguf tensors: ${f.tensor_names()!.join(' ')}')
	exact := f.tensor('proj.weight')!
	q8 := f.tensor('proj.weight_q8')!
	mut maxd := 0.0
	for i in 0 .. 64 {
		maxd = math.max(maxd, math.abs(exact[i] - q8[i]))
	}
	println('gguf f32 exact: ${exact == w} q8 max|Δ|: ${maxd:.4f}')

	g := rlx.Graph.new(name: 'proj')!
	x := g.input('x', [1, 8])!
	g.set_outputs([g.matmul(x, g.param('proj.weight', [8, 8])!)!])!
	c := rlx.Session.new()!.compile(g)!
	c.set_param('proj.weight', exact)!
	y := c.run({
		'x': []f64{len: 8, init: 1}
	})![0]
	println('gguf -> graph: ${y[..3].map('${it:.2f}').join(' ')}')

	st := os.join_path(tmp, 'rlx-example-v.safetensors')
	mut header := '{"proj.weight":{"dtype":"F32","shape":[8,8],"data_offsets":[0,256]}}'
	header += ' '.repeat((8 - header.len % 8) % 8)
	mut bytes := []u8{len: 8}
	binary.little_endian_put_u64(mut bytes, u64(header.len))
	bytes << header.bytes()
	for v in w {
		mut b := []u8{len: 4}
		binary.little_endian_put_u32(mut b, math.f32_bits(f32(v)))
		bytes << b
	}
	os.write_file_array(st, bytes)!
	loaded := rlx.load_weights('safetensors', st)!
	println('safetensors: [proj.weight] exact: ${loaded['proj.weight'] == w}')

	packed := rlx.quantize(w, 'Q8_0')!
	back := rlx.dequantize(packed, 'Q8_0', count: w.len)!
	println('quantize q8_0: ${w.len * 4} -> ${packed.len} bytes, round trip ${back.len} values')
	os.rm(gguf)!
	os.rm(st)!
	println('argmax: ${rlx.argmax([0.1, 2, 0.3])!}')
}
