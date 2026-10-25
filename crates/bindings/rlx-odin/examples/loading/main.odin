// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
// into a compiled graph, and quantizing in memory.
//   odin run examples/loading -collection:rlx=.
#+feature dynamic-literals
package main

import "core:bytes"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:slice"
import "core:strings"
import rlx "rlx:rlx"

main :: proc() {
	tmp := tmp_dir()
	w := make([]f32, 64)
	for i in 0 ..< 64 do w[i] = f32(i % 9) / 4 - 1

	gguf := strings.concatenate({tmp, "/rlx-example-odin.gguf"})
	writer := rlx.must(rlx.gguf_writer_new())
	rlx.gguf_writer_add_f32(writer, "proj.weight", {8, 8}, "F32", w)
	rlx.gguf_writer_add_f32(writer, "proj.weight_q8", {8, 8}, "Q8_0", w)
	rlx.must(rlx.gguf_writer_write(writer, gguf, architecture = "demo", metadata = `{"demo.layers": 1}`))
	f := rlx.must(rlx.gguf_open(gguf))
	fmt.printfln("gguf architecture: %s", rlx.must(rlx.gguf_architecture(f)).? or_else "")
	fmt.printfln("gguf tensors: %s", strings.join(rlx.must(rlx.gguf_tensor_names(f)), " "))
	exact := rlx.must(rlx.gguf_tensor(f, "proj.weight"))
	q8 := rlx.must(rlx.gguf_tensor(f, "proj.weight_q8"))
	maxd: f32
	for i in 0 ..< 64 do maxd = max(maxd, abs(exact[i] - q8[i]))
	fmt.printfln("gguf f32 exact: %v q8 max|Δ|: %.4f", slice.equal(exact, w), maxd)

	g := rlx.must(rlx.graph_new(name = "proj"))
	x := rlx.must(rlx.input(g, "x", {1, 8}))
	rlx.set_outputs(g, {rlx.must(rlx.matmul(g, x, rlx.must(rlx.param(g, "proj.weight", {8, 8}))))})
	c := rlx.must(rlx.session_compile(rlx.must(rlx.session_new()), g))
	rlx.compiled_set_param(c, "proj.weight", exact)
	y := rlx.must(rlx.compiled_run(c, map[string][]f32{"x" = {1, 1, 1, 1, 1, 1, 1, 1}}))[0]
	fmt.printfln("gguf -> graph: %.2f %.2f %.2f", y[0], y[1], y[2])

	st := strings.concatenate({tmp, "/rlx-example-odin.safetensors"})
	header := `{"proj.weight":{"dtype":"F32","shape":[8,8],"data_offsets":[0,256]}}`
	pad := (8 - len(header) % 8) % 8
	buf: bytes.Buffer
	n := u64le(len(header) + pad)
	bytes.buffer_write(&buf, mem.any_to_bytes(n))
	bytes.buffer_write_string(&buf, header)
	for _ in 0 ..< pad do bytes.buffer_write_byte(&buf, ' ')
	bytes.buffer_write(&buf, slice.to_bytes(w))
	_ = os.write_entire_file(st, bytes.buffer_to_bytes(&buf))
	loaded := rlx.must(rlx.load_weights("safetensors", st))
	fmt.printfln("safetensors: [proj.weight] exact: %v", slice.equal(loaded["proj.weight"], w))

	packed := rlx.must(rlx.quantize(w, "Q8_0"))
	back := rlx.must(rlx.dequantize(packed, "Q8_0", count = len(w)))
	fmt.printfln("quantize q8_0: %d -> %d bytes, round trip %d values", len(w) * 4, len(packed), len(back))
	os.remove(gguf)
	os.remove(st)
	fmt.printfln("argmax: %d", rlx.must(rlx.argmax({0.1, 2, 0.3})))
}

tmp_dir :: proc() -> string {
	t := os.get_env("TMPDIR", context.allocator)
	return t == "" ? "/tmp" : t
}
