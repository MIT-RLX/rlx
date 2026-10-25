// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
// into a compiled graph, and quantizing in memory. RLX_TOKENIZER=<tokenizer.json>
// adds the tokenizer.
//
//	go run ./examples/loading
package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"

	rlx "github.com/MIT-RLX/rlx/crates/bindings/rlx-go"
)

func must[T any](v T, err error) T {
	if err != nil {
		log.Fatal(err)
	}
	return v
}

func equal(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func main() {
	w := make([]float32, 64)
	for i := range w {
		w[i] = float32(i%9)/4 - 1
	}
	gguf := filepath.Join(os.TempDir(), "rlx-example-go.gguf")
	writer := must(rlx.NewGgufWriter())
	must(0, writer.AddF32("proj.weight", []int{8, 8}, "F32", w))
	must(0, writer.AddF32("proj.weight_q8", []int{8, 8}, "Q8_0", w))
	must(writer.Write(gguf, &rlx.GgufWriterWriteOpts{Architecture: rlx.Ptr("demo"), Metadata: map[string]any{"demo.layers": 1}}))
	f := must(rlx.GgufOpen(gguf))
	fmt.Println("gguf architecture:", *must(f.Architecture()))
	fmt.Println("gguf tensors:", strings.Join(must(f.TensorNames()), " "))
	exact := must(f.Tensor("proj.weight"))
	q8 := must(f.Tensor("proj.weight_q8"))
	maxd := 0.0
	for i := range exact {
		maxd = math.Max(maxd, math.Abs(float64(exact[i]-q8[i])))
	}
	fmt.Printf("gguf f32 exact: %v q8 max|Δ|: %.4f\n", equal(exact, w), maxd)

	g := must(rlx.NewGraph(&rlx.GraphOpts{Name: rlx.Ptr("proj")}))
	x := must(g.Input("x", []int{1, 8}, nil))
	must(0, g.SetOutputs([]rlx.Node{must(g.Matmul(x, must(g.Param("proj.weight", []int{8, 8}, nil))))}))
	c := must(must(rlx.NewSession(nil)).Compile(g, nil))
	must(0, c.SetParam("proj.weight", exact))
	y := must(c.Run(map[string][]float32{"x": {1, 1, 1, 1, 1, 1, 1, 1}}))[0]
	fmt.Printf("gguf -> graph: %.2f %.2f %.2f\n", y[0], y[1], y[2])

	st := filepath.Join(os.TempDir(), "rlx-example-go.safetensors")
	header := fmt.Sprintf(`{"proj.weight":{"dtype":"F32","shape":[8,8],"data_offsets":[0,%d]}}`, 4*len(w))
	for len(header)%8 != 0 {
		header += " "
	}
	buf := binary.LittleEndian.AppendUint64(nil, uint64(len(header)))
	buf = append(buf, header...)
	for _, v := range w {
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
	}
	must(0, os.WriteFile(st, buf, 0o644))
	loaded := must(rlx.LoadWeights("safetensors", st))
	fmt.Println("safetensors: [proj.weight] exact:", equal(loaded["proj.weight"], w))

	packed := must(rlx.Quantize(w, "Q8_0"))
	n := len(w)
	back := must(rlx.Dequantize(packed, "Q8_0", &rlx.DequantizeOpts{Count: &n}))
	fmt.Printf("quantize q8_0: %d -> %d bytes, round trip %d values\n", 4*len(w), len(packed), len(back))
	os.Remove(gguf)
	os.Remove(st)

	if path := os.Getenv("RLX_TOKENIZER"); path != "" {
		tok := must(rlx.TokenizerLoad(path))
		ids := must(tok.Encode("Hello from Go", nil))
		fmt.Printf("tokenizer: %d ids -> %q\n", len(ids), must(tok.Decode(ids, nil)))
	} else {
		fmt.Println("tokenizer: skipped (set RLX_TOKENIZER=<tokenizer.json>)")
	}
	fmt.Println("argmax:", must(rlx.Argmax([]float32{0.1, 2, 0.3})))
}
