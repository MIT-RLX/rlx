// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Picking a backend: what this build has, what suits a graph, and the ways to
// choose — by name, by cost model, by policy, by benchmark, per run.
//
//	go run ./examples/backends
package main

import (
	"fmt"
	"log"
	"math"
	"strings"

	rlx "github.com/MIT-RLX/rlx/crates/bindings/rlx-go"
)

func must[T any](v T, err error) T {
	if err != nil {
		log.Fatal(err)
	}
	return v
}

func model() *rlx.Graph {
	g := must(rlx.NewGraph(&rlx.GraphOpts{Name: rlx.Ptr("mlp")}))
	x := must(g.Input("x", []int{4, 16}, nil))
	must(0, g.SetOutputs([]rlx.Node{must(g.Tanh(must(g.Matmul(x, must(g.Param("w", []int{16, 16}, nil))))))}))
	return g
}

func main() {
	x := map[string][]float32{"x": make([]float32, 64)}
	for i := range x["x"] {
		x["x"][i] = float32(i%7) / 7
	}
	w := make([]float32, 256)
	for i := range w {
		w[i] = float32(i%5)/10 - 0.2
	}
	available := must(rlx.Devices())
	fmt.Println("devices:", strings.Join(available, " "))
	fmt.Println("metal available:", rlx.IsAvailable("metal"))
	fmt.Println("fastest for graph:", must(rlx.FastestDeviceFor(model(), nil)))
	var reference []float32
	for _, device := range available {
		c := must(must(rlx.NewSession(&rlx.SessionOpts{Device: &device})).Compile(model(), nil))
		must(0, c.SetParam("w", w))
		y := must(c.Run(x))[0]
		if reference == nil {
			reference = y
		}
		d := 0.0
		for i := range y {
			d = math.Max(d, math.Abs(float64(y[i]-reference[i])))
		}
		fmt.Printf("session %s: max|Δ| vs first = %.2e\n", device, d)
	}
	flex := must(rlx.NewFlexibleSession(&rlx.FlexibleSessionOpts{Policy: map[string]any{"prefer": []string{"metal", "cpu"}}}))
	fmt.Println("flexible session picked:", must(must(flex.Compile(model(), nil)).Device()))
	runner := must(rlx.NewRunner(model(), nil))
	must(0, runner.SetParam("w", w))
	fmt.Println("runner warmed:", strings.Join(must(runner.WarmAll()), " "))
	runs := 3
	for _, row := range must(runner.Benchmark(x, &rlx.RunnerBenchmarkOpts{Runs: &runs})) {
		fmt.Printf("  bench %v: exec_ms=%v\n", row["device"], row["exec_ms"])
	}
	fmt.Println("runner ran on:", must(runner.Run(x, nil)).Device)
	fmt.Println("run_try chain:", must(runner.RunTry([]string{"metal", "cpu"}, x)).Device)
	router := must(rlx.NewRouter(model(), nil))
	must(0, router.SetParam("w", w))
	fmt.Println("router ran on:", must(router.RunChain(x, nil)).Device)
}
