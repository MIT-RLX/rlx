// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Inference: build a graph, compile it for a device, set its weights, run it.
//
//	go run ./examples/inference [device]
//
// A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].
package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	rlx "github.com/MIT-RLX/rlx/crates/bindings/rlx-go"
)

func must[T any](v T, err error) T {
	if err != nil {
		log.Fatal(err)
	}
	return v
}

func weights(n, seed int) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = float32((i*7919+seed*104729)%1000)/1000 - 0.5
	}
	return w
}

func main() {
	device := "cpu"
	if len(os.Args) > 1 {
		device = os.Args[1]
	}
	g := must(rlx.NewGraph(&rlx.GraphOpts{Name: rlx.Ptr("classifier")}))
	x := must(g.Input("x", []int{1, 4}, nil))
	hidden := must(g.Relu(must(g.Matmul(x, must(g.Param("w1", []int{4, 8}, nil))))))
	must(0, g.SetOutputs([]rlx.Node{must(g.Softmax(must(g.Matmul(hidden, must(g.Param("w2", []int{8, 3}, nil)))), nil))}))

	model := must(must(rlx.NewSession(&rlx.SessionOpts{Device: &device})).Compile(g, nil)) // consumes g
	defer model.Close()
	fmt.Println("compiled on:", must(model.Device()))
	must(model.SetParams(map[string][]float32{"w1": weights(32, 1), "w2": weights(24, 2)}))
	probs := must(model.Run(map[string][]float32{"x": {0.5, -1, 2, 0.25}}))[0]
	parts := []string{}
	sum := float32(0)
	for _, p := range probs {
		parts = append(parts, fmt.Sprintf("%.4f", p))
		sum += p
	}
	fmt.Println("probs:", strings.Join(parts, " "))
	fmt.Printf("probs sum: %.4f\n", sum)
	fmt.Println("output shapes:", must(model.OutputShapes()), "dtypes:", must(model.OutputDtypes()))
}
