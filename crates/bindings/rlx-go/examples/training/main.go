// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
// the same fit by hand with Grad and an Optimizer.
//
//	go run ./examples/training [device]
package main

import (
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

var wTrue = []float32{2, -1, 0.5}

const batch = 8

func data(step uint64) map[string][]float32 {
	x := make([]float32, 0, batch*3)
	y := make([]float32, 0, batch)
	for i := 0; i < batch; i++ {
		s := float32(0)
		for k := 0; k < 3; k++ {
			v := float32((int(step)*31+i*7+k*3)%17)/8.5 - 1
			x = append(x, v)
			s += v * wTrue[k]
		}
		y = append(y, s)
	}
	return map[string][]float32{"x": x, "y": y}
}

func fmtv(v []float32) string {
	p := []string{}
	for _, a := range v {
		p = append(p, fmt.Sprintf("%.3f", a))
	}
	return strings.Join(p, " ")
}

func main() {
	device := "cpu"
	if len(os.Args) > 1 {
		device = os.Args[1]
	}
	g := must(rlx.NewGraph(&rlx.GraphOpts{Name: rlx.Ptr("regression")}))
	x := must(g.Input("x", []int{batch, 3}, nil))
	y := must(g.Input("y", []int{batch, 1}, nil))
	w := must(g.Param("w", []int{3, 1}, nil))
	d := must(g.Sub(must(g.Matmul(x, w)), y))
	must(0, g.SetOutputs([]rlx.Node{must(g.Mean(must(g.Mul(d, d)), []int{0, 1}, nil))}))

	opts := map[string]any{"device": device, "optimizer": map[string]any{"kind": "adamw", "lr": 0.1}}
	trainer := must(rlx.NewTrainer(must(g.Clone()), []string{"w"}, map[string][]float32{"w": {0, 0, 0}}, &rlx.TrainerOpts{Options: opts}))
	first := must(trainer.Evaluate(data(0)))
	must(trainer.Run(400, data, &rlx.RunOpts{
		LR:     func(s uint64) float32 { return float32(0.1 * 0.5 * (1 + math.Cos(math.Pi*float64(s)/400))) },
		OnStep: func(_ uint64, loss float32) bool { return loss > 1e-6 },
	}))
	fmt.Printf("trainer loss: %.4f -> %.6f in %d steps\n", first, must(trainer.Evaluate(data(0))), must(trainer.Steps()))
	fmt.Println("trainer w:", fmtv(must(trainer.Params())["w"]))

	path := filepath.Join(os.TempDir(), "rlx-example-train-go.gguf")
	written := must(trainer.Save(path))
	restored := must(rlx.NewTrainer(must(g.Clone()), []string{"w"}, map[string][]float32{"w": {0, 0, 0}},
		&rlx.TrainerOpts{Options: map[string]any{"device": device}}))
	fmt.Printf("checkpoint: %d tensors, restored step %v\n", written, must(restored.Load(path))["step"])
	os.Remove(path)
	must(restored.Freeze(&rlx.TrainerFreezeOpts{Names: []string{"w"}}))
	before := fmtv(must(restored.Params())["w"])
	must(restored.Step(data(1)))
	fmt.Println("frozen:", must(restored.Frozen()), "unchanged:", fmtv(must(restored.Params())["w"]) == before)
	must(restored.Unfreeze(nil))

	backward := must(must(rlx.NewSession(&rlx.SessionOpts{Device: &device})).Compile(must(rlx.Grad(g, []rlx.Node{w})), nil))
	opt := must(rlx.NewOptimizer(&rlx.OptimizerOpts{Kind: rlx.Ptr("sgd"), Options: map[string]any{"lr": 0.2}}))
	params := map[string][]float32{"w": {0, 0, 0}}
	var loss float32
	for step := uint64(0); step < 300; step++ {
		must(0, backward.SetParam("w", params["w"]))
		feed := data(step)
		feed["d_output"] = []float32{1}
		outs := must(backward.Run(feed))
		loss = outs[0][0]
		must(0, opt.Step(params, map[string][]float32{"w": outs[1]}))
	}
	fmt.Printf("manual loss: %.6f w: %s\n", loss, fmtv(params["w"]))
}
