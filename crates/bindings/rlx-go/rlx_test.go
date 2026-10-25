// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

package rlx

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// must panics on an error — Go forwards a call's results only as a whole
// argument list, so it cannot also take the *testing.T.
func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func ok(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// lossGraph is mean((x·w − y)²), x/y [1, 2], w [2, 2].
func lossGraph(t *testing.T) (*Graph, Node) {
	g := must(NewGraph(&GraphOpts{Name: ptrTo("mse")}))
	x := must(g.Input("x", []int{1, 2}, nil))
	y := must(g.Input("y", []int{1, 2}, nil))
	w := must(g.Param("w", []int{2, 2}, nil))
	d := must(g.Sub(must(g.Matmul(x, w)), y))
	ok(t, g.SetOutputs([]Node{must(g.Mean(must(g.Mul(d, d)), []int{0, 1}, nil))}))
	return g, w
}

var batch = map[string][]float32{"x": {1, 1}, "y": {1, -1}}

func TestInference(t *testing.T) {
	g, _ := lossGraph(t)
	m := must(must(NewSession(nil)).Compile(g, nil))
	defer m.Close()
	if !g.IsConsumed() {
		t.Fatal("compile should consume the graph")
	}
	ok(t, m.SetParam("w", []float32{1, 0, 0, 1}))
	out := must(m.Run(map[string][]float32{"x": {1, 2}, "y": {0, 0}}))
	if out[0][0] != 2.5 {
		t.Fatalf("loss %v", out[0][0])
	}
	if must(m.Device()) != "cpu" {
		t.Fatal("device")
	}
}

func TestErrors(t *testing.T) {
	g := must(NewGraph(nil))
	_, err := g.Relu(42)
	var e *Error
	if !errors.As(err, &e) || e.Kind != ErrRange || !strings.Contains(e.Message, "does not exist") {
		t.Fatalf("want a range error, got %v", err)
	}
	if _, err := NewSession(&SessionOpts{Device: ptrTo("no-such-device")}); err == nil {
		t.Fatal("unknown device should fail")
	}
}

func TestTraining(t *testing.T) {
	g, w := lossGraph(t)
	bwd := must(must(NewSession(nil)).Compile(must(Grad(g, []Node{w})), nil))
	ok(t, bwd.SetParam("w", []float32{1, 0, 0, 1}))
	outs := must(bwd.Run(map[string][]float32{"x": {1, 2}, "y": {0, 0}, "d_output": {1}}))
	if outs[1][3] != 4 {
		t.Fatalf("dw %v", outs[1])
	}
	tr := must(NewTrainer(must(g.Clone()), []string{"w"}, map[string][]float32{"w": {0, 0, 0, 0}},
		&TrainerOpts{Options: map[string]any{"optimizer": map[string]any{"kind": "adamw", "lr": 0.1}}}))
	summary := must(tr.Run(300, func(uint64) map[string][]float32 { return batch },
		&RunOpts{OnStep: func(_ uint64, l float32) bool { return l > 1e-6 }}))
	if summary["last_loss"].(float32) > 1e-4 {
		t.Fatalf("did not converge: %v", summary)
	}
	path := filepath.Join(t.TempDir(), "ck.gguf")
	if must(tr.Save(path)) < 1 {
		t.Fatal("save")
	}
	opt := must(NewOptimizer(&OptimizerOpts{Kind: ptrTo("sgd"), Options: map[string]any{"lr": 0.5}}))
	p := map[string][]float32{"p": {1, -1}}
	ok(t, opt.Step(p, map[string][]float32{"p": {2, -2}}))
	if p["p"][0] != 0 || p["p"][1] != 0 {
		t.Fatalf("step %v", p)
	}
}

func TestLoading(t *testing.T) {
	path := filepath.Join(t.TempDir(), "w.gguf")
	w := []float32{0.5, -1, 2, 0.25}
	wr := must(NewGgufWriter())
	ok(t, wr.AddF32("w", []int{4}, "F32", w))
	if must(wr.Write(path, &GgufWriterWriteOpts{Architecture: ptrTo("demo")})) != 1 {
		t.Fatal("write")
	}
	f := must(GgufOpen(path))
	if *must(f.Architecture()) != "demo" {
		t.Fatal("architecture")
	}
	back := must(f.Tensor("w"))
	for i := range w {
		if back[i] != w[i] {
			t.Fatalf("tensor %v", back)
		}
	}
	os.Remove(path)
}

func TestBackendsAndMesh(t *testing.T) {
	devs := must(Devices())
	if devs[0] != "cpu" || !IsAvailable("cpu") {
		t.Fatal("cpu")
	}
	g, _ := lossGraph(t)
	r := must(NewRunner(g, &RunnerOpts{Policy: map[string]any{"only": []string{"cpu"}}}))
	ok(t, r.SetParam("w", []float32{1, 0, 0, 1}))
	res := must(r.Run(map[string][]float32{"x": {1, 2}, "y": {0, 0}}, nil))
	if res.Device != "cpu" || res.Outputs[0][0] != 2.5 {
		t.Fatalf("runner %v", res)
	}

	peers := must(LoopbackPeers(2))
	done := make(chan map[string]any, 1)
	go func() {
		g, err := GroupConnect(1, 2, &GroupConnectOpts{Options: map[string]any{"peers": peers}})
		if err != nil {
			done <- nil
			return
		}
		g.AllReduce([]float32{3}, &GroupAllReduceOpts{Op: ptrTo("max")})
		rep, _ := g.ServeWorker(&GroupServeWorkerOpts{Options: map[string]any{"steps": 1}})
		done <- rep
	}()
	mesh := must(GroupConnect(0, 2, &GroupConnectOpts{Options: map[string]any{"peers": peers}}))
	if m := must(mesh.AllReduce([]float32{5}, &GroupAllReduceOpts{Op: ptrTo("max")})); m[0] != 5 {
		t.Fatal("all_reduce")
	}
	if len(must(mesh.CollectCaps())) != 1 {
		t.Fatal("caps")
	}
	s := must(NewGraph(nil))
	x := must(s.Input("x", []int{2}, nil))
	ok(t, s.SetOutputs([]Node{must(s.Add(x, x))}))
	ok(t, mesh.ShipStage(1, s, map[string]any{"device": "cpu"}))
	ok(t, mesh.SendActivation(1, []float32{1, 2}))
	if y := must(mesh.RecvActivation(1)); y[0] != 2 || y[1] != 4 {
		t.Fatalf("activation %v", y)
	}
	if rep := <-done; rep == nil || rep["activations"].(float64) != 1 {
		t.Fatalf("worker %v", rep)
	}
}
