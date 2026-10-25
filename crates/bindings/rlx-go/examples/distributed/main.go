// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Distributed: three ranks as goroutines on loopback; across machines, one rank
// per host with the same peers. Collectives, pipelined inference (a stage
// shipped per worker), data-parallel training.
//
//	go run ./examples/distributed
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

const world = 3
const perRank = 16

func must[T any](v T, err error) T {
	if err != nil {
		log.Fatal(err)
	}
	return v
}

var tmp = must(os.MkdirTemp("", "rlx-dist-go"))

func blob(name string, values []float32) string {
	var b []byte
	for _, v := range values {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
	}
	path := filepath.Join(tmp, name)
	must(0, os.WriteFile(path, b, 0o644))
	return "file://" + path
}

func matvec(x, w []float32) []float32 {
	out := make([]float32, 4)
	for j := 0; j < 4; j++ {
		s := float32(0)
		for i := 0; i < 4; i++ {
			s += x[i] * w[i*4+j]
		}
		out[j] = float32(math.Max(0, float64(s)))
	}
	return out
}

func toF32(v any) []float32 {
	out := []float32{}
	for _, x := range v.([]any) {
		out = append(out, float32(x.(float64)))
	}
	return out
}

func spec(rank int, uris map[string]string) map[string]any {
	start := rank * perRank
	return map[string]any{
		"params": []any{map[string]any{"name": "w", "uri": uris["w"]}},
		"data": []any{
			map[string]any{"input": "X", "uri": uris["X"], "elem": 3, "shard_start": start, "shard_len": perRank},
			map[string]any{"input": "y", "uri": uris["y"], "elem": 1, "shard_start": start, "shard_len": perRank},
		},
		"batch": 4, "lr_per_epoch": []float64{0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2},
		"device": "cpu",
	}
}

func main() {
	defer os.RemoveAll(tmp)
	stageW := map[int][]float32{1: make([]float32, 16), 2: make([]float32, 16)}
	for i := 0; i < 16; i++ {
		stageW[1][i] = float32(i*5%7)/7 - 0.3
		stageW[2][i] = float32(i*3%5)/5 - 0.2
	}
	wTrue := []float32{1.5, -2, 0.5}
	xAll := make([]float32, perRank*world*3)
	for i := range xAll {
		xAll[i] = float32(i*37%23)/11.5 - 1
	}
	yAll := make([]float32, perRank*world)
	for i := range yAll {
		for k := 0; k < 3; k++ {
			yAll[i] += xAll[i*3+k] * wTrue[k]
		}
	}

	peers := must(rlx.LoopbackPeers(world))
	opts := &rlx.GroupConnectOpts{Options: map[string]any{"peers": peers, "device": "cpu"}}
	results := make([]chan []float32, world)
	for rank := 1; rank < world; rank++ {
		results[rank] = make(chan []float32, 1)
		go func(rank int) {
			g := must(rlx.GroupConnect(uint32(rank), world, opts))
			must(g.AllReduce([]float32{float32(rank + 1)}, &rlx.GroupAllReduceOpts{Op: rlx.Ptr("sum")}))
			must(g.AllGather([]float32{float32(rank + 1)}))
			must(g.Broadcast(0, []float32{0})) // the root's length; it overwrites
			must(g.ServeWorker(&rlx.GroupServeWorkerOpts{Options: map[string]any{"steps": 2}}))
			rep := must(g.ServeTrainer(nil))
			results[rank] <- toF32(rep["params"].(map[string]any)["w"])
		}(rank)
	}

	g := must(rlx.GroupConnect(0, world, opts))
	fmt.Printf("mesh: rank %d of %d, leader: %v\n", must(g.Rank()), must(g.World()), must(g.IsLeader()))
	fmt.Println("all_reduce sum:", must(g.AllReduce([]float32{1}, &rlx.GroupAllReduceOpts{Op: rlx.Ptr("sum")})))
	fmt.Println("all_gather:", must(g.AllGather([]float32{1})))
	fmt.Println("broadcast:", must(g.Broadcast(0, []float32{42})))

	caps := must(g.CollectCaps())
	names := []string{}
	for _, c := range caps {
		names = append(names, fmt.Sprintf("rank %v on %v", c["rank"], c["platform"]))
	}
	fmt.Println("workers:", strings.Join(names, ", "))
	stage := must(rlx.NewGraph(&rlx.GraphOpts{Name: rlx.Ptr("stage")}))
	sx := must(stage.Input("x", []int{1, 4}, nil))
	must(0, stage.SetOutputs([]rlx.Node{must(stage.Relu(must(stage.Matmul(sx, must(stage.Param("w", []int{4, 4}, nil))))))}))
	for r := 1; r < world; r++ {
		must(0, g.ShipStage(uint32(r), stage, map[string]any{"device": "cpu", "weights": map[string]any{"w": blob(fmt.Sprintf("w%d.bin", r), stageW[r])}}))
	}
	for _, xv := range [][]float32{{1, -1, 0.5, 2}, {0.25, 0.5, -0.75, 1}} {
		must(0, g.SendActivation(1, xv))
		yv := must(g.RecvActivation(world - 1))
		want := matvec(matvec(xv, stageW[1]), stageW[2])
		d := 0.0
		for i := range yv {
			d = math.Max(d, math.Abs(float64(yv[i]-want[i])))
		}
		fmt.Printf("pipeline max|Δ| vs local = %.1e\n", d)
	}

	must(g.CollectCaps())
	lr := must(rlx.NewGraph(&rlx.GraphOpts{Name: rlx.Ptr("lr")}))
	X := must(lr.Input("X", []int{4, 3}, nil))
	Y := must(lr.Input("y", []int{4, 1}, nil))
	wv := must(lr.Param("w", []int{3, 1}, nil))
	d := must(lr.Sub(must(lr.Matmul(X, wv)), Y))
	must(0, lr.SetOutputs([]rlx.Node{must(lr.Mean(must(lr.Mul(d, d)), []int{0, 1}, nil))}))
	bwd := must(rlx.Grad(lr, []rlx.Node{wv}))
	uris := map[string]string{"X": blob("X.bin", xAll), "y": blob("y.bin", yAll), "w": blob("w0.bin", []float32{0, 0, 0})}
	for r := 1; r < world; r++ {
		must(0, g.ShipTrain(uint32(r), bwd, spec(r, uris)))
	}
	mine := must(g.RunTrain(bwd, spec(0, uris)))
	m := mine["metrics"].(map[string]any)
	fmt.Printf("training loss: %.4f -> %.2e\n", m["first_loss"], m["last_loss"])
	w0 := toF32(mine["params"].(map[string]any)["w"])
	fmt.Printf("rank 0 w: %.3f %.3f %.3f (true: 1.5 -2 0.5)\n", w0[0], w0[1], w0[2])
	for r := 1; r < world; r++ {
		theirs := <-results[r]
		agree := len(theirs) == len(w0)
		for i := range w0 {
			agree = agree && math.Abs(float64(theirs[i]-w0[i])) < 1e-5
		}
		fmt.Printf("rank %d w agrees: %v\n", r, agree)
	}
}
