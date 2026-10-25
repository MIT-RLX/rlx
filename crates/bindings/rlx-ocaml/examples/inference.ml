(* RLX — versatile ML compiler + runtime.
   Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
   SPDX-License-Identifier: MIT OR Apache-2.0 *)

(* Inference: build a graph, compile it for a device, set its weights, run it.
   A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].
     dune exec examples/inference.exe -- [device] *)
open Rlx

let weights n seed = Array.init n (fun i -> float ((i * 7919 + seed * 104729) mod 1000) /. 1000. -. 0.5)

let () =
  let device = if Array.length Sys.argv > 1 then Sys.argv.(1) else "cpu" in
  let g = Graph.create ~name:"classifier" () in
  let x = Graph.input g "x" [| 1; 4 |] in
  let hidden = Graph.relu g (Graph.matmul g x (Graph.param g "w1" [| 4; 8 |])) in
  Graph.set_outputs g [| Graph.softmax g (Graph.matmul g hidden (Graph.param g "w2" [| 8; 3 |])) |];
  let model = Session.compile (Session.create ~device ()) g (* consumes g *) in
  Printf.printf "compiled on: %s\n" (Compiled.device model);
  ignore (Compiled.set_params model [ ("w1", weights 32 1); ("w2", weights 24 2) ]);
  let probs = (Compiled.run model [ ("x", [| 0.5; -1.; 2.; 0.25 |]) ]).(0) in
  Printf.printf "probs: %s\n" (String.concat " " (Array.to_list (Array.map (Printf.sprintf "%.4f") probs)));
  Printf.printf "probs sum: %.4f\n" (Array.fold_left ( +. ) 0. probs);
  let shapes = Compiled.output_shapes model in
  Printf.printf "output shapes: [%s] dtypes: [%s]\n"
    (String.concat "," (Array.to_list (Array.map (fun s -> "[" ^ String.concat "," (Array.to_list (Array.map string_of_int s)) ^ "]") shapes)))
    (String.concat "," (Array.to_list (Compiled.output_dtypes model)))
