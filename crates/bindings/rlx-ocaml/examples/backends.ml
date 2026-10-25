(* RLX — versatile ML compiler + runtime.
   Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
   SPDX-License-Identifier: MIT OR Apache-2.0 *)

(* Picking a backend: what this build has, what suits a graph, and the ways
   to choose — by name, by cost model, by policy, by benchmark, per run.
     dune exec examples/backends.exe *)
open Rlx

let model () =
  let g = Graph.create ~name:"mlp" () in
  let x = Graph.input g "x" [| 4; 16 |] in
  Graph.set_outputs g [| Graph.tanh g (Graph.matmul g x (Graph.param g "w" [| 16; 16 |])) |];
  g

let () =
  let w = Array.init 256 (fun i -> float (i mod 5) /. 10. -. 0.2) in
  let x = [ ("x", Array.init 64 (fun i -> float (i mod 7) /. 7.)) ] in
  let available = devices () in
  Printf.printf "devices: %s\n" (String.concat " " (Array.to_list available));
  Printf.printf "metal available: %b\n" (is_available "metal");
  Printf.printf "fastest for graph: %s\n" (fastest_device_for (model ()));
  let reference = ref None in
  Array.iter
    (fun device ->
      let c = Session.compile (Session.create ~device ()) (model ()) in
      Compiled.set_param c "w" w;
      let y = (Compiled.run c x).(0) in
      let r = match !reference with Some r -> r | None -> reference := Some y; y in
      let d = ref 0. in
      Array.iteri (fun i v -> d := Float.max !d (Float.abs (v -. r.(i)))) y;
      Printf.printf "session %s: max|Δ| vs first = %.2e\n" device !d)
    available;
  let flex = FlexibleSession.create ~policy:Json.(Obj [ ("prefer", Arr [ Str "metal"; Str "cpu" ]) ]) () in
  Printf.printf "flexible session picked: %s\n" (Compiled.device (FlexibleSession.compile flex (model ())));
  let runner = Runner.create (model ()) in
  Runner.set_param runner "w" w;
  Printf.printf "runner warmed: %s\n" (String.concat " " (Array.to_list (Runner.warm_all runner)));
  List.iter
    (fun row -> Printf.printf "  bench %s: exec_ms=%g\n" (Json.to_str (Json.member "device" row)) (Json.to_float (Json.member "exec_ms" row)))
    (Json.to_list (Runner.benchmark ~runs:3 runner x));
  Printf.printf "runner ran on: %s\n" (fst (Runner.run runner x));
  Printf.printf "run_try chain: %s\n" (fst (Runner.run_try runner [| "metal"; "cpu" |] x));
  let router = Router.create (model ()) in
  Router.set_param router "w" w;
  Printf.printf "router ran on: %s\n" (fst (Router.run_chain router x))
