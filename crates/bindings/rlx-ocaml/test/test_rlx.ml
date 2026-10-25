(* RLX — versatile ML compiler + runtime.
   Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
   SPDX-License-Identifier: MIT OR Apache-2.0 *)

(* The OCaml binding against the real library; exits non-zero on the first
   failure. *)
open Rlx

let checks = ref 0

let check ok what =
  incr checks;
  if not ok then (prerr_endline ("failed: " ^ what); exit 1)

let loss_graph () =
  let g = Graph.create ~name:"mse" () in
  let x = Graph.input g "x" [| 1; 2 |] and y = Graph.input g "y" [| 1; 2 |] in
  let w = Graph.param g "w" [| 2; 2 |] in
  let d = Graph.sub g (Graph.matmul g x w) y in
  Graph.set_outputs g [| Graph.mean g (Graph.mul g d d) [| 0; 1 |] |];
  (g, w)

let () =
  let feed = [ ("x", [| 1.; 2. |]); ("y", [| 0.; 0. |]) ] in
  let g, _ = loss_graph () in
  let m = Session.compile (Session.create ()) g in
  check (Graph.is_consumed g) "compile consumes";
  Compiled.set_param m "w" [| 1.; 0.; 0.; 1. |];
  check ((Compiled.run m feed).(0).(0) = 2.5) "loss 2.5";
  check (Compiled.device m = "cpu") "device";
  (match Graph.relu (Graph.create ()) 42 with
  | _ -> check false "range error raised"
  | exception Error (Range_error, _) -> check true "range error");
  let g, w = loss_graph () in
  let bwd = Session.compile (Session.create ()) (grad g [| w |]) in
  Compiled.set_param bwd "w" [| 1.; 0.; 0.; 1. |];
  check ((Compiled.run bwd (feed @ [ ("d_output", [| 1. |]) ])).(1).(3) = 4.) "dw";
  let t =
    Trainer.create ~options:Json.(Obj [ ("optimizer", Obj [ ("kind", Str "adamw"); ("lr", Num 0.1) ]) ]) (Graph.clone g) [| "w" |] [ ("w", [| 0.; 0.; 0.; 0. |]) ]
  in
  let r = Trainer.run t 300 (fun _ -> Some [ ("x", [| 1.; 1. |]); ("y", [| 1.; -1. |]) ]) ~on_step:(fun _ l -> l > 1e-6) in
  check (Option.get r.last_loss < 1e-4) "trainer converges";
  let opt = Optimizer.create ~kind:"sgd" ~options:Json.(Obj [ ("lr", Num 0.5) ]) () in
  check (Optimizer.step opt [ ("p", [| 1.; -1. |]) ] [ ("p", [| 2.; -2. |]) ] = [ ("p", [| 0.; 0. |]) ]) "optimizer";
  let path = Filename.concat (Filename.get_temp_dir_name ()) "rlx-ocaml-test.gguf" in
  let gw = GgufWriter.create () in
  GgufWriter.add_f32 gw "w" [| 4 |] "F32" [| 0.5; -1.; 2.; 0.25 |];
  check (GgufWriter.write ~architecture:"demo" gw path = 1) "write";
  let f = Gguf.open_ path in
  check (Gguf.architecture f = Some "demo") "architecture";
  check ((Gguf.tensor f "w").(2) = 2.) "tensor";
  check (Json.to_str (Json.member "dtype" (Gguf.info f "w")) = "F32") "info";
  Sys.remove path;
  check ((devices ()).(0) = "cpu" && is_available "cpu") "cpu";
  let g, _ = loss_graph () in
  let runner = Runner.create ~policy:Json.(Obj [ ("only", Arr [ Str "cpu" ]) ]) g in
  Runner.set_param runner "w" [| 1.; 0.; 0.; 1. |];
  let dev, outs = Runner.run runner feed in
  check (dev = "cpu" && outs.(0).(0) = 2.5) "runner";
  check ((Group.all_reduce ~op:"max" (Group.connect 0 1) [| 2.; 3. |]).(1) = 3.) "all_reduce";
  Printf.printf "test_rlx: %d checks passed\n" !checks
