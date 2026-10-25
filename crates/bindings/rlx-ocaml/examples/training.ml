(* RLX — versatile ML compiler + runtime.
   Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
   SPDX-License-Identifier: MIT OR Apache-2.0 *)

(* Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
   the same fit by hand with grad and an Optimizer.
     dune exec examples/training.exe -- [device] *)
open Rlx

let w_true = [| 2.; -1.; 0.5 |]
let batch = 8

let data step =
  let x = Array.make (batch * 3) 0. and y = Array.make batch 0. in
  for i = 0 to batch - 1 do
    for k = 0 to 2 do
      let v = float ((step * 31 + i * 7 + k * 3) mod 17) /. 8.5 -. 1. in
      x.((i * 3) + k) <- v;
      y.(i) <- y.(i) +. (v *. w_true.(k))
    done
  done;
  [ ("x", x); ("y", y) ]

let fmt v = String.concat " " (Array.to_list (Array.map (Printf.sprintf "%.3f") v))

let () =
  let device = if Array.length Sys.argv > 1 then Sys.argv.(1) else "cpu" in
  let g = Graph.create ~name:"regression" () in
  let x = Graph.input g "x" [| batch; 3 |] and y = Graph.input g "y" [| batch; 1 |] in
  let w = Graph.param g "w" [| 3; 1 |] in
  let d = Graph.sub g (Graph.matmul g x w) y in
  Graph.set_outputs g [| Graph.mean g (Graph.mul g d d) [| 0; 1 |] |];
  let options = Json.(Obj [ ("device", Str device); ("optimizer", Obj [ ("kind", Str "adamw"); ("lr", Num 0.1) ]) ]) in
  let trainer = Trainer.create ~options (Graph.clone g) [| "w" |] [ ("w", [| 0.; 0.; 0. |]) ] in
  let first = Trainer.evaluate trainer (data 0) in
  ignore
    (Trainer.run trainer 400 (fun s -> Some (data s))
       ~lr:(fun s -> 0.1 *. 0.5 *. (1. +. cos (Float.pi *. float s /. 400.)))
       ~on_step:(fun _ loss -> loss > 1e-6));
  Printf.printf "trainer loss: %.4f -> %.6f in %d steps\n" first (Trainer.evaluate trainer (data 0)) (Trainer.steps trainer);
  Printf.printf "trainer w: %s\n" (fmt (List.assoc "w" (Trainer.params trainer)));
  let path = Filename.concat (Filename.get_temp_dir_name ()) "rlx-example-train-ocaml.gguf" in
  let written = Trainer.save trainer path in
  let restored = Trainer.create ~options:Json.(Obj [ ("device", Str device) ]) (Graph.clone g) [| "w" |] [ ("w", [| 0.; 0.; 0. |]) ] in
  let info = Trainer.load restored path in
  Printf.printf "checkpoint: %d tensors, restored step %d\n" written (Json.to_int (Json.member "step" info));
  Sys.remove path;
  ignore (Trainer.freeze ~names:[| "w" |] restored);
  let before = fmt (List.assoc "w" (Trainer.params restored)) in
  ignore (Trainer.step restored (data 1));
  Printf.printf "frozen: [%s] unchanged: %b\n" (String.concat "," (Array.to_list (Trainer.frozen restored)))
    (fmt (List.assoc "w" (Trainer.params restored)) = before);
  ignore (Trainer.unfreeze restored);
  let backward = Session.compile (Session.create ~device ()) (grad g [| w |]) (* outputs [loss, dw] *) in
  let opt = Optimizer.create ~kind:"sgd" ~options:Json.(Obj [ ("lr", Num 0.2) ]) () in
  let params = ref [ ("w", [| 0.; 0.; 0. |]) ] and loss = ref 0. in
  for step = 0 to 299 do
    Compiled.set_param backward "w" (List.assoc "w" !params);
    let outs = Compiled.run backward (data step @ [ ("d_output", [| 1. |]) ]) in
    loss := outs.(0).(0);
    params := Optimizer.step opt !params [ ("w", outs.(1)) ]
  done;
  Printf.printf "manual loss: %.6f w: %s\n" !loss (fmt (List.assoc "w" !params))
