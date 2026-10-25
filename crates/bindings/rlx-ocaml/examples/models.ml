(* RLX — versatile ML compiler + runtime.
   Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
   SPDX-License-Identifier: MIT OR Apache-2.0 *)

(* Models: any provider's model, opened by path or id and run by name. A graph
   saved as an .rlxp package runs anywhere; a plugin's models (a downstream
   collection's LLMs, or the reference plugin's demo:count) load through
   RLX_MODEL_PLUGINS.
     RLX_MODEL_PLUGINS=…/librlx_model_plugin.so dune exec examples/models.exe *)
open Rlx

let () =
  let g = Graph.create ~name:"affine" () in
  let x = Graph.input g "x" [| 1; 2 |] in
  Graph.set_outputs g [| Graph.matmul g x (Graph.param g "w" [| 2; 2 |]) |];
  let path = Filename.concat (Filename.get_temp_dir_name ()) "rlx-example-ocaml.rlxp" in
  save_model g path [ ("w", [| 2.; 0.; 0.; 3. |]) ]
    ~model:Json.(Obj [ ("task", Str "graph"); ("family", Str "affine"); ("outputs", Arr [ Str "y" ]) ]);

  let m = Model.open_ ~options:Json.(Obj [ ("device", Str "cpu") ]) path in
  let info = Model.info m in
  let s k v = Json.to_str (Json.member k v) in
  Printf.printf "model: %s (%s) on %s\n" (s "family" info) (s "task" info) (s "device" info);
  let y = (Model.run m [ ("x", [| 1.; 1. |]) ]).(0) in
  Printf.printf "model run: %.3f %.3f\n" y.(0) y.(1);
  Model.close m;
  Sys.remove path;

  let lm = Model.open_ "demo:count" in
  let lm_info = Model.info lm in
  Printf.printf "plugin model: %s (%s)\n" (s "family" lm_info) (s "task" lm_info);
  Printf.printf "model text: %s\n" (Model.generate_text ~options:Json.(Obj [ ("max_tokens", Num 4.) ]) lm "c");
  Printf.printf "model ids: %s\n"
    (String.concat " " (Array.to_list (Array.map string_of_int (Model.generate ~max_tokens:3 lm [| 2 |]))));
  (* By task: a chat, speech out and back in, embeddings. *)
  let msgs = Json.(Arr [ Obj [ ("role", Str "user"); ("content", Str "c") ] ]) in
  Printf.printf "chat: %s\n" (Model.chat ~options:Json.(Obj [ ("max_tokens", Num 4.) ]) lm msgs);
  Model.close lm;
  let voice = Model.open_ "demo:voice" and ear = Model.open_ "demo:speech" and embedder = Model.open_ "demo:embed" in
  let audio, rate = Model.synthesize voice "hello" in
  Printf.printf "voice: %d samples at %d Hz\n" (Array.length audio) rate;
  Printf.printf "heard: %s\n" (Model.transcribe ~sample_rate:rate ear audio);
  let rows = Model.embed embedder [| "hello"; "sky" |] in
  let values = List.concat_map (fun r -> List.map (fun v -> string_of_int (int_of_float (Json.to_float v))) (Json.to_list r)) (Json.to_list rows) in
  Printf.printf "embed: %s\n" (String.concat " " values);
  List.iter Model.close [ voice; ear; embedder ]
