(* RLX — versatile ML compiler + runtime.
   Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
   SPDX-License-Identifier: MIT OR Apache-2.0 *)

(* Loading: a GGUF file (written here, f32 and quantized) and a safetensors
   file into a compiled graph, and quantizing in memory.
     dune exec examples/loading.exe *)
open Rlx

let () =
  let tmp = Filename.get_temp_dir_name () in
  let w = Array.init 64 (fun i -> float (i mod 9) /. 4. -. 1.) in
  let gguf = Filename.concat tmp "rlx-example-ocaml.gguf" in
  let writer = GgufWriter.create () in
  GgufWriter.add_f32 writer "proj.weight" [| 8; 8 |] "F32" w;
  GgufWriter.add_f32 writer "proj.weight_q8" [| 8; 8 |] "Q8_0" w;
  ignore (GgufWriter.write ~architecture:"demo" ~metadata:Json.(Obj [ ("demo.layers", Num 1.) ]) writer gguf);
  let f = Gguf.open_ gguf in
  Printf.printf "gguf architecture: %s\n" (Option.value (Gguf.architecture f) ~default:"");
  Printf.printf "gguf tensors: %s\n" (String.concat " " (Array.to_list (Gguf.tensor_names f)));
  let exact = Gguf.tensor f "proj.weight" and q8 = Gguf.tensor f "proj.weight_q8" in
  let maxd = ref 0. in
  Array.iteri (fun i v -> maxd := Float.max !maxd (Float.abs (v -. q8.(i)))) exact;
  Printf.printf "gguf f32 exact: %b q8 max|Δ|: %.4f\n" (exact = w) !maxd;
  let g = Graph.create ~name:"proj" () in
  let x = Graph.input g "x" [| 1; 8 |] in
  Graph.set_outputs g [| Graph.matmul g x (Graph.param g "proj.weight" [| 8; 8 |]) |];
  let c = Session.compile (Session.create ()) g in
  Compiled.set_param c "proj.weight" exact;
  let y = (Compiled.run c [ ("x", Array.make 8 1.) ]).(0) in
  Printf.printf "gguf -> graph: %.2f %.2f %.2f\n" y.(0) y.(1) y.(2);
  let st = Filename.concat tmp "rlx-example-ocaml.safetensors" in
  let header = {|{"proj.weight":{"dtype":"F32","shape":[8,8],"data_offsets":[0,256]}}|} in
  let header = header ^ String.make ((8 - (String.length header mod 8)) mod 8) ' ' in
  let b = Buffer.create 512 in
  Buffer.add_int64_le b (Int64.of_int (String.length header));
  Buffer.add_string b header;
  Array.iter (fun v -> Buffer.add_int32_le b (Int32.bits_of_float v)) w;
  Out_channel.with_open_bin st (fun oc -> Buffer.output_buffer oc b);
  let loaded = load_weights "safetensors" st in
  Printf.printf "safetensors: [proj.weight] exact: %b\n" (List.assoc "proj.weight" loaded = w);
  let packed = quantize w "Q8_0" in
  Printf.printf "quantize q8_0: %d -> %d bytes, round trip %d values\n" (Array.length w * 4) (String.length packed)
    (Array.length (dequantize ~count:(Array.length w) packed "Q8_0"));
  Sys.remove gguf;
  Sys.remove st;
  Printf.printf "argmax: %d\n" (argmax [| 0.1; 2.; 0.3 |])
