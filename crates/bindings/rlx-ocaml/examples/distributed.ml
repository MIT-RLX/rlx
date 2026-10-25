(* RLX — versatile ML compiler + runtime.
   Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
   SPDX-License-Identifier: MIT OR Apache-2.0 *)

(* Distributed: each rank is a process; start one per rank with the same
   peers (here three on loopback; across machines, one per host):
     distributed.exe 1 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
     distributed.exe 2 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
     distributed.exe 0 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502
   Rank 0 coordinates: collectives, pipelined inference (a stage shipped per
   worker), data-parallel training. With no arguments it is a mesh of one. *)
open Rlx

let per_rank = 16

let () =
  let arg i = if Array.length Sys.argv > i then Some Sys.argv.(i) else None in
  let rank = Option.fold ~none:0 ~some:int_of_string (arg 1) in
  let world = Option.fold ~none:1 ~some:int_of_string (arg 2) in
  let blobs = ref [] in
  let blob name values =
    let path = Filename.concat (Filename.get_temp_dir_name ()) (Printf.sprintf "rlx-dist-ocaml-%d-%s" rank name) in
    let b = Buffer.create 64 in
    Array.iter (fun v -> Buffer.add_int32_le b (Int32.bits_of_float v)) values;
    Out_channel.with_open_bin path (fun oc -> Buffer.output_buffer oc b);
    blobs := path :: !blobs;
    "file://" ^ path
  in
  let options =
    Json.Obj
      (("device", Json.Str "cpu")
      :: (match arg 3 with Some p -> [ ("peers", Json.Arr (List.map (fun s -> Json.Str s) (String.split_on_char ',' p))) ] | None -> []))
  in
  let g = Group.connect ~options rank world in
  let show a = String.concat "," (Array.to_list (Array.map (Printf.sprintf "%g") a)) in
  let total = Group.all_reduce ~op:"sum" g [| float (rank + 1) |] in
  let gathered = Group.all_gather g [| float (rank + 1) |] in
  let shared = Group.broadcast g 0 [| (if rank = 0 then 42. else 0.) |] (* the root's length *) in
  if rank = 0 then begin
    Printf.printf "mesh: rank %d of %d, leader: %b\n" (Group.rank g) (Group.world g) (Group.is_leader g);
    Printf.printf "all_reduce sum: %s\nall_gather: %s\nbroadcast: %s\n" (show total) (show gathered) (show shared)
  end;
  if world = 1 then print_endline "a mesh of one: pass rank world peers to start more ranks"
  else if rank <> 0 then begin
    ignore (Group.serve_worker ~options:Json.(Obj [ ("steps", Num 2.) ]) g);
    let r = Group.serve_trainer g in
    let w = Array.of_list (List.map Json.to_float (Json.to_list (Json.member "w" (Json.member "params" r)))) in
    Group.send g 0 1 w (* rank 0 compares *)
  end
  else begin
    let stage_w = [| Array.init 16 (fun i -> float (i * 5 mod 7) /. 7. -. 0.3); Array.init 16 (fun i -> float (i * 3 mod 5) /. 5. -. 0.2) |] in
    let caps = Json.to_list (Group.collect_caps g) in
    Printf.printf "workers: %s\n"
      (String.concat ", " (List.map (fun c -> Printf.sprintf "rank %d on %s" (Json.to_int (Json.member "rank" c)) (Json.to_str (Json.member "platform" c))) caps));
    let stage = Graph.create ~name:"stage" () in
    Graph.set_outputs stage [| Graph.relu stage (Graph.matmul stage (Graph.input stage "x" [| 1; 4 |]) (Graph.param stage "w" [| 4; 4 |])) |];
    for r = 1 to world - 1 do
      Group.ship_stage g r stage
        Json.(Obj [ ("device", Str "cpu"); ("weights", Obj [ ("w", Str (blob (Printf.sprintf "w%d.bin" r) stage_w.((r - 1) mod 2))) ]) ])
    done;
    List.iter
      (fun xv ->
        Group.send_activation g 1 xv;
        let y = Group.recv_activation g (world - 1) in
        Printf.printf "pipeline: %s\n" (String.concat " " (Array.to_list (Array.map (Printf.sprintf "%.3f") y))))
      [ [| 1.; -1.; 0.5; 2. |]; [| 0.25; 0.5; -0.75; 1. |] ];
    ignore (Group.collect_caps g);
    let lr = Graph.create ~name:"lr" () in
    let bx = Graph.input lr "X" [| 4; 3 |] and by = Graph.input lr "y" [| 4; 1 |] and wv = Graph.param lr "w" [| 3; 1 |] in
    let d = Graph.sub lr (Graph.matmul lr bx wv) by in
    Graph.set_outputs lr [| Graph.mean lr (Graph.mul lr d d) [| 0; 1 |] |];
    let bwd = grad lr [| wv |] in
    let w_true = [| 1.5; -2.; 0.5 |] in
    let n = per_rank * world in
    let x_all = Array.init (n * 3) (fun i -> float (i * 37 mod 23) /. 11.5 -. 1.) in
    let y_all = Array.init n (fun i -> (x_all.(i * 3) *. w_true.(0)) +. (x_all.((i * 3) + 1) *. w_true.(1)) +. (x_all.((i * 3) + 2) *. w_true.(2))) in
    let ux = blob "X.bin" x_all and uy = blob "y.bin" y_all and uw = blob "w0.bin" [| 0.; 0.; 0. |] in
    let spec r =
      Json.(
        Obj
          [
            ("params", Arr [ Obj [ ("name", Str "w"); ("uri", Str uw) ] ]);
            ( "data",
              Arr
                [
                  Obj [ ("input", Str "X"); ("uri", Str ux); ("elem", Num 3.); ("shard_start", Num (float (r * per_rank))); ("shard_len", Num (float per_rank)) ];
                  Obj [ ("input", Str "y"); ("uri", Str uy); ("elem", Num 1.); ("shard_start", Num (float (r * per_rank))); ("shard_len", Num (float per_rank)) ];
                ] );
            ("batch", Num 4.);
            ("lr_per_epoch", Arr (List.init 40 (fun _ -> Num 0.2)));
            ("device", Str "cpu");
          ])
    in
    for r = 1 to world - 1 do Group.ship_train g r bwd (spec r) done;
    let mine = Group.run_train g bwd (spec 0) in
    let metrics = Json.member "metrics" mine in
    Printf.printf "training loss: %.4f -> %.2e\n" (Json.to_float (Json.member "first_loss" metrics)) (Json.to_float (Json.member "last_loss" metrics));
    let w0 = Array.of_list (List.map Json.to_float (Json.to_list (Json.member "w" (Json.member "params" mine)))) in
    Printf.printf "rank 0 w: %s (true: 1.5 -2 0.5)\n" (String.concat " " (Array.to_list (Array.map (Printf.sprintf "%.3f") w0)));
    for r = 1 to world - 1 do
      let theirs = Group.recv g r 1 in
      Printf.printf "rank %d w agrees: %b\n" r (Array.for_all2 (fun a b -> Float.abs (a -. b) < 1e-5) theirs w0)
    done;
    List.iter Sys.remove !blobs
  end
