(* RLX — versatile ML compiler + runtime.
   Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
   SPDX-License-Identifier: MIT OR Apache-2.0 *)

(* rlx for OCaml, over its C ABI (librlx_capi): every C function, generated
   by rlx-bindgen from rlx-bind's tables (generated.ml, rlx_stubs.c), on the
   hand-written support.ml and rlx_ml_support.c, plus the two rows whose
   shape is not a plain call (Optimizer.step, Trainer.run) here. *)

include Support
(* Optimizer and Trainer gain their manual rows below. *)
include (
  Generated :
    module type of struct
      include Generated
    end
    with module Optimizer := Generated.Optimizer
     and module Trainer := Generated.Trainer)

external c_optimizer_step : Generated.Optimizer.t option -> string array -> float array array -> float array array -> float array array
  = "caml_rlx_support_optimizer_step"

module Optimizer = struct
  include Generated.Optimizer

  (** One step: the parameters updated from their gradients. *)
  let step self (params : (string * float array) list) (grads : (string * float array) list) =
    let params = List.sort (fun (a, _) (b, _) -> compare a b) params in
    let names = Array.of_list (List.map fst params) in
    let gs =
      Array.map
        (fun n ->
          match List.assoc_opt n grads with
          | Some g -> g
          | None -> raise (Error (Type_error, "Optimizer.step: no gradient for '" ^ n ^ "'")))
        names
    in
    let updated = c_optimizer_step (Some self) names (Array.of_list (List.map snd params)) gs in
    Array.to_list (Array.mapi (fun i n -> (n, updated.(i))) names)
end

(** What [Trainer.run] did. *)
type train_summary = {
  steps : int;
  stopped : bool;
  first_loss : float option;
  last_loss : float option;
  mean_loss : float option;
  steps_per_second : float;
  grad_norm : float option;
}

module Trainer = struct
  include Generated.Trainer

  (** Train [steps] steps, pulling inputs from [batch step] ([None]: data
      exhausted). [?lr step] schedules the rate; [?on_step step loss]
      returning [false] stops early (called every [?every] steps). *)
  let run ?lr ?on_step ?(every = 1) self steps batch =
    let every = max every 1 in
    let start = Generated.Trainer.steps self in
    let t0 = Unix.gettimeofday () in
    let rec go i acc =
      if i >= steps then (List.rev acc, false)
      else
        let s = start + i in
        match batch s with
        | None -> (List.rev acc, false)
        | Some b ->
            Option.iter (fun f -> Generated.Trainer.set_lr self (f s)) lr;
            let loss = Generated.Trainer.step self b in
            let keep = match on_step with Some f when (i + 1) mod every = 0 -> f s loss | _ -> true in
            if keep then go (i + 1) (loss :: acc) else (List.rev (loss :: acc), true)
    in
    let losses, stopped = go 0 [] in
    let n = List.length losses in
    let secs = Float.max (Unix.gettimeofday () -. t0) 1e-9 in
    {
      steps = n;
      stopped;
      first_loss = (match losses with x :: _ -> Some x | [] -> None);
      last_loss = (if n > 0 then Some (List.nth losses (n - 1)) else None);
      mean_loss = (if n > 0 then Some (List.fold_left ( +. ) 0. losses /. float n) else None);
      steps_per_second = float n /. secs;
      grad_norm = Generated.Trainer.grad_norm self;
    }
end
