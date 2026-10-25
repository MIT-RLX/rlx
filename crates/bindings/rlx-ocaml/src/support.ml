(* RLX — versatile ML compiler + runtime.
   Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
   SPDX-License-Identifier: MIT OR Apache-2.0 *)

(* What the generated generated.ml stands on: errors, JSON, and the
   conversions the stubs (rlx_stubs.c, rlx_ml_support.c) do not do. *)

(** What kind of failure an [Error] is (the [RLX_ERR_*] status). *)
type error_kind = Type_error | Range_error | Runtime_error | Panic | Not_found_error | Other_error

(** A failed rlx call. *)
exception Error of error_kind * string

let () =
  Callback.register_exception "rlx_error" (Error (Other_error, ""));
  Printexc.register_printer (function
    | Error (_, m) -> Some ("Rlx.Error: " ^ m)
    | _ -> None)

(** "No node", for an optional node argument. *)
let none = 0xFFFFFFFF

(** A tensor of any dtype: its little-endian bytes and dtype name. *)
type typed_tensor = { data : string; dtype : string }

(** JSON: options in, reports out. *)
module Json = struct
  type t = Null | Bool of bool | Num of float | Str of string | Arr of t list | Obj of (string * t) list

  let rec to_string = function
    | Null -> "null"
    | Bool b -> string_of_bool b
    | Num f ->
        if Float.is_integer f && Float.abs f < 1e15 then Printf.sprintf "%.0f" f
        else Printf.sprintf "%.17g" f
    | Str s -> "\"" ^ String.concat "\\\"" (String.split_on_char '"' (String.concat "\\\\" (String.split_on_char '\\' s))) ^ "\""
    | Arr l -> "[" ^ String.concat "," (List.map to_string l) ^ "]"
    | Obj kv -> "{" ^ String.concat "," (List.map (fun (k, v) -> to_string (Str k) ^ ":" ^ to_string v) kv) ^ "}"

  let of_string s =
    let n = String.length s in
    let i = ref 0 in
    let ws () = while !i < n && String.contains " \t\r\n" s.[!i] do incr i done in
    let rec value () =
      ws ();
      match s.[!i] with
      | '{' ->
          incr i;
          ws ();
          if s.[!i] = '}' then (incr i; Obj [])
          else
            let rec fields acc =
              ws ();
              let k = match value () with Str k -> k | _ -> failwith "json key" in
              ws ();
              incr i;
              let v = value () in
              ws ();
              let c = s.[!i] in
              incr i;
              if c = '}' then Obj (List.rev ((k, v) :: acc)) else fields ((k, v) :: acc)
            in
            fields []
      | '[' ->
          incr i;
          ws ();
          if s.[!i] = ']' then (incr i; Arr [])
          else
            let rec items acc =
              let v = value () in
              ws ();
              let c = s.[!i] in
              incr i;
              if c = ']' then Arr (List.rev (v :: acc)) else items (v :: acc)
            in
            items []
      | '"' ->
          incr i;
          let b = Buffer.create 16 in
          while s.[!i] <> '"' do
            (if s.[!i] = '\\' then (
               incr i;
               match s.[!i] with
               | 'n' -> Buffer.add_char b '\n'
               | 't' -> Buffer.add_char b '\t'
               | 'r' -> Buffer.add_char b '\r'
               | 'u' ->
                   let code = int_of_string ("0x" ^ String.sub s (!i + 1) 4) in
                   Buffer.add_utf_8_uchar b (Uchar.of_int code);
                   i := !i + 4
               | c -> Buffer.add_char b c)
             else Buffer.add_char b s.[!i]);
            incr i
          done;
          incr i;
          Str (Buffer.contents b)
      | 't' -> i := !i + 4; Bool true
      | 'f' -> i := !i + 5; Bool false
      | 'n' -> i := !i + 4; Null
      | _ ->
          let j = !i in
          while !i < n && String.contains "+-0123456789.eE" s.[!i] do incr i done;
          Num (float_of_string (String.sub s j (!i - j)))
    in
    value ()

  let member k = function Obj kv -> (try List.assoc k kv with Not_found -> Null) | _ -> Null
  let to_list = function Arr l -> l | _ -> []
  let to_float = function Num f -> f | _ -> nan
  let to_int j = int_of_float (to_float j)
  let to_str = function Str s -> s | _ -> ""
  let strings s = Array.of_list (List.map to_str (to_list (of_string s)))
  let ints s = Array.of_list (List.map to_int (to_list (of_string s)))
  let int_rows s = Array.of_list (List.map (fun r -> Array.of_list (List.map to_int (to_list r))) (to_list (of_string s)))
end

let rows (v : int array array) w splat =
  Array.concat
    (Array.to_list
       (Array.mapi
          (fun i r ->
            if Array.length r = w then r
            else if splat && Array.length r = 1 then Array.make w r.(0)
            else raise (Error (Type_error, Printf.sprintf "row %d needs %d values, got %d" i w (Array.length r))))
          v))

let frows (v : float array array) w =
  Array.concat
    (Array.to_list
       (Array.mapi
          (fun i r ->
            if Array.length r = w then r
            else raise (Error (Type_error, Printf.sprintf "row %d needs %d values, got %d" i w (Array.length r))))
          v))

type inputs = { inames : string array; idata : float array array; ilens : int array }

let inputs (m : (string * float array) list) =
  let m = List.sort (fun (a, _) (b, _) -> compare a b) m in
  let idata = Array.of_list (List.map snd m) in
  { inames = Array.of_list (List.map fst m); idata; ilens = Array.map Array.length idata }

type typed_inputs = { tnames : string array; tdata : string array; tlens : int array; tdtypes : string array }

let typed_inputs (m : (string * typed_tensor) list) =
  let m = List.sort (fun (a, _) (b, _) -> compare a b) m in
  let tdata = Array.of_list (List.map (fun (_, t) -> t.data) m) in
  { tnames = Array.of_list (List.map fst m); tdata; tlens = Array.map String.length tdata;
    tdtypes = Array.of_list (List.map (fun (_, t) -> t.dtype) m) }
