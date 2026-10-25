# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

defmodule Rlx.Error do
  @moduledoc "A failed rlx call: `kind` is :type, :range, :runtime, :panic, :not_found or :other."
  defexception [:kind, :message]
end

defmodule Rlx.Support do
  @moduledoc false
  # What the generated lib/rlx/generated.ex stands on: loading the NIF,
  # results, options, conversions.

  @none 0xFFFFFFFF
  def none, do: @none

  # The NIF: RLX_NIF when set, else priv/rlx_nif of the :rlx app (or of this checkout).
  def nif_path do
    path =
      System.get_env("RLX_NIF") ||
        case :code.priv_dir(:rlx) do
          {:error, _} -> Path.expand("../../priv/rlx_nif", __DIR__)
          dir -> Path.join(dir, "rlx_nif")
        end

    String.to_charlist(path)
  end

  def unwrap({:ok, v}), do: v
  def unwrap({:error, kind, message}), do: raise(Rlx.Error, kind: kind, message: message)

  def opts(given, defaults) do
    unknown = Keyword.keys(given) -- Keyword.keys(defaults)

    if unknown != [] do
      raise Rlx.Error,
        kind: :type,
        message: "unknown option #{inspect(hd(unknown))} (takes: #{Enum.map_join(Keyword.keys(defaults), ", ", &inspect/1)})"
    end

    Map.new(Keyword.merge(defaults, given))
  end

  def json(v), do: JSON.encode!(v)
  def json_or_nil(nil), do: nil
  def json_or_nil(v), do: json(v)
  def decode(nil), do: nil
  def decode(s), do: JSON.decode!(s)

  def box(nil), do: []
  def box(v), do: [v]

  def optional({v, true}), do: v
  def optional({_, false}), do: nil

  def rows(v, w, splat) do
    v
    |> Enum.with_index()
    |> Enum.flat_map(fn
      {r, _} when length(r) == w -> r
      {[x], _} when splat -> List.duplicate(x, w)
      {r, i} -> raise Rlx.Error, kind: :type, message: "row #{i} needs #{w} values, got #{length(r)}"
    end)
  end

  defp entries(m), do: m |> Enum.map(fn {k, v} -> {to_string(k), v} end) |> Enum.sort_by(&elem(&1, 0))

  def inputs(m) do
    e = entries(m)
    data = Enum.map(e, &elem(&1, 1))
    %{names: Enum.map(e, &elem(&1, 0)), data: data, lens: Enum.map(data, &length/1)}
  end

  def typed_inputs(m) do
    e = entries(m)
    data = Enum.map(e, fn {_, t} -> t.data end)

    %{
      names: Enum.map(e, &elem(&1, 0)),
      data: data,
      lens: Enum.map(data, &byte_size/1),
      dtypes: Enum.map(e, fn {_, t} -> t.dtype end)
    }
  end
end
