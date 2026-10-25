# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

defmodule Rlx.Manual do
  @moduledoc false
  # The two rows whose shape is not a plain call: Optimizer.step returns the
  # updated params (Elixir values are immutable), Trainer.run takes callbacks.

  alias Rlx.Support, as: S

  def optimizer_step(opt, params, grads) do
    names = params |> Map.keys() |> Enum.map(&to_string/1) |> Enum.sort()
    get = fn m, n -> Map.get(m, n) || Map.get(m, String.to_atom(n)) end

    gs =
      Enum.map(names, fn n ->
        get.(grads, n) || raise Rlx.Error, kind: :type, message: "Optimizer.step: no gradient for '#{n}'"
      end)

    updated = S.unwrap(Rlx.Native.support_optimizer_step(opt, names, Enum.map(names, &get.(params, &1)), gs))
    Map.new(Enum.zip(names, updated))
  end

  # run(trainer, steps, batch, lr: fn step -> rate end, on_step: fn step, loss -> keep_going end, every: 1)
  def trainer_run(t, steps, batch, opts \\ []) do
    lr = Keyword.get(opts, :lr)
    on_step = Keyword.get(opts, :on_step)
    every = max(Keyword.get(opts, :every, 1), 1)
    start = Rlx.Trainer.steps(t)
    t0 = System.monotonic_time(:nanosecond)

    {losses, stopped} =
      Enum.reduce_while(0..(steps - 1)//1, {[], false}, fn i, {acc, _} ->
        s = start + i

        case batch.(s) do
          nil ->
            {:halt, {acc, false}}

          b ->
            if lr, do: Rlx.Trainer.set_lr(t, lr.(s))
            loss = Rlx.Trainer.step(t, b)

            if on_step && rem(i + 1, every) == 0 && !on_step.(s, loss),
              do: {:halt, {[loss | acc], true}},
              else: {:cont, {[loss | acc], false}}
        end
      end)

    losses = Enum.reverse(losses)
    secs = max((System.monotonic_time(:nanosecond) - t0) / 1.0e9, 1.0e-9)

    %{
      steps: length(losses),
      stopped: stopped,
      first_loss: List.first(losses),
      last_loss: List.last(losses),
      mean_loss: if(losses == [], do: nil, else: Enum.sum(losses) / length(losses)),
      steps_per_second: length(losses) / secs,
      grad_norm: Rlx.Trainer.grad_norm(t)
    }
  end
end
