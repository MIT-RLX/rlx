# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

defmodule Rlx.MixProject do
  use Mix.Project

  # The NIF (priv/rlx_nif.so) is built by `make` against librlx_capi in
  # RLX_CAPI_LIB_DIR; see README.md.
  def project do
    [
      app: :rlx,
      version: "0.2.19",
      elixir: "~> 1.18",
      description: "Elixir bindings for rlx over its C ABI.",
      package: [licenses: ["MIT", "Apache-2.0"], files: ~w(lib c_src Makefile mix.exs README.md)],
      deps: []
    ]
  end

  def application, do: []
end
