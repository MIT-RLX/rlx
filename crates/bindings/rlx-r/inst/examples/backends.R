# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Picking a backend: what this build has, what suits a graph, and the ways to
# choose — by name, by cost model, by policy, by benchmark, per run.
#   Rscript backends.R

library(rlx)
model <- function() {
  g <- Graph$new(name = "mlp")
  x <- g$input("x", c(4, 16))
  g$set_outputs(g$tanh(g$matmul(x, g$param("w", c(16, 16)))))
  g
}
w <- (0:255 %% 5) / 10 - 0.2
feed <- list(x = (0:63 %% 7) / 7)

available <- rlx_devices()
cat(sprintf("devices: %s\n", paste(available, collapse = " ")))
cat(sprintf("metal available: %s\n", tolower(rlx_is_available("metal"))))
cat(sprintf("fastest for graph: %s\n", rlx_fastest_device_for(model())))
reference <- NULL
for (device in available) {
  c1 <- Session$new(device = device)$compile(model())
  c1$set_param("w", w)
  y <- c1$run(feed)[[1]]
  if (is.null(reference)) reference <- y
  cat(sprintf("session %s: max|Δ| vs first = %.2e\n", device, max(abs(y - reference))))
}
flex <- FlexibleSession$new(policy = list(prefer = c("metal", "cpu")))
cat(sprintf("flexible session picked: %s\n", flex$compile(model())$device()))
runner <- Runner$new(model())
runner$set_param("w", w)
cat(sprintf("runner warmed: %s\n", paste(runner$warm_all(), collapse = " ")))
for (row in runner$benchmark(feed, runs = 3)) cat(sprintf("  bench %s: exec_ms=%g\n", row$device, row$exec_ms))
cat(sprintf("runner ran on: %s\n", runner$run(feed)$device))
cat(sprintf("run_try chain: %s\n", runner$run_try(c("metal", "cpu"), feed)$device))
router <- Router$new(model())
router$set_param("w", w)
cat(sprintf("router ran on: %s\n", router$run_chain(feed)$device))
