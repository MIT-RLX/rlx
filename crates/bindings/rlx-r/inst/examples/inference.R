# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Inference: build a graph, compile it for a device, set its weights, run it.
#   Rscript inference.R [device]
# A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].

library(rlx)
args <- commandArgs(trailingOnly = TRUE)
device <- if (length(args)) args[[1]] else "cpu"
weights <- function(n, seed) ((0:(n - 1)) * 7919 + seed * 104729) %% 1000 / 1000 - 0.5

g <- Graph$new(name = "classifier")
x <- g$input("x", c(1, 4))
hidden <- g$relu(g$matmul(x, g$param("w1", c(4, 8))))
g$set_outputs(g$softmax(g$matmul(hidden, g$param("w2", c(8, 3)))))

model <- Session$new(device = device)$compile(g) # consumes g
cat(sprintf("compiled on: %s\n", model$device()))
invisible(model$set_params(list(w1 = weights(32, 1), w2 = weights(24, 2))))
probs <- model$run(list(x = c(0.5, -1, 2, 0.25)))[[1]]
cat(sprintf("probs: %s\n", paste(sprintf("%.4f", probs), collapse = " ")))
cat(sprintf("probs sum: %.4f\n", sum(probs)))
cat(sprintf("output shapes: %s dtypes: %s\n", jsonlite::toJSON(model$output_shapes()), paste(model$output_dtypes(), collapse = ",")))
