# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Training: fit y = x·w_true with the Trainer, checkpoint and resume it, then
# the same fit by hand with grad and an Optimizer.
#   Rscript training.R [device]

library(rlx)
args <- commandArgs(trailingOnly = TRUE)
device <- if (length(args)) args[[1]] else "cpu"
w_true <- c(2, -1, 0.5)
batch <- 8
data <- function(step) {
  x <- numeric(0)
  y <- numeric(0)
  for (i in 0:(batch - 1)) {
    v <- ((step * 31 + i * 7 + (0:2) * 3) %% 17) / 8.5 - 1
    x <- c(x, v)
    y <- c(y, sum(v * w_true))
  }
  list(x = x, y = y)
}
fmt <- function(v) paste(sprintf("%.3f", v), collapse = " ")

g <- Graph$new(name = "regression")
x <- g$input("x", c(batch, 3))
y <- g$input("y", c(batch, 1))
w <- g$param("w", c(3, 1))
d <- g$sub(g$matmul(x, w), y)
g$set_outputs(g$mean(g$mul(d, d), c(0, 1)))

trainer <- Trainer$new(g$clone(), "w", list(w = c(0, 0, 0)),
                       options = list(device = device, optimizer = list(kind = "adamw", lr = 0.1)))
first <- trainer$evaluate(data(0))
invisible(trainer$run(400, data, lr = function(s) 0.1 * 0.5 * (1 + cos(pi * s / 400)),
                      on_step = function(s, loss) loss > 1e-6))
cat(sprintf("trainer loss: %.4f -> %.6f in %d steps\n", first, trainer$evaluate(data(0)), as.integer(trainer$steps())))
cat(sprintf("trainer w: %s\n", fmt(trainer$params()$w)))

path <- file.path(tempdir(), "rlx-example-train-r.gguf")
written <- trainer$save(path)
restored <- Trainer$new(g$clone(), "w", list(w = c(0, 0, 0)), options = list(device = device))
cat(sprintf("checkpoint: %d tensors, restored step %d\n", as.integer(written), as.integer(restored$load(path)$step)))
unlink(path)
invisible(restored$freeze(names = "w"))
before <- fmt(restored$params()$w)
invisible(restored$step(data(1)))
cat(sprintf("frozen: %s unchanged: %s\n", paste(restored$frozen(), collapse = ","), tolower(fmt(restored$params()$w) == before)))
invisible(restored$unfreeze())

backward <- Session$new(device = device)$compile(rlx_grad(g, w)) # outputs [loss, dw]
opt <- Optimizer$new(kind = "sgd", options = list(lr = 0.2))
params <- list(w = c(0, 0, 0))
loss <- 0
for (step in 0:299) {
  backward$set_param("w", params$w)
  outs <- backward$run(c(data(step), list(d_output = 1)))
  loss <- outs[[1]]
  params <- opt$step(params, list(w = outs[[2]]))
}
cat(sprintf("manual loss: %.6f w: %s\n", loss, fmt(params$w)))
