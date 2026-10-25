# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# The R binding against the real library (R CMD check runs this); stops on the
# first failure.

library(rlx)
checks <- 0
check <- function(ok, what) {
  checks <<- checks + 1
  if (!isTRUE(ok)) stop("failed: ", what)
}
loss_graph <- function() {
  g <- Graph$new(name = "mse")
  x <- g$input("x", c(1, 2))
  y <- g$input("y", c(1, 2))
  w <- g$param("w", c(2, 2))
  d <- g$sub(g$matmul(x, w), y)
  g$set_outputs(g$mean(g$mul(d, d), c(0, 1)))
  list(g = g, w = w)
}

# inference
lg <- loss_graph()
m <- Session$new()$compile(lg$g)
check(lg$g$is_consumed(), "compile consumes")
m$set_param("w", c(1, 0, 0, 1))
check(m$run(list(x = c(1, 2), y = c(0, 0)))[[1]] == 2.5, "loss 2.5")
check(m$device() == "cpu", "device")
check(identical(rlx_unflat(rlx_flat(matrix(1:4, 2, byrow = TRUE)), c(2, 2)), matrix(1:4, 2, byrow = TRUE)), "flat")

# errors
e <- tryCatch(Graph$new()$relu(42), rlx_error = function(e) e)
check(inherits(e, "rlx_range") && e$kind == "range", "range error")

# training
lg <- loss_graph()
bwd <- Session$new()$compile(rlx_grad(lg$g, lg$w))
bwd$set_param("w", c(1, 0, 0, 1))
check(bwd$run(list(x = c(1, 2), y = c(0, 0), d_output = 1))[[2]][4] == 4, "dw")
t <- Trainer$new(lg$g$clone(), "w", list(w = c(0, 0, 0, 0)), options = list(optimizer = list(kind = "adamw", lr = 0.1)))
r <- t$run(300, function(s) list(x = c(1, 1), y = c(1, -1)), on_step = function(s, l) l > 1e-6)
check(r$last_loss < 1e-4, "trainer converges")
opt <- Optimizer$new(kind = "sgd", options = list(lr = 0.5))
check(identical(opt$step(list(p = c(1, -1)), list(p = c(2, -2)))$p, c(0, 0)), "optimizer")

# loading
path <- file.path(tempdir(), "rlx-r-test.gguf")
gw <- GgufWriter$new()
gw$add_f32("w", 4, "F32", c(0.5, -1, 2, 0.25))
check(gw$write(path, architecture = "demo") == 1, "write")
f <- Gguf$open(path)
check(f$architecture() == "demo", "architecture")
check(f$tensor("w")[3] == 2, "tensor")
check(f$info("w")$dtype == "F32", "info")
unlink(path)

# backends and a one-rank mesh
check(rlx_devices()[[1]] == "cpu" && rlx_is_available("cpu"), "cpu")
lg <- loss_graph()
runner <- Runner$new(lg$g, policy = list(only = list("cpu")))
runner$set_param("w", c(1, 0, 0, 1))
res <- runner$run(list(x = c(1, 2), y = c(0, 0)))
check(res$device == "cpu" && res$outputs[[1]] == 2.5, "runner")
check(Group$connect(0, 1)$all_reduce(c(2, 3), op = "max")[2] == 3, "all_reduce")
cat(sprintf("test_rlx: %d checks passed\n", checks))
