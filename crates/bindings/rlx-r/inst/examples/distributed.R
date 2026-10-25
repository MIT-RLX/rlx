# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Distributed: R is single-threaded, so each rank is a process; start one per
# rank with the same peers (here three on loopback; across machines, one per
# host):
#   Rscript distributed.R 1 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
#   Rscript distributed.R 2 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 &
#   Rscript distributed.R 0 3 127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502
# Rank 0 coordinates: collectives, pipelined inference (a stage shipped per
# worker), data-parallel training. With no arguments it is a mesh of one.

library(rlx)
args <- commandArgs(trailingOnly = TRUE)
rank <- if (length(args) >= 1) as.integer(args[[1]]) else 0L
world <- if (length(args) >= 2) as.integer(args[[2]]) else 1L
per_rank <- 16
blobs <- character(0)
blob <- function(name, values) {
  path <- file.path(tempdir(), paste0("rlx-dist-r-", rank, "-", name))
  writeBin(as.numeric(values), path, size = 4, endian = "little")
  blobs <<- c(blobs, path)
  paste0("file://", path)
}

options <- list(device = "cpu")
if (length(args) >= 3) options$peers <- strsplit(args[[3]], ",")[[1]]
g <- Group$connect(rank, world, options = options)
total <- g$all_reduce(rank + 1, op = "sum")
gathered <- g$all_gather(rank + 1)
shared <- g$broadcast(0, if (rank == 0) 42 else 0) # the root's length
if (rank == 0) {
  cat(sprintf("mesh: rank %d of %d, leader: %s\n", as.integer(g$rank()), as.integer(g$world()), tolower(g$is_leader())))
  cat(sprintf("all_reduce sum: %g\n", total))
  cat(sprintf("all_gather: %s\n", paste(gathered, collapse = ",")))
  cat(sprintf("broadcast: %g\n", shared))
}
if (world == 1) {
  cat("a mesh of one: pass rank world peers to start more ranks\n")
  quit(save = "no")
}
if (rank != 0) {
  invisible(g$serve_worker(options = list(steps = 2)))
  g$send(0, 1, g$serve_trainer()$params$w) # rank 0 compares
  quit(save = "no")
}

stage_w <- list(((0:15) * 5) %% 7 / 7 - 0.3, ((0:15) * 3) %% 5 / 5 - 0.2)
caps <- g$collect_caps()
cat(sprintf("workers: %s\n", paste(vapply(caps, function(c) sprintf("rank %d on %s", as.integer(c$rank), c$platform), ""), collapse = ", ")))
stage <- Graph$new(name = "stage")
stage$set_outputs(stage$relu(stage$matmul(stage$input("x", c(1, 4)), stage$param("w", c(4, 4)))))
for (r in seq_len(world - 1)) {
  g$ship_stage(r, stage, list(device = "cpu", weights = list(w = blob(sprintf("w%d.bin", r), stage_w[[(r - 1) %% 2 + 1]]))))
}
for (xv in list(c(1, -1, 0.5, 2), c(0.25, 0.5, -0.75, 1))) {
  g$send_activation(1, xv)
  cat(sprintf("pipeline: %s\n", paste(sprintf("%.3f", g$recv_activation(world - 1)), collapse = " ")))
}

invisible(g$collect_caps())
lr <- Graph$new(name = "lr")
X <- lr$input("X", c(4, 3))
Y <- lr$input("y", c(4, 1))
wv <- lr$param("w", c(3, 1))
d <- lr$sub(lr$matmul(X, wv), Y)
lr$set_outputs(lr$mean(lr$mul(d, d), c(0, 1)))
bwd <- rlx_grad(lr, wv)
w_true <- c(1.5, -2, 0.5)
n <- per_rank * world
x_all <- ((0:(n * 3 - 1)) * 37) %% 23 / 11.5 - 1
y_all <- as.vector(matrix(x_all, ncol = 3, byrow = TRUE) %*% w_true)
uris <- list(X = blob("X.bin", x_all), y = blob("y.bin", y_all), w = blob("w0.bin", c(0, 0, 0)))
spec <- function(r) list(
  params = list(list(name = "w", uri = uris$w)),
  data = list(list(input = "X", uri = uris$X, elem = 3, shard_start = r * per_rank, shard_len = per_rank),
              list(input = "y", uri = uris$y, elem = 1, shard_start = r * per_rank, shard_len = per_rank)),
  batch = 4, lr_per_epoch = rep(0.2, 40), device = "cpu")
for (r in seq_len(world - 1)) g$ship_train(r, bwd, spec(r))
mine <- g$run_train(bwd, spec(0))
cat(sprintf("training loss: %.4f -> %.2e\n", mine$metrics$first_loss, mine$metrics$last_loss))
w0 <- mine$params$w
cat(sprintf("rank 0 w: %s (true: 1.5 -2 0.5)\n", paste(sprintf("%.3f", w0), collapse = " ")))
for (r in seq_len(world - 1)) {
  theirs <- g$recv(r, 1)
  cat(sprintf("rank %d w agrees: %s\n", r, tolower(all(abs(theirs - w0) < 1e-5))))
}
unlink(blobs)
