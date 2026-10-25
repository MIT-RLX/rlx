# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# What the generated R/generated.R stands on: method dispatch for the handle
# classes, the class generators, errors, conversions, and the two rows whose
# shape is not a plain call (Optimizer$step, Trainer$run).

#' "No node", for an optional node argument.
NONE <- 4294967295

.methods <- new.env(parent = emptyenv())

.wrap <- function(p, cls) {
  class(p) <- c(paste0("rlx_", cls), "rlx_handle")
  p
}

#' @export
`$.rlx_handle` <- function(x, name) {
  cls <- class(x)[1]
  if (name == "close") {
    free <- get(paste0("R_rlx_", tolower(sub("^rlx_", "", cls)), "_free"), envir = asNamespace("rlx"))
    return(function() invisible(.Call(free, x)))
  }
  f <- .methods[[cls]][[name]]
  if (is.null(f)) stop(sprintf("%s has no method '%s'", sub("^rlx_", "", cls), name), call. = FALSE)
  function(...) f(x, ...)
}

#' @export
print.rlx_handle <- function(x, ...) {
  cat(sprintf("<rlx %s>\n", sub("^rlx_", "", class(x)[1])))
  invisible(x)
}

.generator <- function(name, members) structure(members, class = "rlx_generator", rlx_name = name)

#' @export
`$.rlx_generator` <- function(x, name) {
  f <- unclass(x)[[name]]
  if (is.null(f)) stop(sprintf("%s has no constructor or static '%s'", attr(x, "rlx_name"), name), call. = FALSE)
  f
}

#' @export
print.rlx_generator <- function(x, ...) {
  cat(sprintf("<rlx class %s: %s>\n", attr(x, "rlx_name"), paste(names(unclass(x)), collapse = ", ")))
  invisible(x)
}

.kinds <- c("type", "range", "runtime", "panic", "not_found")

.check <- function(rc) {
  if (rc != 0L) {
    kind <- if (rc < 0L && -rc <= length(.kinds)) .kinds[[-rc]] else "other"
    msg <- .Call(R_support_last_error)
    stop(structure(class = c(paste0("rlx_", kind), "rlx_error", "error", "condition"),
                   list(message = msg, call = NULL, kind = kind)))
  }
}

# Options to JSON: named lists are objects, vectors of length > 1 and unnamed
# lists arrays (wrap a single value in list() to send an array of one).
.json <- function(v) as.character(jsonlite::toJSON(v, auto_unbox = TRUE, null = "null", digits = NA))

.decode <- function(s) {
  if (is.null(s)) return(NULL)
  jsonlite::fromJSON(s, simplifyVector = TRUE, simplifyDataFrame = FALSE, simplifyMatrix = FALSE)
}

.rows <- function(v, w, splat) {
  if (is.matrix(v)) {
    if (ncol(v) == w) return(as.vector(t(v)))
    if (splat && ncol(v) == 1) return(rep(as.vector(v), each = w))
    stop(sprintf("rows need %d values each, got a %dx%d matrix", w, nrow(v), ncol(v)), call. = FALSE)
  }
  unlist(lapply(seq_along(v), function(i) {
    r <- v[[i]]
    if (length(r) == w) r
    else if (splat && length(r) == 1) rep(r, w)
    else stop(sprintf("row %d needs %d values, got %d", i, w, length(r)), call. = FALSE)
  }))
}

.entries <- function(m) {
  if (length(m) && is.null(names(m))) stop("named tensors need names", call. = FALSE)
  m[order(names(m))]
}

.inputs <- function(m) {
  m <- .entries(m)
  list(names = names(m), data = unname(m), lens = vapply(m, length, numeric(1), USE.NAMES = FALSE))
}

.typed_inputs <- function(m) {
  m <- .entries(m)
  list(names = names(m), data = lapply(m, `[[`, "data"), lens = vapply(m, function(t) length(t$data), numeric(1)),
       dtypes = vapply(m, `[[`, character(1), "dtype", USE.NAMES = FALSE))
}

#' An R array as the flat row-major vector rlx takes (R is column-major).
rlx_flat <- function(a) {
  d <- dim(a)
  if (is.null(d)) return(as.vector(a))
  as.vector(aperm(a, rev(seq_along(d))))
}

#' A flat row-major vector rlx returned, as an R array of `shape`.
rlx_unflat <- function(v, shape) {
  if (length(shape) == 1) return(v)
  aperm(array(v, dim = rev(shape)), rev(seq_along(shape)))
}

# ── the manual rows ──────────────────────────────────────────

# Optimizer$step(params, grads): R lists are values, so this returns the
# updated params (params <- opt$step(params, grads)).
.optimizer_step <- function(self, params, grads) {
  params <- .entries(params)
  missing <- setdiff(names(params), names(grads))
  if (length(missing)) stop(sprintf("Optimizer step: no gradient for '%s'", missing[[1]]), call. = FALSE)
  r <- .Call(R_support_optimizer_step, self, names(params), unname(params), unname(grads[names(params)]))
  .check(r[[1]])
  out <- r[[2]]
  names(out) <- names(params)
  out
}

# Trainer$run(steps, batch, lr = NULL, on_step = NULL, every = 1): batch(step)
# gives the inputs (NULL ends early), lr(step) the rate, on_step(step, loss)
# FALSE stops.
.trainer_run <- function(self, steps, batch, lr = NULL, on_step = NULL, every = 1) {
  every <- max(every, 1)
  start <- self$steps()
  losses <- numeric(0)
  stopped <- FALSE
  t0 <- proc.time()[["elapsed"]]
  for (i in seq_len(steps) - 1) {
    s <- start + i
    b <- batch(s)
    if (is.null(b)) break
    if (!is.null(lr)) self$set_lr(lr(s))
    loss <- self$step(b)
    losses <- c(losses, loss)
    if (!is.null(on_step) && (i + 1) %% every == 0 && !isTRUE(on_step(s, loss))) {
      stopped <- TRUE
      break
    }
  }
  secs <- max(proc.time()[["elapsed"]] - t0, 1e-9)
  list(steps = length(losses), stopped = stopped,
       first_loss = if (length(losses)) losses[[1]] else NULL,
       last_loss = if (length(losses)) losses[[length(losses)]] else NULL,
       mean_loss = if (length(losses)) mean(losses) else NULL,
       steps_per_second = length(losses) / secs, grad_norm = self$grad_norm())
}
