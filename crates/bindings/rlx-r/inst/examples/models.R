# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Models: any provider's model, opened by path or id and run by name. A graph
# saved as an .rlxp package runs anywhere; a plugin's models (a downstream
# collection's LLMs, or the reference plugin's demo:count) load through
# RLX_MODEL_PLUGINS.
#   RLX_MODEL_PLUGINS=…/librlx_model_plugin.so Rscript models.R

library(rlx)

g <- Graph$new(name = "affine")
x <- g$input("x", c(1, 2))
g$set_outputs(g$matmul(x, g$param("w", c(2, 2))))
path <- file.path(tempdir(), "rlx-example-r.rlxp")
rlx_save_model(g, path, list(w = c(2, 0, 0, 3)),
               model = list(task = "graph", family = "affine", outputs = list("y")))

m <- Model$open(path, options = list(device = "cpu"))
info <- m$info()
cat(sprintf("model: %s (%s) on %s\n", info$family, info$task, info$device))
y <- m$run(list(x = c(1, 1)))[[1]]
cat(sprintf("model run: %.3f %.3f\n", y[[1]], y[[2]]))
unlink(path)

lm <- Model$open("demo:count")
lm_info <- lm$info()
cat(sprintf("plugin model: %s (%s)\n", lm_info$family, lm_info$task))
cat(sprintf("model text: %s\n", lm$generate_text("c", options = list(max_tokens = 4))))
cat(sprintf("model ids: %s\n", paste(lm$generate(2, max_tokens = 3), collapse = " ")))

# By task: a chat, speech out and back in, embeddings.
cat(sprintf("chat: %s\n", lm$chat(list(list(role = "user", content = "c")), options = list(max_tokens = 4))))
speech <- Model$open("demo:voice")$synthesize("hello")
cat(sprintf("voice: %d samples at %d Hz\n", length(speech$audio), as.integer(speech$sample_rate)))
cat(sprintf("heard: %s\n", Model$open("demo:speech")$transcribe(speech$audio, sample_rate = speech$sample_rate)))
rows <- Model$open("demo:embed")$embed(c("hello", "sky"))
cat(sprintf("embed: %s\n", paste(as.integer(unlist(rows)), collapse = " ")))
