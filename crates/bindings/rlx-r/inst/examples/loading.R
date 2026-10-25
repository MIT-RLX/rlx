# RLX — versatile ML compiler + runtime.
# Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
# SPDX-License-Identifier: MIT OR Apache-2.0

# Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
# into a compiled graph, and quantizing in memory.
#   Rscript loading.R

library(rlx)
w <- (0:63 %% 9) / 4 - 1

gguf <- file.path(tempdir(), "rlx-example-r.gguf")
writer <- GgufWriter$new()
invisible(writer$add_f32("proj.weight", c(8, 8), "F32", w))
invisible(writer$add_f32("proj.weight_q8", c(8, 8), "Q8_0", w))
invisible(writer$write(gguf, architecture = "demo", metadata = list(demo.layers = 1)))
f <- Gguf$open(gguf)
cat(sprintf("gguf architecture: %s\n", f$architecture()))
cat(sprintf("gguf tensors: %s\n", paste(f$tensor_names(), collapse = " ")))
exact <- f$tensor("proj.weight")
q8 <- f$tensor("proj.weight_q8")
cat(sprintf("gguf f32 exact: %s q8 max|Δ|: %.4f\n", tolower(identical(exact, w)), max(abs(exact - q8))))

g <- Graph$new(name = "proj")
x <- g$input("x", c(1, 8))
g$set_outputs(g$matmul(x, g$param("proj.weight", c(8, 8))))
c1 <- Session$new()$compile(g)
c1$set_param("proj.weight", exact)
y <- c1$run(list(x = rep(1, 8)))[[1]]
cat(sprintf("gguf -> graph: %s\n", paste(sprintf("%.2f", y[1:3]), collapse = " ")))

st <- file.path(tempdir(), "rlx-example-r.safetensors")
header <- '{"proj.weight":{"dtype":"F32","shape":[8,8],"data_offsets":[0,256]}}'
header <- paste0(header, strrep(" ", (8 - nchar(header) %% 8) %% 8))
con <- file(st, "wb")
writeBin(as.integer(c(nchar(header), 0)), con, size = 4, endian = "little")
writeBin(charToRaw(header), con)
writeBin(w, con, size = 4, endian = "little")
close(con)
loaded <- rlx_load_weights("safetensors", st)
cat(sprintf("safetensors: [proj.weight] exact: %s\n", tolower(identical(loaded[["proj.weight"]], w))))

packed <- rlx_quantize(w, "Q8_0")
cat(sprintf("quantize q8_0: %d -> %d bytes, round trip %d values\n", length(w) * 4L, length(packed),
            length(rlx_dequantize(packed, "Q8_0", count = length(w)))))
unlink(c(gguf, st))
cat(sprintf("argmax: %d\n", as.integer(rlx_argmax(c(0.1, 2, 0.3)))))
