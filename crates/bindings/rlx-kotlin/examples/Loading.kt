// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Loading: a GGUF file (written here, f32 and quantized) and a safetensors file
// into a compiled graph, and quantizing in memory.

package org.rlx.kt.examples

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import org.rlx.kt.Gguf
import org.rlx.kt.GgufWriter
import org.rlx.kt.Graph
import org.rlx.kt.RLX
import org.rlx.kt.Session

fun main() {
    val w = FloatArray(64) { (it % 9) / 4f - 1 }
    val gguf = File.createTempFile("rlx-example-kt", ".gguf")
    GgufWriter().use { writer ->
        writer.addF32("proj.weight", longArrayOf(8, 8), "F32", w)
        writer.addF32("proj.weight_q8", longArrayOf(8, 8), "Q8_0", w)
        writer.write(gguf.path, architecture = "demo", metadata = mapOf("demo.layers" to 1))
    }
    val exact: FloatArray
    Gguf.open(gguf.path).use { f ->
        println("gguf architecture: ${f.architecture()}")
        println("gguf tensors: ${f.tensorNames().joinToString(" ")}")
        exact = f.tensor("proj.weight")
        val q8 = f.tensor("proj.weight_q8")
        println("gguf f32 exact: ${exact.contentEquals(w)} q8 max|Δ|: %.4f".format(exact.indices.maxOf { abs(exact[it] - q8[it]) }))
    }
    val g = Graph("proj")
    val x = g.input("x", longArrayOf(1, 8))
    g.setOutputs(intArrayOf(g.matmul(x, g.param("proj.weight", longArrayOf(8, 8)))))
    val c = Session().compile(g)
    c.setParam("proj.weight", exact)
    val y = c.run(mapOf("x" to FloatArray(8) { 1f }))[0]
    println("gguf -> graph: %.2f %.2f %.2f".format(y[0], y[1], y[2]))

    val st = File.createTempFile("rlx-example-kt", ".safetensors")
    var header = "{\"proj.weight\":{\"dtype\":\"F32\",\"shape\":[8,8],\"data_offsets\":[0,${4 * w.size}]}}"
    while (header.length % 8 != 0) header += " "
    val buf = ByteBuffer.allocate(8 + header.length + 4 * w.size).order(ByteOrder.LITTLE_ENDIAN)
    buf.putLong(header.length.toLong()).put(header.toByteArray())
    w.forEach { buf.putFloat(it) }
    st.writeBytes(buf.array())
    val loaded = RLX.loadWeights("safetensors", st.path)
    println("safetensors: ${loaded.keys} exact: ${loaded["proj.weight"]!!.contentEquals(w)}")

    val packed = RLX.quantize(w, "Q8_0")
    println("quantize q8_0: ${4 * w.size} -> ${packed.size} bytes, round trip ${RLX.dequantize(packed, "Q8_0", w.size.toLong()).size} values")
    gguf.delete()
    st.delete()
    println("argmax: ${RLX.argmax(floatArrayOf(0.1f, 2f, 0.3f))}")
}
