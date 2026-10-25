// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Models: any provider's model, opened by path or id and run by name. A graph
// saved as an .rlxp package runs anywhere; a plugin's models (a downstream
// collection's LLMs, or the reference plugin's demo:count) load through
// RLX_MODEL_PLUGINS.

package org.rlx.kt.examples

import java.io.File
import org.rlx.kt.Graph
import org.rlx.kt.Model
import org.rlx.kt.RLX

fun main() {
    val g = Graph("affine")
    val x = g.input("x", longArrayOf(1, 2))
    g.setOutputs(intArrayOf(g.matmul(x, g.param("w", longArrayOf(2, 2)))))
    val file = File(System.getProperty("java.io.tmpdir"), "rlx-example-kotlin.rlxp")
    RLX.saveModel(g, file.path, mapOf("w" to floatArrayOf(2f, 0f, 0f, 3f)),
        mapOf("task" to "graph", "family" to "affine", "outputs" to listOf("y")))

    Model.open(file.path, mapOf("device" to "cpu")).use { m ->
        val info = m.info()
        println("model: ${info["family"]} (${info["task"]}) on ${info["device"]}")
        val y = m.run(mapOf("x" to floatArrayOf(1f, 1f)))[0]
        println("model run: %.3f %.3f".format(y[0], y[1]))
    }
    file.delete()

    Model.open("demo:count").use { lm ->
        val info = lm.info()
        println("plugin model: ${info["family"]} (${info["task"]})")
        println("model text: ${lm.generateText("c", mapOf("max_tokens" to 4))}")
        println("model ids: ${lm.generate(intArrayOf(2), maxTokens = 3).joinToString(" ")}")

        // By task: a chat, speech out and back in, embeddings.
        println("chat: ${lm.chat(listOf(mapOf("role" to "user", "content" to "c")), mapOf("max_tokens" to 4))}")
    }
    val speech = Model.open("demo:voice").use { it.synthesize("hello") }
    println("voice: ${speech.audio.size} samples at ${speech.sampleRate} Hz")
    Model.open("demo:speech").use { println("heard: ${it.transcribe(speech.audio, speech.sampleRate)}") }
    Model.open("demo:embed").use { e ->
        val rows = e.embed(listOf("hello", "sky")) as List<*>
        println("embed: ${rows.flatMap { it as List<*> }.joinToString(" ") { (it as Number).toInt().toString() }}")
    }
}
