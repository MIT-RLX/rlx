// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

package org.rlx.kt

/**
 * Loads librlx_jni (which links librlx_capi): `-Drlx.jni.lib=<path>` when set,
 * else `System.loadLibrary("rlx_jni")` (Android: the APK's jniLibs; JVM:
 * `java.library.path`).
 */
internal object Loader {
    @Volatile private var loaded = false

    fun load() {
        if (loaded) return
        synchronized(this) {
            if (loaded) return
            val path = System.getProperty("rlx.jni.lib")
            if (path != null) System.load(path) else System.loadLibrary("rlx_jni")
            loaded = true
        }
    }
}

/** The hand-written JNI half (`rlx_jni_support.c`). */
internal object NativeSupport {
    init { Loader.load() }

    @JvmStatic external fun lastError(): String
    @JvmStatic external fun takeString(p: Long): String
    @JvmStatic external fun takeFloats(p: Long, n: Long): FloatArray
    @JvmStatic external fun takeBytes(p: Long, n: Long): ByteArray
    @JvmStatic external fun takeInts(p: Long, n: Long): IntArray
    @JvmStatic external fun takeOutputs(p: Long): Array<FloatArray>
    @JvmStatic external fun takeTypedOutputs(p: Long): Array<Any?>
    @JvmStatic external fun takeTensors(p: Long): Array<Any?>
    @JvmStatic external fun optimizerStep(h: Long, names: Array<Any?>, params: Array<Any?>, grads: Array<Any?>): Int
}

/** A failed rlx call, with its `RLX_ERR_*` kind. */
class RlxException(val kind: Kind, message: String) : RuntimeException(message) {
    enum class Kind { TYPE, RANGE, RUNTIME, PANIC, NOT_FOUND, OTHER }
}

/** A tensor of any dtype as raw bytes (`"f32"`, `"i64"`, …). */
class TypedTensor(val bytes: ByteArray, val dtype: String)

/** The base of every handle class: `close()` frees now, collection frees otherwise. */
abstract class Handle internal constructor(h: Long, private val free: (Long) -> Unit) : AutoCloseable {
    internal var h: Long = h
        private set

    init {
        if (h == 0L) throw RlxException(RlxException.Kind.RUNTIME, "rlx returned NULL")
    }

    override fun close() {
        if (h != 0L) {
            free(h)
            h = 0L
        }
    }

    @Suppress("removal", "DEPRECATION")
    protected fun finalize() = close()
}

internal object S {
    fun check(rc: Int) {
        if (rc == 0) return
        val kind = when (rc) {
            -1 -> RlxException.Kind.TYPE
            -2 -> RlxException.Kind.RANGE
            -3 -> RlxException.Kind.RUNTIME
            -4 -> RlxException.Kind.PANIC
            -5 -> RlxException.Kind.NOT_FOUND
            else -> RlxException.Kind.OTHER
        }
        throw RlxException(kind, NativeSupport.lastError())
    }

    fun takeString(p: Long): String = NativeSupport.takeString(p)

    fun rows(rows: List<LongArray>, w: Int, splat: Boolean): LongArray {
        val out = ArrayList<Long>()
        rows.forEachIndexed { i, r ->
            when {
                r.size == w -> r.forEach { out.add(it) }
                splat && r.size == 1 -> repeat(w) { out.add(r[0]) }
                else -> throw RlxException(RlxException.Kind.TYPE, "row $i needs $w values, got ${r.size}")
            }
        }
        return out.toLongArray()
    }

    fun frows(rows: List<FloatArray>, w: Int): FloatArray {
        rows.forEachIndexed { i, r ->
            if (r.size != w) throw RlxException(RlxException.Kind.TYPE, "row $i needs $w values")
        }
        return rows.flatMap { it.asList() }.toFloatArray()
    }

    class CInputs(val names: Array<Any?>, val data: Array<Any?>, val lens: LongArray, val dtypes: Array<Any?>?, val n: Long)

    fun inputs(m: Map<String, FloatArray>): CInputs {
        val keys = m.keys.toList()
        return CInputs(keys.toTypedArray(), keys.map { m[it] }.toTypedArray(), keys.map { m[it]!!.size.toLong() }.toLongArray(), null, keys.size.toLong())
    }

    fun typedInputs(m: Map<String, TypedTensor>): CInputs {
        val keys = m.keys.toList()
        return CInputs(keys.toTypedArray(), keys.map { m[it]!!.bytes }.toTypedArray(),
            keys.map { m[it]!!.bytes.size.toLong() }.toLongArray(), keys.map { m[it]!!.dtype }.toTypedArray(), keys.size.toLong())
    }

    @Suppress("UNCHECKED_CAST")
    fun obj(v: Any?): Map<String, Any?> = v as Map<String, Any?>

    @Suppress("UNCHECKED_CAST")
    fun rowsOf(v: Any?): List<Map<String, Any?>> = v as List<Map<String, Any?>>

    @Suppress("UNCHECKED_CAST")
    fun strings(v: Any?): List<String> = v as List<String>

    @Suppress("UNCHECKED_CAST")
    fun shapes(v: Any?): List<LongArray> = (v as List<List<Number>>).map { r -> r.map { it.toLong() }.toLongArray() }

    fun typedOutputs(a: Array<Any?>): List<TypedTensor> =
        (a.indices step 2).map { TypedTensor(a[it] as ByteArray, a[it + 1] as String) }

    fun tensors(a: Array<Any?>): Map<String, FloatArray> =
        (a.indices step 2).associate { (a[it] as String) to (a[it + 1] as FloatArray) }
}

/** JSON for options in and reports out (the JVM has no JSON in its standard library). */
object Json {
    fun encode(v: Any?): String = StringBuilder().also { write(it, v) }.toString()

    private fun write(sb: StringBuilder, v: Any?) {
        when (v) {
            null -> sb.append("null")
            is Boolean, is Int, is Long, is Short, is Byte -> sb.append(v)
            is Number -> v.toDouble().let { if (it.isFinite()) sb.append(it) else sb.append("null") }
            is CharSequence -> {
                sb.append('"')
                for (c in v) when {
                    c == '"' -> sb.append("\\\"")
                    c == '\\' -> sb.append("\\\\")
                    c == '\n' -> sb.append("\\n")
                    c < ' ' -> sb.append(String.format("\\u%04x", c.code))
                    else -> sb.append(c)
                }
                sb.append('"')
            }
            is Map<*, *> -> {
                sb.append('{')
                v.entries.forEachIndexed { i, e -> if (i > 0) sb.append(','); write(sb, e.key.toString()); sb.append(':'); write(sb, e.value) }
                sb.append('}')
            }
            is Iterable<*> -> { sb.append('['); v.forEachIndexed { i, x -> if (i > 0) sb.append(','); write(sb, x) }; sb.append(']') }
            is Array<*> -> write(sb, v.asList())
            is FloatArray -> write(sb, v.asList())
            is DoubleArray -> write(sb, v.asList())
            is IntArray -> write(sb, v.asList())
            is LongArray -> write(sb, v.asList())
            else -> throw RlxException(RlxException.Kind.TYPE, "options are not JSON-encodable: ${v::class}")
        }
    }

    /** JSON text → Map / List / Long / Double / String / Boolean / null. */
    fun parse(s: String): Any? = Parser(s).value()

    private class Parser(val s: String) {
        var i = 0
        fun skip() { while (i < s.length && s[i].isWhitespace()) i++ }
        fun value(): Any? {
            skip()
            return when (val c = s[i]) {
                '{' -> { i++; val m = LinkedHashMap<String, Any?>(); skip(); if (s[i] == '}') { i++; return m }
                    while (true) { skip(); val k = string(); skip(); i++; m[k] = value(); skip(); if (s[i++] == '}') return m } }
                '[' -> { i++; val l = ArrayList<Any?>(); skip(); if (s[i] == ']') { i++; return l }
                    while (true) { l.add(value()); skip(); if (s[i++] == ']') return l } }
                '"' -> string()
                't' -> { i += 4; true }
                'f' -> { i += 5; false }
                'n' -> { i += 4; null }
                else -> { val start = i; while (i < s.length && (s[i].isDigit() || s[i] in "+-.eE")) i++
                    val t = s.substring(start, i); if (t.any { it in ".eE" }) t.toDouble() else t.toLong().also { _ -> c } }
            }
        }
        fun string(): String {
            val sb = StringBuilder(); i++
            while (s[i] != '"') {
                val c = s[i++]
                if (c == '\\') when (val e = s[i++]) {
                    'n' -> sb.append('\n'); 't' -> sb.append('\t'); 'r' -> sb.append('\r'); 'b' -> sb.append('\b')
                    'u' -> { sb.append(s.substring(i, i + 4).toInt(16).toChar()); i += 4 }
                    else -> sb.append(e)
                } else sb.append(c)
            }
            i++
            return sb.toString()
        }
    }
}

// ── the manual rows ─────────────────────────────────────────

/** `Optimizer.step`. */
interface OptimizerRows {
    /** Update each `params[name]` in place from `grads[name]`. */
    fun step(params: Map<String, FloatArray>, grads: Map<String, FloatArray>) {
        val names = params.keys.toList()
        for (n in names) {
            val g = grads[n]
            if (g == null || g.size != params[n]!!.size) throw RlxException(RlxException.Kind.TYPE, "Optimizer.step: no matching gradient for '$n'")
        }
        S.check(NativeSupport.optimizerStep((this as Handle).h, names.toTypedArray(),
            names.map { params[it] }.toTypedArray(), names.map { grads[it] }.toTypedArray()))
    }
}

/** `Trainer.run`. */
interface TrainerRows {
    fun steps(): Long
    fun step(batch: Map<String, FloatArray>): Float
    fun setLr(lr: Float)
    fun gradNorm(): Float?

    /** Train `steps` steps, pulling inputs from `batch(step)` (null: data exhausted). */
    fun run(
        steps: Int,
        lr: ((Long) -> Float)? = null,
        onStep: ((Long, Float) -> Boolean)? = null,
        every: Int = 1,
        batch: (Long) -> Map<String, FloatArray>?,
    ): Map<String, Any?> {
        val start = steps()
        val losses = ArrayList<Float>()
        var stopped = false
        val t0 = System.nanoTime()
        for (i in 0 until steps) {
            val s = start + i
            val b = batch(s) ?: break
            lr?.let { setLr(it(s)) }
            val loss = step(b)
            losses.add(loss)
            if (onStep != null && (i + 1) % maxOf(every, 1) == 0 && !onStep(s, loss)) { stopped = true; break }
        }
        return linkedMapOf(
            "steps" to losses.size, "stopped" to stopped,
            "first_loss" to losses.firstOrNull(), "last_loss" to losses.lastOrNull(),
            "mean_loss" to if (losses.isEmpty()) null else losses.average(),
            "steps_per_second" to losses.size / maxOf((System.nanoTime() - t0) / 1e9, 1e-9),
            "grad_norm" to gradNorm(),
        )
    }
}
