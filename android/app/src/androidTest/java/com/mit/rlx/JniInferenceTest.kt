package com.mit.rlx

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class JniInferenceTest {

    @Test
    fun runInference_returnsTwoFiniteOutputs() {
        val out = RlxNative.runInference()
        assertEquals(2, out.size)
        assertTrue(out[0].isFinite())
        assertTrue(out[1].isFinite())
    }

    @Test
    fun backendName_isNonEmpty() {
        val name = RlxNative.backendName()
        assertTrue(name.isNotBlank())
    }

    @Test
    fun runInference_isDeterministic() {
        val a = RlxNative.runInference()
        val b = RlxNative.runInference()
        assertEquals(a.size, b.size)
        for (i in a.indices) {
            assertEquals(a[i], b[i], 1e-5f)
        }
    }

    @Test
    fun runMnist_predictsEmbeddedDigit() {
        val expected = RlxNative.mnistExpectedLabel()
        assertTrue(expected in 0..9)
        val pred = RlxNative.mnistPredict()
        assertEquals(expected, pred)
        val logits = RlxNative.runMnist()
        assertEquals(10, logits.size)
        assertTrue(logits.all { it.isFinite() })
        val argmax = logits.indices.maxBy { logits[it] }
        assertEquals(pred, argmax)
    }
}
