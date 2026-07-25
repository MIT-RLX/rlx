package com.mit.rlx

import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * Loads [librlx_jni.so][System.loadLibrary] and runs RLX graphs on device:
 * a tiny GELU demo and an embedded MNIST MLP (784→32→10).
 */
class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val backendText = findViewById<TextView>(R.id.backendText)
        val outputText = findViewById<TextView>(R.id.outputText)
        val runButton = findViewById<Button>(R.id.runButton)
        val mnistButton = findViewById<Button>(R.id.mnistButton)

        runButton.setOnClickListener {
            try {
                backendText.text = getString(R.string.backend_label, RlxNative.backendName())
                val out = RlxNative.runInference()
                outputText.text = out.joinToString(prefix = "[", postfix = "]") { "%.4f".format(it) }
            } catch (e: RuntimeException) {
                outputText.text = e.message ?: e.toString()
            }
        }

        mnistButton.setOnClickListener {
            try {
                backendText.text = getString(R.string.backend_label, RlxNative.backendName())
                val pred = RlxNative.mnistPredict()
                val expected = RlxNative.mnistExpectedLabel()
                val logits = RlxNative.runMnist()
                val logitStr = logits.joinToString(prefix = "[", postfix = "]") { "%.2f".format(it) }
                outputText.text = getString(R.string.mnist_result, pred, expected, logitStr)
            } catch (e: RuntimeException) {
                outputText.text = e.message ?: e.toString()
            }
        }
    }
}
