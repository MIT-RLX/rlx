package com.mit.rlx

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * Joins an RLX mesh as a worker rank.
 *
 * Pair with the desktop coordinator:
 * ```
 * cargo run -p rlx-ffi --example node_coordinator -- --world 2
 * ```
 *
 * The node serves on a native thread, so this screen only starts it and polls
 * for status. Work happens whether or not the UI is showing — which is exactly
 * why [onStop] tears the node down.
 */
class NodeActivity : AppCompatActivity() {

    companion object {
        const val TAG = "RlxNode"
        const val EXTRA_RANK = "rank"
        const val EXTRA_WORLD = "world"
        const val EXTRA_PEERS = "peers"
        const val EXTRA_DEVICE = "device"
        const val EXTRA_DISCOVER = "discover"
        const val EXTRA_MODE = "mode"
        const val EXTRA_AUTOJOIN = "autojoin"

        /** Copy the node extras from [src] onto an intent for this activity. */
        fun intentFrom(context: android.content.Context, src: Intent): Intent =
            Intent(context, NodeActivity::class.java).apply {
                src.extras?.let { putExtras(it) }
            }
    }

    private val poll = Handler(Looper.getMainLooper())
    private lateinit var statusText: TextView
    private var polling = false
    private var lastLogged: String? = null

    private val tick = object : Runnable {
        override fun run() {
            val s = RlxNode.status()
            if (s != lastLogged) {
                lastLogged = s
                Log.i(TAG, "node status: $s")
            }
            statusText.text = getString(R.string.node_status, s)
            if (polling) poll.postDelayed(this, 500)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_node)

        val rank = findViewById<EditText>(R.id.rankInput)
        val world = findViewById<EditText>(R.id.worldInput)
        val peers = findViewById<EditText>(R.id.peersInput)
        val device = findViewById<EditText>(R.id.deviceInput)
        val discover = findViewById<CheckBox>(R.id.discoverCheck)
        val trainCheck = findViewById<CheckBox>(R.id.trainCheck)
        val startButton = findViewById<Button>(R.id.startButton)
        val stopButton = findViewById<Button>(R.id.stopButton)
        statusText = findViewById(R.id.statusText)

        // Peers and discovery are alternatives: an empty peer list is what
        // tells the native side to look for the coordinator by UDP.
        discover.setOnCheckedChangeListener { _, checked ->
            peers.isEnabled = !checked
            if (checked) peers.setText("")
        }

        // Scripted runs: prefill (and optionally join) from intent extras, so a
        // headless emulator test needs no UI driving. MainActivity forwards
        // these, which keeps this activity unexported.
        //
        //   adb shell am start -n com.mit.rlx.demo/com.mit.rlx.MainActivity \
        //     --ei rank 1 --ei world 2 --es peers 10.0.2.2:29500 --ez autojoin true
        //
        // From an emulator the host is 10.0.2.2 — its own loopback is not yours.
        intent?.let { i ->
            if (i.hasExtra(EXTRA_RANK)) rank.setText(i.getIntExtra(EXTRA_RANK, 1).toString())
            if (i.hasExtra(EXTRA_WORLD)) world.setText(i.getIntExtra(EXTRA_WORLD, 2).toString())
            i.getStringExtra(EXTRA_PEERS)?.let { peers.setText(it) }
            i.getStringExtra(EXTRA_DEVICE)?.let { device.setText(it) }
            if (i.getBooleanExtra(EXTRA_DISCOVER, false)) discover.isChecked = true
            if (i.getStringExtra(EXTRA_MODE) == "train") trainCheck.isChecked = true
        }

        startButton.setOnClickListener {
            try {
                val useDiscovery = discover.isChecked
                RlxNode.start(
                    context = this,
                    rank = rank.text.toString().trim().toIntOrNull()
                        ?: error("rank must be an integer"),
                    world = world.text.toString().trim().toIntOrNull()
                        ?: error("world must be an integer"),
                    peers = if (useDiscovery) "" else peers.text.toString().trim(),
                    device = device.text.toString().trim().ifEmpty { "auto" },
                    discovery = useDiscovery,
                    mode = if (trainCheck.isChecked) RlxNode.Mode.TRAIN else RlxNode.Mode.INFER,
                )
                startPolling()
            } catch (e: Throwable) {
                // Includes the native RuntimeException carrying the Rust-side
                // reason (bad rank/world, peer count vs world, …).
                val msg = e.message ?: e.toString()
                Log.w(TAG, "node start failed: $msg")
                statusText.text = getString(R.string.node_status, msg)
            }
        }

        if (intent?.getBooleanExtra(EXTRA_AUTOJOIN, false) == true) {
            startButton.performClick()
        }

        stopButton.setOnClickListener {
            RlxNode.stop()
            // Keep polling: the stop is cooperative, so the terminal status
            // ("ok: …" / "error: …") lands a moment later.
            startPolling()
        }
    }

    private fun startPolling() {
        if (!polling) {
            polling = true
            poll.post(tick)
        }
    }

    override fun onStop() {
        super.onStop()
        polling = false
        poll.removeCallbacks(tick)
        // Android suspends a backgrounded process, and a suspended rank stalls
        // every peer waiting on it — the mesh has no timeout that rescues you.
        RlxNode.stop()
    }
}
