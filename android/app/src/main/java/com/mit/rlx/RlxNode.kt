package com.mit.rlx

import android.content.Context
import android.net.wifi.WifiManager
import android.util.Log

/**
 * Android-side lifecycle around [RlxNative]'s distributed node.
 *
 * Two things the native layer cannot do for itself:
 *
 *  1. **Multicast lock.** Android drops multicast/broadcast frames at the Wi-Fi
 *     chipset to save power, so UDP peer discovery finds nothing until a
 *     [WifiManager.MulticastLock] is held. Static peer lists don't need it.
 *  2. **Process lifecycle.** A serving node holds a socket and burns battery;
 *     it must be stopped when the app leaves the foreground, or Android will
 *     eventually kill the process out from under the mesh.
 */
object RlxNode {
    /** What this node joins the mesh to do. */
    enum class Mode(val wire: String) {
        /** Serve a shipped inference stage; stoppable between activations. */
        INFER("infer"),

        /**
         * Join a data-parallel training run. **Not** stoppable partway: the
         * gradient reduce is a barrier, so leaving stalls every other rank.
         */
        TRAIN("train"),
    }

    private const val TAG = "RlxNode"
    private const val LOCK_TAG = "rlx-node-discovery"

    private var lock: WifiManager.MulticastLock? = null

    /**
     * Start a worker node. Pass [discovery] = true when [peers] is empty and
     * the node should find the coordinator by UDP broadcast.
     *
     * Returns the native status string, or throws on a bad configuration.
     */
    @Synchronized
    fun start(
        context: Context,
        rank: Int,
        world: Int,
        peers: String,
        device: String = "auto",
        discovery: Boolean = false,
        mode: Mode = Mode.INFER,
    ): String {
        if (discovery) acquireMulticastLock(context)
        return try {
            RlxNative.nodeStart(rank, world, peers, device, mode.wire)
        } catch (e: Throwable) {
            releaseMulticastLock()
            throw e
        }
    }

    /** Current node state, as reported by the native serving thread. */
    fun status(): String = RlxNative.nodeStatus()

    /**
     * Ask the node to leave the mesh and drop the multicast lock.
     *
     * Call from `onStop`. The stop is cooperative — a node parked in `recv`
     * exits when its peer sends or the link drops, so `status()` may report
     * `running` briefly afterwards.
     */
    @Synchronized
    fun stop() {
        try {
            RlxNative.nodeStop()
        } catch (e: Throwable) {
            Log.w(TAG, "nodeStop failed", e)
        } finally {
            releaseMulticastLock()
        }
    }

    private fun acquireMulticastLock(context: Context) {
        if (lock != null) return
        val wifi = context.applicationContext
            .getSystemService(Context.WIFI_SERVICE) as? WifiManager ?: run {
            Log.w(TAG, "no WifiManager; UDP discovery may not receive peers")
            return
        }
        lock = wifi.createMulticastLock(LOCK_TAG).apply {
            setReferenceCounted(false)
            acquire()
        }
    }

    private fun releaseMulticastLock() {
        lock?.let { if (it.isHeld) it.release() }
        lock = null
    }
}
