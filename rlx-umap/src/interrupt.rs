// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! Ctrl-C graceful cancellation (fast-umap compatible).

use crossbeam_channel::Receiver;

/// Channel that receives `()` when the user presses Ctrl-C.
pub fn install_ctrlc_handler() -> Receiver<()> {
    let (exit_tx, exit_rx) = crossbeam_channel::unbounded();
    let _ = ctrlc::set_handler(move || {
        let _ = exit_tx.send(());
    });
    exit_rx
}
