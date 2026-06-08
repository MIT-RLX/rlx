// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

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
