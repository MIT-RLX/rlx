// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

import SwiftUI

@main
struct RlxDemoApp: App {
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            NodeView()
        }
        .onChange(of: scenePhase) { phase in
            // iOS suspends a backgrounded process within seconds, and a
            // suspended rank stalls every peer waiting on it — the mesh has no
            // timeout that will rescue you.
            if phase == .background { RlxNode.stop() }
        }
    }
}
