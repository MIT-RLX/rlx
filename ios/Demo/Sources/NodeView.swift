// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

import SwiftUI

/// Joins an RLX mesh as a worker rank.
///
/// Pair with the desktop coordinator:
/// ```
/// cargo run -p rlx-ffi --example node_coordinator -- --world 2 \
///     --peers <mac-ip>:29500,<phone-ip>:29500
/// ```
struct NodeView: View {
    @State private var rank = "1"
    @State private var world = "2"
    @State private var peers = ""
    @State private var device = "auto"
    @State private var useDiscovery = false
    @State private var mode: RlxNode.Mode = .infer
    @State private var status = "idle"
    @State private var failure: String?

    /// The node serves on a native thread; this only polls it.
    private let tick = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    var body: some View {
        NavigationView {
            Form {
                Section {
                    Text("Join an RLX mesh as a worker rank. Start the desktop coordinator first.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    HStack {
                        Text("Platform")
                        Spacer()
                        Text(RlxNode.platform).foregroundStyle(.secondary)
                    }
                }

                Section("Rank") {
                    field("Rank", "this node", $rank, numeric: true)
                    field("World", "total ranks", $world, numeric: true)
                }

                Section("Peers") {
                    Toggle("Find coordinator automatically", isOn: $useDiscovery)
                    if useDiscovery {
                        Text("UDP discovery needs the multicast entitlement. Without it, use an explicit peer list.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        field("Coordinator", "host:port", $peers)
                    }
                    field("Device", "auto | cpu | metal", $device)
                }

                Section("Job") {
                    Picker("Mode", selection: $mode) {
                        Text("Inference").tag(RlxNode.Mode.infer)
                        Text("Training").tag(RlxNode.Mode.train)
                    }
                    .pickerStyle(.segmented)
                    if mode == .train {
                        Text("A training rank cannot leave partway — the gradient reduce is a barrier, so backgrounding the app stalls every other rank.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section {
                    Button("Join mesh") { start() }
                    Button("Leave mesh", role: .destructive) { RlxNode.stop() }
                }

                Section("Status") {
                    Text(status).font(.system(.body, design: .monospaced))
                    if let failure {
                        Text(failure)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.red)
                    }
                }
            }
            .navigationTitle("RLX Node")
        }
        .navigationViewStyle(.stack)
        .onReceive(tick) { _ in status = RlxNode.status }
        .onAppear { applyLaunchArguments() }
    }

    /// A labelled row — a bare `TextField` shows only its value once filled,
    /// which leaves "1" and "2" sitting on screen with nothing to say what
    /// they are.
    @ViewBuilder
    private func field(
        _ label: String,
        _ hint: String,
        _ text: Binding<String>,
        numeric: Bool = false
    ) -> some View {
        HStack {
            Text(label).frame(width: 100, alignment: .leading)
            TextField(hint, text: text)
                .multilineTextAlignment(.trailing)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
                .keyboardType(numeric ? .numberPad : .default)
        }
    }

    /// Seed the form from launch arguments and optionally join immediately.
    ///
    /// `UserDefaults` picks up `-key value` launch arguments, so a scripted run
    /// needs no UI driving:
    ///
    /// ```sh
    /// xcrun simctl launch <sim> com.mit.rlx.nodedemo \
    ///     -rank 1 -world 2 -peers 127.0.0.1:29500 -autojoin YES
    /// ```
    private func applyLaunchArguments() {
        let d = UserDefaults.standard
        if let v = d.string(forKey: "rank") { rank = v }
        if let v = d.string(forKey: "world") { world = v }
        if let v = d.string(forKey: "peers") { peers = v }
        if let v = d.string(forKey: "device") { device = v }
        if d.string(forKey: "mode") == "train" { mode = .train }
        if d.bool(forKey: "discover") { useDiscovery = true }
        FileHandle.standardError.write(
            "rlx-demo: launch args rank=\(rank) world=\(world) peers=\(peers) mode=\(mode.rawValue) autojoin=\(d.bool(forKey: "autojoin"))\n"
                .data(using: .utf8)!)
        if d.bool(forKey: "autojoin") { start() }
    }

    private func start() {
        failure = nil
        guard let r = Int(rank.trimmingCharacters(in: .whitespaces)),
              let w = Int(world.trimmingCharacters(in: .whitespaces)) else {
            failure = "rank and world must be integers"
            return
        }
        // An empty peer list is what tells the native side to discover.
        let list = useDiscovery
            ? []
            : peers.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                   .filter { !$0.isEmpty }
        do {
            try RlxNode.start(rank: r, world: w, peers: list,
                              device: device.isEmpty ? "auto" : device,
                              mode: mode)
        } catch {
            failure = "\(error)"
        }
        status = RlxNode.status
        FileHandle.standardError.write(
            "rlx-demo: start -> status=\(status) failure=\(failure ?? "none")\n"
                .data(using: .utf8)!)
    }
}

#Preview { NodeView() }
