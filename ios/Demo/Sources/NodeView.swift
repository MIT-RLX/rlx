// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

import SwiftUI

/// Joins an RLX mesh as a worker rank.
///
/// One view, four Apple OSes: iOS, tvOS, watchOS and visionOS all build this
/// file. The `#if os(...)` seams below are the places where SwiftUI genuinely
/// differs — text entry on a watch, picker styles on a TV — not styling.
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
    /// `onAppear` can fire more than once; launch arguments apply once.
    @State private var launched = false
    /// Bonjour record for this device (`_rlx-node._tcp`) while the app is open.
    private static let advertiser = RlxAdvertiser()

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
                        Text("Any model").tag(RlxNode.Mode.model)
                    }
                    // `.segmented` is iOS/macOS/visionOS only. A TV picks with
                    // focus and a watch with the crown, and the default style
                    // is the native shape on both.
                    #if !os(tvOS) && !os(watchOS)
                    .pickerStyle(.segmented)
                    #endif
                    if mode != .infer {
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
        // visionOS puts a NavigationView in its own window; forcing the stack
        // style there is both unavailable and wrong.
        #if !os(visionOS)
        .navigationViewStyle(.stack)
        #endif
        .onReceive(tick) { _ in
            let now = RlxNode.status
            if now != status {
                // Status changes on stderr, so a launcher streaming the console
                // (`devicectl … --console`) sees the rank's progress.
                FileHandle.standardError.write("rlx-demo: status \(now)\n".data(using: .utf8)!)
            }
            status = now
            // `-exitwhendone YES`: a launched rank ends with its job, so the
            // launcher sees it finish (and how).
            if UserDefaults.standard.bool(forKey: "exitwhendone"),
               now.hasPrefix("ok:") || now.hasPrefix("error:") {
                exit(now.hasPrefix("ok:") ? 0 : 1)
            }
        }
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
        #if os(watchOS)
        // A watch row is too narrow for a label beside its field, and watchOS
        // has no `keyboardType` — text entry is dictation/scribble, and the
        // numeric hint has nowhere to land.
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            TextField(hint, text: text)
        }
        #else
        HStack {
            Text(label).frame(width: 100, alignment: .leading)
            TextField(hint, text: text)
                .multilineTextAlignment(.trailing)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
                .keyboardType(numeric ? .numberPad : .default)
        }
        #endif
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
        guard !launched else { return }
        launched = true
        let d = UserDefaults.standard
        // `-logfile YES`: stderr — this view's lines and the Rust node's —
        // goes to Documents/rlx-node.log, headed by the launcher's `-jobid`.
        // A launcher reads it with `devicectl device copy from`, so the app
        // can stay open after a job: `--console` would tie the app's life to
        // the devicectl session (it forwards every signal to the app).
        if d.bool(forKey: "logfile"),
           let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let path = dir.appendingPathComponent("rlx-node.log").path
            FileManager.default.createFile(
                atPath: path,
                contents: "rlx-demo: job \(d.string(forKey: "jobid") ?? "-")\n".data(using: .utf8))
            if freopen(path, "a", stderr) != nil {
                setvbuf(stderr, nil, _IONBF, 0)
            }
        }
        // Say on the LAN that this device can be a rank — `cargo rlx nodes`
        // lists it, and a cluster file host `addr = "mdns"` finds it by name.
        // iOS 16+ reports a generic device name without a special
        // entitlement, so `-nodename` sets the one a cluster file uses.
        let nodeName = d.string(forKey: "nodename") ?? Self.deviceName()
        let advertisedRank = Int32(d.string(forKey: "rank") ?? rank) ?? 1
        Self.advertiser.onChange = { st in
            FileHandle.standardError.write("rlx-demo: bonjour \(st)\n".data(using: .utf8)!)
        }
        Self.advertiser.start(name: nodeName, port: 29500 + advertisedRank)
        FileHandle.standardError.write(
            "rlx-demo: advertising \(nodeName) as _rlx-node._tcp \(RlxNode.advert)\n".data(using: .utf8)!)
        if let v = d.string(forKey: "rank") { rank = v }
        if let v = d.string(forKey: "world") { world = v }
        if let v = d.string(forKey: "peers") { peers = v }
        if let v = d.string(forKey: "device") { device = v }
        if let m = d.string(forKey: "mode").flatMap(RlxNode.Mode.init(rawValue:)) { mode = m }
        if d.bool(forKey: "discover") { useDiscovery = true }
        // The other ranks dial this address, so say it where a launcher sees it.
        FileHandle.standardError.write(
            "rlx-demo: lan address \(Self.lanAddress() ?? "none")\n".data(using: .utf8)!)
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

extension NodeView {
    static func deviceName() -> String {
        #if os(watchOS)
        return "watch"
        #else
        return UIDevice.current.name
        #endif
    }

    /// This device's IPv4 address on Wi-Fi (`en0`), if it has one.
    static func lanAddress() -> String? {
        var head: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&head) == 0, let first = head else { return nil }
        defer { freeifaddrs(head) }
        for p in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let a = p.pointee
            guard let sa = a.ifa_addr, sa.pointee.sa_family == UInt8(AF_INET),
                  String(cString: a.ifa_name) == "en0" else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(sa, socklen_t(sa.pointee.sa_len), &host, socklen_t(host.count),
                           nil, 0, NI_NUMERICHOST) == 0 {
                return String(cString: host)
            }
        }
        return nil
    }
}

#Preview { NodeView() }
