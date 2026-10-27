// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A TCP link carried over Bluetooth LE (feature `ble`, on
//! [`webbluetooth`](https://crates.io/crates/webbluetooth)).
//!
//! The transport stays TCP: each end bridges a local TCP socket to an L2CAP
//! connection-oriented channel, a byte pipe with no per-write round trip.
//!
//! * [`serve`] (the peripheral, e.g. an iPad node): publishes an L2CAP
//!   channel, a GATT service whose one characteristic holds the PSM the system
//!   assigned it, and advertises both under the node's name. Each channel a
//!   central opens is joined to a fresh connection to `127.0.0.1:port` — the
//!   node's own mesh listener.
//! * [`bridge`] (the central, e.g. the launcher on the Mac): listens on
//!   `127.0.0.1:port`; each connection made there is joined to a channel
//!   opened to the peripheral advertising `name`.
//!
//! So a rank reaches a peer over BLE by dialing the bridge's local port, and
//! the peer sees the connection arrive from localhost: neither end's
//! transport knows it went over the air.

use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;

use futures_executor::block_on;
use webbluetooth::prelude::*;

/// The service an RLX node advertises over BLE.
pub const SERVICE: &str = "6b5a0001-3c1e-4d6b-9f3a-1a2b3c4d5e6f";
/// Its characteristic holding the published PSM (little-endian `u16`).
pub const PSM_CHARACTERISTIC: &str = "6b5a0002-3c1e-4d6b-9f3a-1a2b3c4d5e6f";

fn ble_err(e: impl std::fmt::Display) -> io::Error {
    io::Error::other(format!("ble: {e}"))
}

/// How long [`join`] keeps a channel open after its TCP side ends, for what
/// it sent to arrive, unless the far end closes first.
const LINGER: std::time::Duration = std::time::Duration::from_secs(5);

/// Bytes [`join`] queues on a channel before waiting for them to go out.
const QUEUE_MAX: usize = 1 << 20;

/// Copy `tcp` → `channel` until either closes; `channel` → `tcp` on a thread.
fn join(tcp: TcpStream, channel: webbluetooth::L2capChannel) {
    let _ = tcp.set_nodelay(true);
    let Some(mut incoming) = channel.take_incoming() else {
        eprintln!(
            "[ble] channel {:#06x}: its incoming stream was taken",
            channel.psm()
        );
        return;
    };
    let mut out = match tcp.try_clone() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[ble] channel {:#06x}: {e}", channel.psm());
            return;
        }
    };
    // The receiving side holds only the stream: the channel closes when the
    // sending side, below, is done with it (an L2CAP channel has no half
    // close), and the stream then ends here too.
    let psm = channel.psm();
    let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
    thread::spawn(move || {
        let _done = done_tx;
        block_on(async move {
            while let Some(chunk) = incoming.next().await {
                if let Err(e) = out.write_all(&chunk) {
                    eprintln!("[ble] channel {psm:#06x} → tcp: {e}");
                    break;
                }
            }
            let _ = out.shutdown(std::net::Shutdown::Write);
        });
    });
    let mut tcp = tcp;
    let mut buf = vec![0u8; 64 << 10];
    let mut queued = 0usize;
    loop {
        match tcp.read(&mut buf) {
            Ok(0) => break,
            Err(e) => {
                eprintln!("[ble] tcp → channel {:#06x}: {e}", channel.psm());
                break;
            }
            Ok(n) => {
                // Back-pressure, every [`QUEUE_MAX`] queued: waiting on each
                // write left the radio idle between them.
                let mut sent = channel.send(&buf[..n]).map_err(ble_err);
                queued += n;
                if sent.is_ok() && queued >= QUEUE_MAX {
                    queued = 0;
                    sent = block_on(channel.flushed()).map_err(ble_err);
                }
                if let Err(e) = sent {
                    eprintln!("[ble] tcp → channel {:#06x}: {e}", channel.psm());
                    break;
                }
            }
        }
    }
    // The TCP side ended: what is queued still goes out, and is given time
    // to arrive — flushed means handed to the radio, and letting the channel
    // (on the central, its connection) go at once lost a last message in the
    // air. The far end closing its side ends the wait early.
    let _ = block_on(channel.flushed());
    let _ = done_rx.recv_timeout(LINGER);
}

/// Run the peripheral side for the rest of the process (on its own thread):
/// advertise `name`, and join every channel opened to it with a connection
/// to `127.0.0.1:port`. Returns once advertising has started.
#[cfg(not(any(target_os = "tvos", target_os = "watchos", target_os = "visionos")))]
pub fn serve(name: &str, port: u16) -> io::Result<()> {
    use webbluetooth::peripheral::{Advertising, Characteristic, Peripheral, Request, Service};
    let name = name.to_string();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<io::Result<()>>();
    thread::spawn(move || {
        block_on(async move {
            let (peripheral, mut requests) = Peripheral::new();
            let started = async {
                peripheral.availability().await.map_err(ble_err)?;
                let psm = peripheral
                    .publish_l2cap_channel(false)
                    .await
                    .map_err(ble_err)?;
                let psm_char = Characteristic::new(PSM_CHARACTERISTIC)
                    .map_err(ble_err)?
                    .read()
                    .value(psm.to_le_bytes().to_vec());
                peripheral
                    .publish(
                        Service::new(SERVICE)
                            .map_err(ble_err)?
                            .characteristic(psm_char),
                    )
                    .await
                    .map_err(ble_err)?;
                peripheral
                    .start_advertising(
                        Advertising::new()
                            .local_name(name.clone())
                            .service(SERVICE)
                            .map_err(ble_err)?,
                    )
                    .await
                    .map_err(ble_err)?;
                io::Result::Ok(psm)
            }
            .await;
            match started {
                Ok(psm) => {
                    eprintln!("rlx-node: BLE: {name} advertising, L2CAP PSM {psm:#06x}");
                    let _ = ready_tx.send(Ok(()));
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e));
                    return;
                }
            }
            while let Some(request) = requests.next().await {
                if let Request::ChannelOpened(channel) = request {
                    match TcpStream::connect(("127.0.0.1", port)) {
                        Ok(tcp) => {
                            thread::spawn(move || join(tcp, channel));
                        }
                        Err(e) => eprintln!("rlx-node: BLE: no local listener at {port}: {e}"),
                    }
                }
            }
        });
    });
    ready_rx
        .recv()
        .map_err(|_| ble_err("the peripheral thread stopped"))?
}

/// Without a peripheral role on this platform (tvOS, watchOS, visionOS).
#[cfg(any(target_os = "tvos", target_os = "watchos", target_os = "visionos"))]
pub fn serve(_name: &str, _port: u16) -> io::Result<()> {
    Err(ble_err("this platform has no BLE peripheral role"))
}

/// Run the central side for the rest of the process: every connection made
/// to `127.0.0.1:port` is joined to a channel opened to the node advertising
/// `name` (found by name and [`SERVICE`], its PSM read from
/// [`PSM_CHARACTERISTIC`]). Returns once the port is bound.
pub fn bridge(name: &str, port: u16) -> io::Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let name = name.to_string();
    thread::spawn(move || {
        for tcp in listener.incoming().flatten() {
            let name = name.clone();
            thread::spawn(move || match block_on(open_retrying(&name)) {
                // The session and device stay alive with the channel: dropping
                // them lets the connection go, and the channel with it.
                Ok((channel, _session, _device)) => join(tcp, channel),
                Err(e) => eprintln!("[ble] {name}: {e}"),
            });
        }
    });
    Ok(())
}

/// How many times [`open`] is tried, and how long each try may take.
const OPEN_TRIES: usize = 10;
const OPEN_DEADLINE: std::time::Duration = std::time::Duration::from_secs(15);

/// [`open`], tried again until it works: the peer may not be advertising
/// yet (a launcher starts both ends at once, so the first scan can find
/// nothing), and `connect` to a device that went away waits forever.
async fn open_retrying(
    name: &str,
) -> io::Result<(webbluetooth::L2capChannel, Bluetooth, BluetoothDevice)> {
    let mut last = ble_err("never tried");
    for _ in 0..OPEN_TRIES {
        match webbluetooth::timeout(OPEN_DEADLINE, open(name)).await {
            Ok(Ok(opened)) => return Ok(opened),
            Ok(Err(e)) => last = e,
            Err(()) => last = ble_err(format!("no channel within {OPEN_DEADLINE:?}")),
        }
        webbluetooth::sleep(std::time::Duration::from_secs(1)).await;
    }
    Err(last)
}

/// Find `name`, connect, read its PSM and open the channel; with the session
/// and device the channel lives on.
async fn open(name: &str) -> io::Result<(webbluetooth::L2capChannel, Bluetooth, BluetoothDevice)> {
    let bluetooth = Bluetooth::with_chooser(webbluetooth::chooser::StrongestSignal::default());
    bluetooth.availability().await.map_err(ble_err)?;
    let device = bluetooth
        .request_device(
            RequestDeviceOptions::new()
                .filter(
                    DeviceFilter::new()
                        .service(SERVICE)
                        .map_err(ble_err)?
                        .name(name),
                )
                .optional_service(SERVICE)
                .map_err(ble_err)?,
        )
        .await
        .map_err(ble_err)?;
    let gatt = device.gatt();
    gatt.connect().await.map_err(ble_err)?;
    let psm = gatt
        .get_primary_service(SERVICE)
        .await
        .map_err(ble_err)?
        .get_characteristic(PSM_CHARACTERISTIC)
        .await
        .map_err(ble_err)?
        .read_value()
        .await
        .map_err(ble_err)?;
    let psm = u16::from_le_bytes([
        *psm.first().ok_or_else(|| ble_err("empty PSM"))?,
        psm.get(1).copied().unwrap_or(0),
    ]);
    let channel = device.open_l2cap_channel(psm).await.map_err(ble_err)?;
    Ok((channel, bluetooth, device))
}
