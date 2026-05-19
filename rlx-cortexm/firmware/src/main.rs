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

//! MNIST-on-nRF52840 over USB CDC.
//!
//! Wire protocol (synchronous, framed by fixed length):
//!   host → device : 784 bytes  (one 28×28 i8 NHWC image, x_zp = 0)
//!   device → host : 1 byte     (predicted digit, 0..=9)
//!
//! Repeat. No headers, no checksum, no length prefix. The host knows
//! the framing because the model's input size is fixed.
//!
//! USB stack: nrf-usbd + usb-device + usbd-serial (CDC ACM). HFCLK is
//! cranked up to the external crystal because the USBD peripheral
//! needs it for proper enumeration.

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;

use nrf52840_hal as hal;
use nrf_usbd::Usbd;
use usb_device::class_prelude::UsbBusAllocator;
use usb_device::device::{StringDescriptors, UsbDeviceBuilder, UsbVidPid};
use usbd_serial::{SerialPort, USB_CLASS_CDC};

use rlx_cortexm::model::{infer, INPUT_LEN, SCRATCH_LEN};

// Two scratch buffers + the input buffer in BSS.
static mut BUF_A: [i8; SCRATCH_LEN] = [0; SCRATCH_LEN];
static mut BUF_B: [i8; SCRATCH_LEN] = [0; SCRATCH_LEN];
static mut INPUT: [i8; INPUT_LEN]   = [0; INPUT_LEN];

#[entry]
fn main() -> ! {
    let p = hal::pac::Peripherals::take().unwrap();

    // USB needs the external HF crystal. HF_OSC -> ext, then start the
    // 32 kHz crystal too (the USBD peripheral references it for some
    // power-state transitions).
    let clocks = hal::clocks::Clocks::new(p.CLOCK)
        .enable_ext_hfosc()
        .start_lfclk();

    // USB bus allocator. The peripheral wrapper takes the USBD reg
    // block and a reference to the clocks (needed to gate USB power).
    let usb_peri = hal::usbd::UsbPeripheral::new(p.USBD, &clocks);
    let usb_bus: UsbBusAllocator<Usbd<hal::usbd::UsbPeripheral>> =
        UsbBusAllocator::new(Usbd::new(usb_peri));

    let mut serial = SerialPort::new(&usb_bus);

    // VID/PID 0x16c0:0x27dd is the V-USB / pid.codes test pair for
    // CDC ACM. Fine for development; replace with a registered VID
    // before shipping anything.
    let mut usb_dev = UsbDeviceBuilder::new(&usb_bus, UsbVidPid(0x16c0, 0x27dd))
        .strings(&[StringDescriptors::default()
            .manufacturer("rlx")
            .product("rlx-cortexm-mnist")
            .serial_number("0001")])
        .unwrap()
        .device_class(USB_CLASS_CDC)
        .build();

    // Per-iteration accumulator: how many bytes of the current image
    // we've received.
    let mut filled: usize = 0;

    loop {
        if !usb_dev.poll(&mut [&mut serial]) {
            continue;
        }

        // Read into the tail of INPUT.
        // SAFETY: we own INPUT for the program lifetime; the borrow
        // is local to this match arm.
        let input_slice: &mut [i8] = unsafe {
            &mut *core::ptr::addr_of_mut!(INPUT)
        };
        // serial.read takes &mut [u8]; cast our i8 buffer.
        let tail_u8: &mut [u8] = unsafe {
            core::slice::from_raw_parts_mut(
                input_slice.as_mut_ptr().add(filled) as *mut u8,
                INPUT_LEN - filled,
            )
        };

        match serial.read(tail_u8) {
            Ok(0) | Err(_) => {}
            Ok(n) => {
                filled += n;
                if filled >= INPUT_LEN {
                    // SAFETY: BUF_A / BUF_B are exclusively borrowed
                    // for the duration of this call.
                    let pred = unsafe {
                        let a = &mut *core::ptr::addr_of_mut!(BUF_A);
                        let b = &mut *core::ptr::addr_of_mut!(BUF_B);
                        infer(input_slice, a, b)
                    };
                    // Send the prediction byte back. If the host
                    // isn't reading we just retry next poll cycle.
                    let _ = serial.write(&[pred as u8]);
                    let _ = serial.flush();
                    filled = 0;
                }
            }
        }
    }
}
