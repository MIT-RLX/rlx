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

//! INT8 ReLU — `out = max(x, zero_point)` in place.

#[inline(always)]
pub fn relu_i8(buf: &mut [i8], zero_point: i32) {
    let zp = zero_point.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
    for v in buf.iter_mut() {
        if *v < zp {
            *v = zp;
        }
    }
}
