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

//! Memory-plan SVG visualizer (plan #39).
//!
//! Borrowed from MAX's `layout/_print_svg.mojo`. Renders a memory
//! plan as a horizontal bar chart of arena slots laid out by
//! offset, with per-buffer rectangles colored by node id. Useful
//! to spot:
//!   - Wasted gaps between live ranges.
//!   - One huge buffer dominating the arena (often a sign of
//!     missing fusion / view-aliasing).
//!   - Aliased view nodes overlapping their parents (which is
//!     correct, but visually striking).
//!
//! Pure SVG string output — no extern deps. Pipe to a `.svg` file
//! and open in any browser.

use crate::memory::MemoryPlan;
use rlx_ir::NodeId;

/// Render `plan` as an SVG document. Width auto-scales to a fixed
/// pixel-per-byte ratio; tall enough to show every buffer on its
/// own row.
pub fn render(plan: &MemoryPlan) -> String {
    let row_height = 24u32;
    let pad = 8u32;
    let bytes_per_pixel = (plan.arena_size.max(1) / 800).max(1); // target ~800px wide
    let width = (plan.arena_size as u32 / bytes_per_pixel as u32).max(200) + 2 * pad;
    let n_buffers = plan.assignments.len() as u32;
    let height = n_buffers * row_height + 2 * pad + row_height; // header row

    let mut rows: Vec<(usize, usize, NodeId)> = plan
        .assignments
        .iter()
        .map(|(id, s)| (s.offset, s.size, *id))
        .collect();
    rows.sort();

    let mut s = String::new();
    s.push_str(&format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="monospace" font-size="11">"##
    ));
    s.push_str(r##"<rect x="0" y="0" width="100%" height="100%" fill="#fafafa"/>"##);

    // Title line.
    s.push_str(&format!(
        r##"<text x="{}" y="{}" fill="#333">arena_size={} ({} unshared, saved {})</text>"##,
        pad,
        pad + 12,
        plan.arena_size,
        plan.total_unshared_bytes(),
        plan.bytes_saved(),
    ));

    let track_y_start = pad + row_height;
    for (i, &(offset, size, id)) in rows.iter().enumerate() {
        let x = pad + (offset as u32 / bytes_per_pixel as u32);
        let w = (size as u32 / bytes_per_pixel as u32).max(2);
        let y = track_y_start + (i as u32 * row_height);
        let color = color_for(id);
        s.push_str(&format!(
            r##"<rect x="{x}" y="{y}" width="{w}" height="{rh}" fill="{color}" fill-opacity="0.8" stroke="#333" stroke-width="0.5"/>"##,
            rh = row_height - 2,
        ));
        s.push_str(&format!(
            r##"<text x="{}" y="{}" fill="#222">%{}: off={} sz={}</text>"##,
            x + 4,
            y + 14,
            id.0,
            offset,
            size,
        ));
    }

    s.push_str("</svg>");
    s
}

/// Stable color for a node id — deterministic rotation through a
/// muted palette so two visualizations of the same plan render
/// identically.
fn color_for(id: NodeId) -> &'static str {
    const PALETTE: [&str; 8] = [
        "#a5d8ff", "#b2f2bb", "#ffd8a8", "#fcc2d7", "#d0bfff", "#ffe066", "#ced4da", "#a5b4fc",
    ];
    PALETTE[(id.0 as usize) % PALETTE.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::plan_memory;
    use rlx_ir::*;

    #[test]
    fn render_emits_svg() {
        let mut g = Graph::new("svg-test");
        let f = DType::F32;
        let x = g.input("x", Shape::new(&[8, 8], f));
        let w = g.param("w", Shape::new(&[8, 8], f));
        let mm = g.matmul(x, w, Shape::new(&[8, 8], f));
        g.set_outputs(vec![mm]);
        let plan = plan_memory(&g);
        let svg = render(&plan);
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("arena_size="));
        // At least one buffer rectangle should be present.
        assert!(svg.contains("<rect"));
    }

    #[test]
    fn color_is_stable_for_same_id() {
        let a = color_for(NodeId(7));
        let b = color_for(NodeId(7));
        assert_eq!(a, b);
    }
}
