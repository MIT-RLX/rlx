// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Triangle geometry, and a conforming structured builder for layered domains.

/// Interpolation order of the element library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// Linear triangles: three nodes, constant gradient per element.
    P1,
    /// Quadratic triangles: three vertices plus three edge midpoints, gradient
    /// varying linearly within the element.
    ///
    /// Sides stay straight. An isoparametric element would curve them to follow
    /// a boundary, which buys accuracy on a curved domain and costs a
    /// position-dependent Jacobian; the domains here are polygonal, so straight
    /// sides keep the Jacobian constant and every geometric quantity — area,
    /// shape-function gradients — exactly as the linear element had them. Only
    /// the interpolation changes.
    P2,
}

impl Order {
    /// Nodes carried by one element.
    pub fn nodes_per_element(&self) -> usize {
        match self {
            Order::P1 => 3,
            Order::P2 => 6,
        }
    }
}

/// A triangulated planar domain.
///
/// Geometry only: the mesh carries no material data, so the caller keeps its own
/// per-element array in the same order as [`Mesh::tris`]. That keeps this type
/// reusable across problems whose element data have nothing in common.
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    /// Node coordinates `(x, y)`.
    pub nodes: Vec<[f64; 2]>,
    /// Triangles, counter-clockwise, as node indices.
    pub tris: Vec<[u32; 3]>,
    /// Node indices along the minimum-x edge, ascending in y.
    pub left_edge: Vec<u32>,
    /// Node indices along the maximum-x edge, ascending in y.
    ///
    /// Same length and same y coordinates as [`Mesh::left_edge`] when the mesh
    /// comes from [`layered::build`], which is what lets a periodic boundary be
    /// a plain node-to-node identification rather than an interpolation.
    pub right_edge: Vec<u32>,
    /// Node indices along the minimum-y edge, ascending in x.
    pub bottom_edge: Vec<u32>,
    /// Node indices along the maximum-y edge, ascending in x.
    pub top_edge: Vec<u32>,
    /// For a quadratic mesh, the three edge-midpoint nodes of each element, in
    /// the order (0-1, 1-2, 2-0) against [`Mesh::tris`]. `None` for a linear one.
    pub midside: Option<Vec<[u32; 3]>>,
}

impl Mesh {
    /// Interpolation order this mesh carries.
    pub fn order(&self) -> Order {
        if self.midside.is_some() {
            Order::P2
        } else {
            Order::P1
        }
    }

    /// The nodes of element `e`: three vertices, then three edge midpoints.
    ///
    /// The trailing three are meaningless on a linear mesh and are reported as
    /// the vertices themselves so the slice is always addressable; callers
    /// should consult [`Mesh::order`] rather than inspect them.
    pub fn element_nodes(&self, e: usize) -> [u32; 6] {
        let v = self.tris[e];
        match &self.midside {
            Some(m) => [v[0], v[1], v[2], m[e][0], m[e][1], m[e][2]],
            None => [v[0], v[1], v[2], v[0], v[1], v[2]],
        }
    }

    /// Shape-function gradients at a point of element `e`, given in area
    /// coordinates.
    ///
    /// Returns one gradient per node of the element, in the order of
    /// [`Mesh::element_nodes`]. On a linear element these are constant and the
    /// area coordinates are ignored.
    pub fn shape_gradients(&self, e: usize, l: [f64; 3]) -> [[f64; 2]; 6] {
        let (b, c) = self.grad_coeffs(e);
        let inv = 1.0 / (2.0 * self.area(e));
        // Gradients of the area coordinates: constant, because the sides are
        // straight and the Jacobian with them.
        let dl = [
            [b[0] * inv, c[0] * inv],
            [b[1] * inv, c[1] * inv],
            [b[2] * inv, c[2] * inv],
        ];
        match self.order() {
            Order::P1 => [dl[0], dl[1], dl[2], [0.0; 2], [0.0; 2], [0.0; 2]],
            Order::P2 => {
                // Vertex: N = L(2L - 1). Midside: N = 4 L_a L_b.
                let vertex = |i: usize| {
                    let k = 4.0 * l[i] - 1.0;
                    [k * dl[i][0], k * dl[i][1]]
                };
                let mid = |a: usize, b_: usize| {
                    [
                        4.0 * (l[a] * dl[b_][0] + l[b_] * dl[a][0]),
                        4.0 * (l[a] * dl[b_][1] + l[b_] * dl[a][1]),
                    ]
                };
                [
                    vertex(0),
                    vertex(1),
                    vertex(2),
                    mid(0, 1),
                    mid(1, 2),
                    mid(2, 0),
                ]
            }
        }
    }

    /// Shape-function values at a point of element `e`, in area coordinates.
    pub fn shape_values(&self, l: [f64; 3], order: Order) -> [f64; 6] {
        match order {
            Order::P1 => [l[0], l[1], l[2], 0.0, 0.0, 0.0],
            Order::P2 => [
                l[0] * (2.0 * l[0] - 1.0),
                l[1] * (2.0 * l[1] - 1.0),
                l[2] * (2.0 * l[2] - 1.0),
                4.0 * l[0] * l[1],
                4.0 * l[1] * l[2],
                4.0 * l[2] * l[0],
            ],
        }
    }

    /// Add edge-midpoint nodes, raising a linear mesh to quadratic.
    ///
    /// Midpoints are shared between the elements meeting on an edge, so the
    /// interpolation stays continuous across element boundaries. Boundary node
    /// lists are rebuilt to interleave the new midpoints, which is what keeps a
    /// tied or fixed edge complete: leaving them out would pin the vertices of a
    /// periodic boundary and leave the midpoints between them free.
    pub fn into_quadratic(mut self) -> Mesh {
        if self.midside.is_some() {
            return self;
        }
        use std::collections::HashMap;
        let mut lookup: HashMap<(u32, u32), u32> = HashMap::new();
        let mut midside = Vec::with_capacity(self.tris.len());

        for tri in &self.tris {
            let mut row = [0u32; 3];
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                let key = if a < b { (a, b) } else { (b, a) };
                row[k] = *lookup.entry(key).or_insert_with(|| {
                    let (p, q) = (self.nodes[a as usize], self.nodes[b as usize]);
                    self.nodes.push([0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])]);
                    (self.nodes.len() - 1) as u32
                });
            }
            midside.push(row);
        }

        // Rebuild each boundary run so consecutive vertices have their shared
        // midpoint between them.
        let interleave = |run: &[u32], lookup: &HashMap<(u32, u32), u32>| -> Vec<u32> {
            let mut out = Vec::with_capacity(run.len() * 2);
            for w in run.windows(2) {
                out.push(w[0]);
                let key = if w[0] < w[1] {
                    (w[0], w[1])
                } else {
                    (w[1], w[0])
                };
                if let Some(&m) = lookup.get(&key) {
                    out.push(m);
                }
            }
            if let Some(&last) = run.last() {
                out.push(last);
            }
            out
        };
        self.left_edge = interleave(&self.left_edge, &lookup);
        self.right_edge = interleave(&self.right_edge, &lookup);
        self.bottom_edge = interleave(&self.bottom_edge, &lookup);
        self.top_edge = interleave(&self.top_edge, &lookup);

        self.midside = Some(midside);
        self
    }

    /// Number of triangles.
    pub fn len(&self) -> usize {
        self.tris.len()
    }

    /// Whether the mesh has no elements.
    pub fn is_empty(&self) -> bool {
        self.tris.is_empty()
    }

    /// Signed area of element `e`. Positive for counter-clockwise winding.
    pub fn area(&self, e: usize) -> f64 {
        let [a, b, c] = self.tris[e];
        let (p, q, r) = (
            self.nodes[a as usize],
            self.nodes[b as usize],
            self.nodes[c as usize],
        );
        0.5 * ((q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1]))
    }

    /// Centroid of element `e`.
    pub fn centroid(&self, e: usize) -> [f64; 2] {
        let [a, b, c] = self.tris[e];
        let (p, q, r) = (
            self.nodes[a as usize],
            self.nodes[b as usize],
            self.nodes[c as usize],
        );
        [(p[0] + q[0] + r[0]) / 3.0, (p[1] + q[1] + r[1]) / 3.0]
    }

    /// Shape-function gradient coefficients `(b, c)` for element `e`, such that
    /// `grad N_i = (b_i, c_i) / (2 * area)`.
    ///
    /// These are the whole of a linear triangle: stiffness, gradient recovery
    /// and the flux-source term are all built from nothing else.
    pub fn grad_coeffs(&self, e: usize) -> ([f64; 3], [f64; 3]) {
        let [i, j, k] = self.tris[e];
        let (p, q, r) = (
            self.nodes[i as usize],
            self.nodes[j as usize],
            self.nodes[k as usize],
        );
        let b = [q[1] - r[1], r[1] - p[1], p[1] - q[1]];
        let c = [r[0] - q[0], p[0] - r[0], q[0] - p[0]];
        (b, c)
    }

    /// Gradient of a nodal field in each element, at the element centroid.
    ///
    /// Constant across a linear element, so for P1 the centroid value *is* the
    /// element value. On a quadratic element the gradient varies, and the
    /// centroid is where it is most accurate — the superconvergent point — so it
    /// is the right single value to report. Callers integrating a quantity built
    /// from the gradient should use [`Mesh::gradient_at`] over a quadrature rule
    /// instead of treating this as constant.
    pub fn gradient(&self, u: &[f64]) -> Vec<[f64; 2]> {
        let centroid = [1.0 / 3.0; 3];
        (0..self.len())
            .map(|e| self.gradient_at(e, centroid, u))
            .collect()
    }

    /// Gradient of a nodal field at a point of element `e`, in area coordinates.
    pub fn gradient_at(&self, e: usize, l: [f64; 3], u: &[f64]) -> [f64; 2] {
        if self.area(e) == 0.0 {
            return [0.0, 0.0];
        }
        let grad = self.shape_gradients(e, l);
        let nodes = self.element_nodes(e);
        let n = self.order().nodes_per_element();
        let (mut gx, mut gy) = (0.0, 0.0);
        for i in 0..n {
            let value = u[nodes[i] as usize];
            gx += value * grad[i][0];
            gy += value * grad[i][1];
        }
        [gx, gy]
    }
}

/// One tagged segment across `x` within a band.
///
/// Generic over the coordinate type, defaulting to `f64` so ordinary use is
/// unchanged. Building the same geometry with [`crate::Dual`] coordinates
/// instead makes every node position carry its derivative with respect to a
/// design parameter — which is what lets a shape derivative be differentiated
/// rather than differenced, without a second copy of the geometry that could
/// disagree with the first.
#[derive(Debug, Clone)]
pub struct Segment<T, S = f64> {
    /// Left edge.
    pub x0: S,
    /// Right edge.
    pub x1: S,
    /// Caller data for every element inside this segment.
    pub tag: T,
}

/// A horizontal band of the domain, spanning the full width.
#[derive(Debug, Clone)]
pub struct Band<T, S = f64> {
    /// Lower y bound.
    pub y0: S,
    /// Upper y bound.
    pub y1: S,
    /// Segments tiling `[0, width]` left to right, contiguous and gapless.
    pub segments: Vec<Segment<T, S>>,
    /// Largest permitted element height in this band.
    ///
    /// Per band rather than global, because a thin feature between thick ones is
    /// the usual reason a structured mesh becomes needlessly expensive.
    ///
    /// Plain `f64` even when the coordinates are dual: this is a meshing control,
    /// not a dimension. How many elements to place is a discrete decision that
    /// must *not* vary with a design parameter, or the mesh renumbers under
    /// perturbation and the derivative ceases to exist.
    pub max_dy: f64,
}

/// Conforming structured meshing over a stack of bands.
pub mod layered {
    use super::{Band, Mesh};
    use crate::dual::{Dual, Scalar};

    /// Merge a sorted list, dropping values closer together than `tol`.
    ///
    /// Ordered and compared on value alone. Which coordinates coincide is a
    /// property of the geometry, not of whichever parameter is being
    /// differentiated, so a derivative must never decide it.
    fn dedupe<S: Scalar>(mut v: Vec<S>, tol: f64) -> Vec<S> {
        v.sort_by(|a, b| {
            a.value()
                .partial_cmp(&b.value())
                .expect("finite coordinates")
        });
        let mut out: Vec<S> = Vec::with_capacity(v.len());
        for x in v {
            if out.last().is_none_or(|last| x.value() - last.value() > tol) {
                out.push(x);
            }
        }
        out
    }

    /// Every x coordinate at which some band changes segment, plus the two ends.
    fn x_features<T, S: Scalar>(width: S, bands: &[Band<T, S>]) -> Vec<S> {
        let zero = S::constant(0.0);
        let mut lines = vec![zero, width];
        for band in bands {
            for s in &band.segments {
                lines.push(s.x0.max_of(zero).min_of(width));
                lines.push(s.x1.max_of(zero).min_of(width));
            }
        }
        dedupe(lines, width.value() * 1e-9)
    }

    /// The x and y grid lines a plan places over a geometry.
    ///
    /// Generic over the coordinate type: at `f64` it is the mesh, and at
    /// [`Dual`] every grid line carries its derivative with respect to a design
    /// parameter. The division counts come from the plan and so are identical in
    /// both cases, which is exactly the property a shape derivative needs — the
    /// nodes move, the numbering does not.
    fn grid<T, S: Scalar>(width: S, bands: &[Band<T, S>], plan: &Plan) -> Option<(Vec<S>, Vec<S>)> {
        if plan.y.len() != bands.len() || plan.x.len() + 1 != x_features(width, bands).len() {
            return None;
        }
        let features = x_features(width, bands);
        let mut xs = Vec::new();
        for (i, w) in features.windows(2).enumerate() {
            let (a, b, n) = (w[0], w[1], plan.x[i]);
            for k in 0..n {
                xs.push(a + (b - a).scale(k as f64 / n as f64));
            }
        }
        xs.push(width);

        let mut ys = Vec::new();
        for (band, &n) in bands.iter().zip(&plan.y) {
            for k in 0..n {
                ys.push(band.y0 + (band.y1 - band.y0).scale(k as f64 / n as f64));
            }
        }
        ys.push(bands.last()?.y1);
        Some((xs, ys))
    }

    /// Derivative of every node position with respect to one design parameter.
    ///
    /// Built by running the *same* grid construction in dual arithmetic, so the
    /// derivative cannot drift from the geometry it belongs to. Node ordering
    /// matches [`build_planned`], so the result indexes alongside
    /// [`Mesh::nodes`].
    ///
    /// Returns `None` when the plan no longer fits, which means the perturbation
    /// changed the structure of the domain rather than only its dimensions.
    pub fn node_tangents<T>(
        width: Dual,
        bands: &[Band<T, Dual>],
        plan: &Plan,
    ) -> Option<Vec<[f64; 2]>> {
        let (xs, ys) = grid(width, bands, plan)?;
        let mut out = Vec::with_capacity(xs.len() * ys.len());
        for y in &ys {
            for x in &xs {
                out.push([x.d, y.d]);
            }
        }
        Some(out)
    }

    /// Divisions needed to keep every piece of `[a, b]` under `max_step`.
    ///
    /// The epsilon before the ceiling is load-bearing. Two domains that are
    /// geometrically similar — every length scaled by the same factor, including
    /// `max_step` — must receive the same number of divisions, or their
    /// discretisation errors differ and stop cancelling when their results are
    /// compared. Without it a ratio that is mathematically 8 arrives as
    /// `8.000000000000002` at one scale and `7.999999999999998` at another, and
    /// the two domains get 9 and 8 divisions.
    fn divisions(span: f64, max_step: f64) -> usize {
        if span <= 0.0 || max_step <= 0.0 {
            return 1;
        }
        (span / max_step - 1e-9).ceil().max(1.0) as usize
    }

    /// How a layered domain is to be divided, separated from where its
    /// boundaries actually are.
    ///
    /// Meshing makes two decisions at once: how many elements to place, and
    /// where to place them. For a single solve that distinction does not matter.
    /// For a *sensitivity* it is the whole game — a derivative with respect to a
    /// dimension is only defined if perturbing that dimension moves the nodes
    /// without renumbering them, and a division count chosen freshly from the
    /// perturbed geometry can step from 8 to 9 and destroy that.
    ///
    /// Capturing the counts once and reusing them makes the mesh a smooth
    /// function of the geometry, which is what a shape derivative requires. It
    /// is also why a structured mesh earns its keep here: an unstructured
    /// mesher has no comparable handle to hold fixed.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Plan {
        /// Divisions for each interval between consecutive x features.
        pub x: Vec<usize>,
        /// Divisions for each band, in order.
        pub y: Vec<usize>,
    }

    impl Plan {
        /// Choose divisions from size bounds.
        ///
        /// Computed from coordinate *values* only. How many elements to place is
        /// a discrete decision, and a decision that moved with a design
        /// parameter would renumber the mesh under perturbation and destroy the
        /// derivative it was meant to support.
        pub fn new<T, S: Scalar>(width: S, bands: &[Band<T, S>], max_dx: f64) -> Plan {
            let features = x_features(width, bands);
            Plan {
                x: features
                    .windows(2)
                    .map(|w| divisions(w[1].value() - w[0].value(), max_dx))
                    .collect(),
                y: bands
                    .iter()
                    .map(|b| divisions(b.y1.value() - b.y0.value(), b.max_dy))
                    .collect(),
            }
        }

        /// Whether `bands` has the feature structure this plan was built for.
        ///
        /// A perturbation that merges or splits a material boundary changes the
        /// structure, not merely the dimensions, and no fixed-topology
        /// derivative exists across it.
        pub fn fits<T, S: Scalar>(&self, width: S, bands: &[Band<T, S>]) -> bool {
            self.y.len() == bands.len() && self.x.len() + 1 == x_features(width, bands).len()
        }
    }

    /// Build a mesh over `[0, width]` crossed with the stacked `bands`.
    ///
    /// Returns the mesh and one cloned tag per triangle, in the same order as
    /// [`Mesh::tris`].
    ///
    /// `max_dx` bounds element width; each band bounds its own element height.
    /// Every segment edge in every band becomes a grid line, so no element ever
    /// straddles two tags — which is what lets the assembler take one
    /// coefficient per element and be exactly right rather than approximately
    /// so.
    ///
    /// # Panics
    ///
    /// If `width` is not positive, if `bands` is empty, or if some band leaves a
    /// y coordinate uncovered.
    pub fn build<T: Clone>(width: f64, bands: Vec<Band<T>>, max_dx: f64) -> (Mesh, Vec<T>) {
        let plan = Plan::new(width, &bands, max_dx);
        build_planned(width, bands, &plan).expect("a plan built from these bands fits them")
    }

    /// Build using division counts decided elsewhere.
    ///
    /// Returns `None` if `plan` does not fit — which is the signal that the
    /// perturbation was structural rather than dimensional, and that no
    /// fixed-topology comparison with the planned mesh is meaningful.
    ///
    /// # Panics
    ///
    /// If `width` is not positive, if `bands` is empty, or if some band leaves a
    /// y coordinate uncovered.
    pub fn build_planned<T: Clone>(
        width: f64,
        bands: Vec<Band<T>>,
        plan: &Plan,
    ) -> Option<(Mesh, Vec<T>)> {
        assert!(width > 0.0, "domain width must be positive");
        assert!(
            !bands.is_empty(),
            "a layered domain needs at least one band"
        );
        let (xs, ys) = grid(width, &bands, plan)?;

        let (nx, ny) = (xs.len(), ys.len());
        let mut nodes = Vec::with_capacity(nx * ny);
        for &y in &ys {
            for &x in &xs {
                nodes.push([x, y]);
            }
        }
        let idx = |i: usize, j: usize| (j * nx + i) as u32;

        let mut tris = Vec::with_capacity(2 * (nx - 1) * (ny - 1));
        let mut tags = Vec::with_capacity(tris.capacity());
        for j in 0..ny - 1 {
            let yc = 0.5 * (ys[j] + ys[j + 1]);
            let band = bands
                .iter()
                .find(|b| yc >= b.y0 && yc <= b.y1)
                .unwrap_or_else(|| panic!("no band covers y = {yc}"));
            for i in 0..nx - 1 {
                let xc = 0.5 * (xs[i] + xs[i + 1]);
                let tag = band
                    .segments
                    .iter()
                    .find(|s| xc >= s.x0 && xc <= s.x1)
                    .unwrap_or_else(|| panic!("no segment covers x = {xc}"))
                    .tag
                    .clone();

                let (n00, n10, n01, n11) =
                    (idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1));
                // Alternating the diagonal keeps the mesh from having a
                // preferred direction, which otherwise appears as a systematic
                // bias in any directional quantity recovered from the gradient.
                let (a, b) = if (i + j) % 2 == 0 {
                    ([n00, n10, n11], [n00, n11, n01])
                } else {
                    ([n00, n10, n01], [n10, n11, n01])
                };
                tris.push(a);
                tris.push(b);
                tags.push(tag.clone());
                tags.push(tag);
            }
        }

        let mesh = Mesh {
            left_edge: (0..ny).map(|j| idx(0, j)).collect(),
            right_edge: (0..ny).map(|j| idx(nx - 1, j)).collect(),
            bottom_edge: (0..nx).map(|i| idx(i, 0)).collect(),
            top_edge: (0..nx).map(|i| idx(i, ny - 1)).collect(),
            nodes,
            tris,
            midside: None,
        };
        Some((mesh, tags))
    }
}

#[cfg(test)]
mod tests {
    use super::layered::{Plan, build, build_planned};
    use super::*;

    fn two_band(width: f64) -> Vec<Band<u8>> {
        vec![
            Band {
                y0: 0.0,
                y1: 0.01,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: width,
                    tag: 1u8,
                }],
                max_dy: 0.005,
            },
            Band {
                y0: 0.01,
                y1: 0.012,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: width,
                    tag: 2u8,
                }],
                max_dy: 0.001,
            },
        ]
    }

    #[test]
    fn every_triangle_is_counter_clockwise_and_nondegenerate() {
        let (m, tags) = layered::build(0.05, two_band(0.05), 0.005);
        assert!(!m.is_empty());
        assert_eq!(tags.len(), m.len());
        for e in 0..m.len() {
            assert!(m.area(e) > 0.0, "element {e} has area {}", m.area(e));
        }
    }

    #[test]
    fn areas_sum_to_the_domain() {
        let (m, _) = layered::build(0.05, two_band(0.05), 0.005);
        let total: f64 = (0..m.len()).map(|e| m.area(e)).sum();
        let expect = 0.05 * 0.012;
        assert!(
            (total - expect).abs() / expect < 1e-12,
            "{total} vs {expect}"
        );
    }

    #[test]
    fn segment_boundaries_become_grid_lines() {
        // A tag occupying the middle half must not be straddled: every element
        // is wholly one tag or wholly the other.
        let width = 0.04;
        let bands = vec![Band {
            y0: 0.0,
            y1: 0.008,
            segments: vec![
                Segment {
                    x0: 0.0,
                    x1: 0.01,
                    tag: 0u8,
                },
                Segment {
                    x0: 0.01,
                    x1: 0.03,
                    tag: 7u8,
                },
                Segment {
                    x0: 0.03,
                    x1: width,
                    tag: 0u8,
                },
            ],
            max_dy: 0.004,
        }];
        let (m, tags) = layered::build(width, bands, 0.003);
        let tagged: f64 = (0..m.len())
            .filter(|&e| tags[e] == 7)
            .map(|e| m.area(e))
            .sum();
        let expect = 0.02 * 0.008;
        assert!(
            (tagged - expect).abs() / expect < 1e-12,
            "{tagged} vs {expect}"
        );
    }

    #[test]
    fn opposing_edges_share_a_coordinate_grid() {
        let (m, _) = layered::build(0.05, two_band(0.05), 0.005);
        assert_eq!(m.left_edge.len(), m.right_edge.len());
        for (&l, &r) in m.left_edge.iter().zip(&m.right_edge) {
            let (yl, yr) = (m.nodes[l as usize][1], m.nodes[r as usize][1]);
            assert!((yl - yr).abs() < 1e-15, "edge mismatch {yl} vs {yr}");
        }
    }

    #[test]
    fn a_thin_band_gets_finer_elements_than_a_thick_one() {
        let (m, tags) = layered::build(0.05, two_band(0.05), 0.005);
        let thin = (0..m.len())
            .filter(|&e| tags[e] == 2)
            .map(|e| m.area(e))
            .fold(0.0, f64::max);
        let thick = (0..m.len())
            .filter(|&e| tags[e] == 1)
            .map(|e| m.area(e))
            .fold(0.0, f64::max);
        assert!(thin < thick, "thin {thin} thick {thick}");
    }

    #[test]
    fn similar_domains_get_identical_meshes() {
        // Scale the domain and every size bound by the same factor: the element
        // count must not change, or per-scale discretisation error stops
        // cancelling between comparable runs.
        let reference = layered::build(0.05, two_band(0.05), 0.05 / 7.0).0.len();
        for scale in [0.61, 0.83, 1.0, 1.27, 1.93, 2.5] {
            let width = 0.05 * scale;
            let got = layered::build(width, two_band(width), width / 7.0).0.len();
            assert_eq!(got, reference, "element count changed at scale {scale}");
        }
    }

    #[test]
    fn a_plan_holds_topology_fixed_while_boundaries_move() {
        // The property shape derivatives need: move a material boundary by a
        // little and the nodes move with it, but the node count does not change.
        let width = 0.04;
        let bands_at = |split: f64| {
            vec![Band {
                y0: 0.0,
                y1: 0.008,
                segments: vec![
                    Segment {
                        x0: 0.0,
                        x1: split,
                        tag: 1u8,
                    },
                    Segment {
                        x0: split,
                        x1: width,
                        tag: 2u8,
                    },
                ],
                max_dy: 0.004,
            }]
        };
        let plan = Plan::new(width, &bands_at(0.017), 0.003);
        let (reference, _) = build_planned(width, bands_at(0.017), &plan).expect("fits");

        for split in [0.0165, 0.0168, 0.017, 0.0172, 0.0175] {
            let (m, _) = build_planned(width, bands_at(split), &plan).expect("fits");
            assert_eq!(
                m.nodes.len(),
                reference.nodes.len(),
                "node count moved at {split}"
            );
            assert_eq!(m.tris, reference.tris, "connectivity moved at {split}");
        }

        // Unplanned, the same sweep does redistribute divisions between the two
        // intervals — which is the failure the plan exists to prevent. Asserting
        // on the division vector rather than the node total, because the total
        // can stay fixed while the two intervals trade divisions, and a mesh
        // that renumbers internally is just as unusable for a derivative.
        let plans: Vec<Vec<usize>> = [0.017, 0.022]
            .iter()
            .map(|&s| Plan::new(width, &bands_at(s), 0.003).x)
            .collect();
        assert!(
            plans[0] != plans[1],
            "expected free meshing to redistribute divisions, got {plans:?}"
        );
    }

    #[test]
    fn a_plan_reports_when_it_no_longer_fits() {
        let width = 0.04;
        let one_band = vec![Band {
            y0: 0.0,
            y1: 0.008,
            segments: vec![Segment {
                x0: 0.0,
                x1: width,
                tag: 1u8,
            }],
            max_dy: 0.004,
        }];
        let plan = Plan::new(width, &one_band, 0.003);
        let two_bands = vec![
            Band {
                y0: 0.0,
                y1: 0.004,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: width,
                    tag: 1u8,
                }],
                max_dy: 0.004,
            },
            Band {
                y0: 0.004,
                y1: 0.008,
                segments: vec![Segment {
                    x0: 0.0,
                    x1: width,
                    tag: 2u8,
                }],
                max_dy: 0.004,
            },
        ];
        assert!(!plan.fits(width, &two_bands));
        assert!(build_planned(width, two_bands, &plan).is_none());
    }

    #[test]
    fn planned_and_free_builds_agree_when_the_plan_came_from_the_same_bands() {
        let (free, _) = build(0.05, two_band(0.05), 0.005);
        let plan = Plan::new(0.05, &two_band(0.05), 0.005);
        let (planned, _) = build_planned(0.05, two_band(0.05), &plan).expect("fits");
        assert_eq!(free.nodes.len(), planned.nodes.len());
        assert_eq!(free.tris, planned.tris);
        for (a, b) in free.nodes.iter().zip(&planned.nodes) {
            assert!((a[0] - b[0]).abs() < 1e-15 && (a[1] - b[1]).abs() < 1e-15);
        }
    }

    #[test]
    fn raising_the_order_shares_each_midpoint_between_its_two_elements() {
        // Continuity across element boundaries depends on it: give the two
        // elements meeting on an edge their own midpoint nodes and the
        // interpolation tears along every interior edge.
        let (m, _) = layered::build(0.05, two_band(0.05), 0.01);
        let vertices = m.nodes.len();
        let q = m.into_quadratic();
        assert_eq!(q.order(), Order::P2);

        let midside = q.midside.clone().expect("midside nodes");
        assert_eq!(midside.len(), q.tris.len());

        // Every midpoint sits exactly on its edge, and interior edges are shared.
        let mut uses: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for (e, tri) in q.tris.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                let mid = midside[e][k];
                let (p, r, mp) = (
                    q.nodes[a as usize],
                    q.nodes[b as usize],
                    q.nodes[mid as usize],
                );
                assert!((mp[0] - 0.5 * (p[0] + r[0])).abs() < 1e-15);
                assert!((mp[1] - 0.5 * (p[1] + r[1])).abs() < 1e-15);
                *uses.entry(mid).or_insert(0) += 1;
            }
        }
        // A structured mesh has interior edges, so some midpoints serve two.
        assert!(uses.values().any(|&n| n == 2), "no midpoint was shared");
        assert!(
            uses.values().all(|&n| n <= 2),
            "an edge served more than two elements"
        );
        // One new node per distinct edge.
        assert_eq!(q.nodes.len() - vertices, uses.len());
    }

    #[test]
    fn raising_the_order_completes_the_boundary_runs() {
        // A tied or fixed edge must list its midpoints too. Omitting them pins
        // the vertices of a periodic boundary and leaves the nodes between them
        // free, which is a subtle way to solve a different problem.
        let (m, _) = layered::build(0.05, two_band(0.05), 0.01);
        let before = m.left_edge.len();
        let q = m.into_quadratic();

        assert_eq!(
            q.left_edge.len(),
            2 * before - 1,
            "left edge not interleaved"
        );
        assert_eq!(q.right_edge.len(), q.left_edge.len());

        // Opposing edges still correspond node for node, which is what makes a
        // periodic tie a plain identification.
        for (&l, &r) in q.left_edge.iter().zip(&q.right_edge) {
            let (yl, yr) = (q.nodes[l as usize][1], q.nodes[r as usize][1]);
            assert!((yl - yr).abs() < 1e-15, "edge mismatch {yl} against {yr}");
        }
        // And they run monotonically, so the pairing is not merely coincidental.
        for w in q.left_edge.windows(2) {
            assert!(q.nodes[w[1] as usize][1] > q.nodes[w[0] as usize][1]);
        }
    }

    #[test]
    fn quadratic_shape_functions_form_a_partition_of_unity() {
        // The property that makes a constant reproducible, and therefore the
        // element consistent at all. Checked away from the nodes, where a wrong
        // shape function would still happen to sum to one.
        let (m, _) = layered::build(0.03, two_band(0.03), 0.01);
        let q = m.into_quadratic();
        for l in [
            [0.2, 0.3, 0.5],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.7, 0.1, 0.2],
        ] {
            let n = q.shape_values(l, Order::P2);
            let sum: f64 = n.iter().sum();
            assert!((sum - 1.0).abs() < 1e-14, "shape functions summed to {sum}");

            // Their gradients must correspondingly sum to zero.
            for e in [0usize, 5, 11] {
                let g = q.shape_gradients(e, l);
                let (gx, gy): (f64, f64) =
                    g.iter().fold((0.0, 0.0), |(x, y), v| (x + v[0], y + v[1]));
                let scale = 1.0 / q.area(e).sqrt();
                assert!(gx.abs() * scale < 1e-8 && gy.abs() * scale < 1e-8);
            }
        }
    }

    #[test]
    fn quadratic_elements_reproduce_a_quadratic_field_exactly() {
        // The defining capability. For u = x^2 the recovered gradient at the
        // centroid must be (2x, 0) exactly — something a linear element cannot
        // do at any resolution.
        let (m, _) = layered::build(0.03, two_band(0.03), 0.006);
        let q = m.into_quadratic();
        let u: Vec<f64> = q.nodes.iter().map(|&[x, _]| x * x).collect();
        for e in 0..q.len() {
            let l = [1.0 / 3.0; 3];
            let grad = q.shape_gradients(e, l);
            let nodes = q.element_nodes(e);
            let mut g = [0.0, 0.0];
            for i in 0..6 {
                g[0] += u[nodes[i] as usize] * grad[i][0];
                g[1] += u[nodes[i] as usize] * grad[i][1];
            }
            let xc = q.centroid(e)[0];
            assert!(
                (g[0] - 2.0 * xc).abs() < 1e-9,
                "d/dx was {} at x = {xc}",
                g[0]
            );
            assert!(g[1].abs() < 1e-9, "spurious d/dy of {}", g[1]);
        }
    }

    #[test]
    fn gradient_recovers_a_linear_field_exactly() {
        // The patch test a constant-gradient triangle must pass: for
        // u = 3x + 5y the recovered gradient is (3, 5) in every element.
        let (m, _) = layered::build(0.03, two_band(0.03), 0.004);
        let u: Vec<f64> = m.nodes.iter().map(|&[x, y]| 3.0 * x + 5.0 * y).collect();
        for g in m.gradient(&u) {
            assert!((g[0] - 3.0).abs() < 1e-9, "dx {}", g[0]);
            assert!((g[1] - 5.0).abs() < 1e-9, "dy {}", g[1]);
        }
    }
}
