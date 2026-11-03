// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! A network run a layer at a time with its activations kept on the GPU.
//!
//! A 3-D net too big to run whole (a MeshNet over a brain, in a phone's tab)
//! runs a layer at a time over slabs of z-planes. Sending each slab's input
//! up and its output back made the run mostly transfers; here a layer's
//! input and output stay in [`ResidentStore`]s — 8 bits a value, scaled to
//! each plane's range — and a slab is four passes on the GPU:
//!
//! 1. *gather* the planes the layer's kernel reads into the plan's input
//!    texture, dequantised, through the instance norm the layer before ended
//!    in and the activations after it;
//! 2. the layer's own [`Plan`] (its convolution) on that texture;
//! 3. *reduce* each output plane to its min, max, sum and sum of squares;
//! 4. *scatter* the planes into the next store, quantised to their range.
//!
//! Only the per-plane sums come back (for the next norm), without waiting,
//! and the last layer's class scores.
//!
//! A store is linear: value `(plane, v)` of a store `[planes, plane]` is
//! texel `e = plane·len + v`, at `(e mod W, e div W)`. So is every tensor of
//! a plan (rlx-webgl lays a tensor row-major, widened or not).

use super::*;

/// A layer's activation on the GPU, `[planes, plane]` with `planes = C·D`.
pub struct ResidentStore {
    /// values, 8 bits (R8: the shader sees `q/255`)
    tex: WebGlTexture,
    /// each plane's low end and step (RG32F, linear)
    ranges: WebGlTexture,
    planes: usize,
    plane: usize,
    width: usize,
    range_width: usize,
}

impl ResidentStore {
    pub fn planes(&self) -> usize {
        self.planes
    }

    /// Bytes on the GPU.
    pub fn bytes(&self) -> usize {
        self.planes * self.plane + self.planes * 8
    }
}

/// Where one slab's input comes from (pass 1).
pub struct Gather<'a> {
    /// channels in, z-planes of the volume
    pub c_in: usize,
    pub d: usize,
    /// window depth and the output planes it serves
    pub depth: usize,
    pub core: usize,
    /// first output plane
    pub z0: usize,
    /// `k` runs of `core` planes `dilation` apart (else `depth` in a row)
    pub gathered: bool,
    pub dilation: usize,
    /// planes before `z0` the window starts (the layer's padding)
    pub reach: usize,
    /// the norm the layer before ended in (scale and shift per channel) and
    /// the activations after it
    pub norm: Option<(&'a [f32], &'a [f32], &'a [rlx_ir::op::Activation])>,
}

/// One slab's plane sums, on their way back.
pub struct PendingStats {
    buf: web_sys::WebGlBuffer,
    planes: usize,
}

const FS_GATHER: &str = r#"uniform highp sampler2D S; uniform highp sampler2D R; uniform highp sampler2D N;
uniform int uW; uniform int uRW; uniform int uOutW; uniform int uLen;
uniform int uPlane; uniform int uDepth; uniform int uCore; uniform int uD; uniform int uZ0;
uniform int uGathered; uniform int uDil; uniform int uReach;
uniform int uNorm; uniform int uActs; uniform ivec4 uAct;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int i = p.y * uOutW + p.x;
    if (i >= uLen) { o = vec4(0.0); return; }
    int v = i % uPlane;
    int jj = (i / uPlane) % uDepth;
    int c = i / (uPlane * uDepth);
    int z = uGathered != 0 ? uZ0 + jj % uCore + (jj / uCore) * uDil - uReach : uZ0 + jj - uReach;
    // outside the volume: the convolution's zero padding
    if (z < 0 || z >= uD) { o = vec4(0.0); return; }
    int pl = c * uD + z;
    int e = pl * uPlane + v;
    float q = texelFetch(S, ivec2(e % uW, e / uW), 0).r * 255.0;
    vec2 r = texelFetch(R, ivec2(pl % uRW, pl / uRW), 0).rg;
    float x = r.x + floor(q + 0.5) * r.y;
    if (uNorm != 0) {
        vec2 n = texelFetch(N, ivec2(c, 0), 0).rg;
        x = x * n.x + n.y;
        for (int a = 0; a < 4; a++) {
            if (a < uActs) x = actF(uAct[a], x);
        }
    }
    o = vec4(x, 0.0, 0.0, 1.0);
}"#;

/// Per output plane and chunk of 1024 values: min, max, sum, sum of squares.
const FS_PARTIAL: &str = r#"uniform highp sampler2D A; uniform int uAW; uniform int uPlane;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int start = p.x * 1024;
    int n = min(1024, uPlane - start);
    float lo = 3.4e38; float hi = -3.4e38; float s = 0.0; float q = 0.0;
    for (int k = 0; k < 1024; k++) {
        if (k >= n) break;
        int e = p.y * uPlane + start + k;
        float x = texelFetch(A, ivec2(e % uAW, e / uAW), 0).r;
        lo = min(lo, x); hi = max(hi, x); s += x; q += x * x;
    }
    o = vec4(lo, hi, s, q);
}"#;

/// Per output plane: low end, step, sum, sum of squares (from the chunks).
const FS_PLANE: &str = r#"uniform highp sampler2D P; uniform int uChunks;
out vec4 o;
void main() {
    int pl = int(gl_FragCoord.x);
    float lo = 3.4e38; float hi = -3.4e38; float s = 0.0; float q = 0.0;
    for (int k = 0; k < 4096; k++) {
        if (k >= uChunks) break;
        vec4 c = texelFetch(P, ivec2(k, pl), 0);
        lo = min(lo, c.x); hi = max(hi, c.y); s += c.z; q += c.w;
    }
    o = vec4(lo, hi > lo ? (hi - lo) / 255.0 : 0.0, s, q);
}"#;

/// Quantised values of output channel `uC`, planes `uZ0..uZ1`, into a store.
const FS_SCATTER: &str = r#"uniform highp sampler2D A; uniform highp sampler2D PS;
uniform int uAW; uniform int uW; uniform int uPlane; uniform int uD; uniform int uCore;
uniform int uC; uniform int uZ0; uniform int uE0; uniform int uE1;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int e = p.y * uW + p.x;
    if (e < uE0 || e >= uE1) discard;
    int v = e % uPlane;
    int z = (e / uPlane) % uD;
    int sp = uC * uCore + (z - uZ0);
    int ea = sp * uPlane + v;
    float x = texelFetch(A, ivec2(ea % uAW, ea / uAW), 0).r;
    vec2 r = texelFetch(PS, ivec2(sp, 0), 0).rg;
    float q = r.y > 0.0 ? floor((x - r.x) / r.y + 0.5) : 0.0;
    o = vec4(clamp(q, 0.0, 255.0) / 255.0, 0.0, 0.0, 1.0);
}"#;

/// The ranges of output channel `uC`, planes `uZ0..uZ1`, into a store's.
const FS_RANGES: &str = r#"uniform highp sampler2D PS;
uniform int uRW; uniform int uD; uniform int uCore; uniform int uC; uniform int uZ0;
uniform int uP0; uniform int uP1;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    int pl = p.y * uRW + p.x;
    if (pl < uP0 || pl >= uP1) discard;
    int z = pl % uD;
    vec2 r = texelFetch(PS, ivec2(uC * uCore + (z - uZ0), 0), 0).rg;
    o = vec4(r, 0.0, 1.0);
}"#;

/// The programs, compiled on first use.
#[derive(Default)]
pub(super) struct Programs {
    gather: Option<WebGlProgram>,
    partial: Option<WebGlProgram>,
    plane: Option<WebGlProgram>,
    scatter: Option<WebGlProgram>,
    ranges: Option<WebGlProgram>,
}

impl GlBackend {
    fn resident_program(
        &self,
        pick: fn(&mut Programs) -> &mut Option<WebGlProgram>,
        body: &str,
    ) -> Result<WebGlProgram> {
        let mut progs = self.resident.borrow_mut();
        let slot = pick(&mut progs);
        if slot.is_none() {
            *slot = Some(Self::program(&self.gl, &format!("{HEAD}{ACT_FNS}{body}"))?);
        }
        Ok(slot.clone().expect("compiled above"))
    }

    /// A float render target of `internal` (RG32F / RGBA32F), not pooled.
    fn resident_target(
        &self,
        w: usize,
        h: usize,
        internal: u32,
        format: u32,
    ) -> Result<WebGlTexture> {
        self.check_size(w, h)?;
        let gl = &self.gl;
        let tex = gl
            .create_texture()
            .ok_or_else(|| WebglError("create_texture".into()))?;
        gl.bind_texture(GL::TEXTURE_2D, Some(&tex));
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
            GL::TEXTURE_2D,
            0,
            internal as i32,
            w as i32,
            h as i32,
            0,
            format,
            GL::FLOAT,
            None,
        )
        .map_err(|e| err("tex_image_2d(resident)", e))?;
        self.set_nearest();
        Ok(tex)
    }

    /// An empty store of `planes` planes of `plane` values.
    pub fn store_new(&self, planes: usize, plane: usize) -> Result<ResidentStore> {
        let side = self.max_texture;
        let width = side.min(16384);
        let rows = (planes * plane).div_ceil(width);
        if rows > side {
            return Err(WebglError(format!(
                "a resident activation of {} M values needs {rows} rows of {width}, past this \
                 browser's {side}",
                (planes * plane) / 1_000_000
            )));
        }
        let gl = &self.gl;
        let tex = gl
            .create_texture()
            .ok_or_else(|| WebglError("create_texture".into()))?;
        gl.bind_texture(GL::TEXTURE_2D, Some(&tex));
        gl.tex_image_2d_with_i32_and_i32_and_i32_and_format_and_type_and_opt_array_buffer_view(
            GL::TEXTURE_2D,
            0,
            GL::R8 as i32,
            width as i32,
            rows as i32,
            0,
            GL::RED,
            GL::UNSIGNED_BYTE,
            None,
        )
        .map_err(|e| err("tex_image_2d(r8 store)", e))?;
        self.set_nearest();
        let range_width = side.min(4096);
        let ranges =
            self.resident_target(range_width, planes.div_ceil(range_width), GL::RG32F, GL::RG)?;
        Ok(ResidentStore {
            tex,
            ranges,
            planes,
            plane,
            width,
            range_width,
        })
    }

    /// Fill `s` from the host: `q` (`planes·plane` bytes) and each plane's
    /// low end and step.
    pub fn store_upload(&self, s: &ResidentStore, q: &[u8], ranges: &[(f32, f32)]) -> Result<()> {
        let gl = &self.gl;
        let n = s.planes * s.plane;
        let rows = n.div_ceil(s.width);
        let mut padded;
        let q = if q.len() < rows * s.width {
            padded = q.to_vec();
            padded.resize(rows * s.width, 0);
            &padded[..]
        } else {
            &q[..rows * s.width]
        };
        gl.bind_texture(GL::TEXTURE_2D, Some(&s.tex));
        gl.pixel_storei(GL::UNPACK_ALIGNMENT, 1);
        // SAFETY: the view is consumed by the upload before any JS alloc.
        unsafe {
            let view = js_sys::Uint8Array::view(q);
            gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_array_buffer_view(
                GL::TEXTURE_2D,
                0,
                0,
                0,
                s.width as i32,
                rows as i32,
                GL::RED,
                GL::UNSIGNED_BYTE,
                Some(&view),
            )
            .map_err(|e| err("tex_sub_image_2d(r8 store)", e))?;
        }
        gl.pixel_storei(GL::UNPACK_ALIGNMENT, 4);
        let rrows = s.planes.div_ceil(s.range_width);
        let mut rg = vec![0f32; rrows * s.range_width * 2];
        for (i, &(lo, step)) in ranges.iter().enumerate().take(s.planes) {
            rg[2 * i] = lo;
            rg[2 * i + 1] = step;
        }
        gl.bind_texture(GL::TEXTURE_2D, Some(&s.ranges));
        // SAFETY: as above.
        unsafe {
            let view = js_sys::Float32Array::view(&rg);
            gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_array_buffer_view(
                GL::TEXTURE_2D,
                0,
                0,
                0,
                s.range_width as i32,
                rrows as i32,
                GL::RG,
                GL::FLOAT,
                Some(&view),
            )
            .map_err(|e| err("tex_sub_image_2d(ranges)", e))?;
        }
        Ok(())
    }

    /// Free a store's GPU memory.
    pub fn store_free(&self, s: ResidentStore) {
        self.gl.delete_texture(Some(&s.tex));
        self.gl.delete_texture(Some(&s.ranges));
    }

    /// Passes 1 and 2 for one slab: gather from `src` into the input `input`
    /// of `plan` (run with `params`), and run it. The output texture, with
    /// its texture width.
    pub fn run_layer_slab(
        &self,
        plan_id: u64,
        plan: &Plan,
        params: &[(&str, &[f32])],
        input: &str,
        src: &ResidentStore,
        g: &Gather<'_>,
    ) -> Result<(WebGlTexture, usize)> {
        let slot = plan
            .steps
            .iter()
            .find_map(|s| match s {
                Step::Leaf {
                    out,
                    src: LeafSource::Input(n),
                } if n == input => Some(*out),
                _ => None,
            })
            .ok_or_else(|| WebglError(format!("the plan has no input '{input}'")))?;
        let (rows, cols) = plan.slot_dims[slot];
        let len = g.c_in * g.depth * src.plane;
        let gl = &self.gl;
        let x = self.target_texture(cols, rows)?;
        // (made before the samplers are bound: making a texture binds it to
        // the active unit)
        let norm_tex = match g.norm {
            Some((scale, shift, _)) => {
                let mut rg = vec![0f32; g.c_in * 2];
                for c in 0..g.c_in {
                    rg[2 * c] = scale[c];
                    rg[2 * c + 1] = shift[c];
                }
                let t = self.resident_target(g.c_in, 1, GL::RG32F, GL::RG)?;
                // SAFETY: the view is consumed by the upload before any JS alloc.
                unsafe {
                    let view = js_sys::Float32Array::view(&rg);
                    gl.tex_sub_image_2d_with_i32_and_i32_and_u32_and_type_and_opt_array_buffer_view(
                        GL::TEXTURE_2D,
                        0,
                        0,
                        0,
                        g.c_in as i32,
                        1,
                        GL::RG,
                        GL::FLOAT,
                        Some(&view),
                    )
                    .map_err(|e| err("tex_sub_image_2d(norm)", e))?;
                }
                Some(t)
            }
            None => None,
        };
        let prog = self.resident_program(|p| &mut p.gather, FS_GATHER)?;
        gl.use_program(Some(&prog));
        self.bind_tex_unit(&prog, "S", 0, &src.tex);
        self.bind_tex_unit(&prog, "R", 1, &src.ranges);
        if let Some(t) = &norm_tex {
            self.bind_tex_unit(&prog, "N", 2, t);
        } else {
            // (unread, but every sampler wants a texture)
            self.bind_tex_unit(&prog, "N", 2, &src.ranges);
        }
        gl.use_program(Some(&prog));
        for (name, v) in [
            ("uW", src.width),
            ("uRW", src.range_width),
            ("uOutW", cols),
            ("uLen", len),
            ("uPlane", src.plane),
            ("uDepth", g.depth),
            ("uCore", g.core),
            ("uD", g.d),
            ("uZ0", g.z0),
            ("uGathered", g.gathered as usize),
            ("uDil", g.dilation),
            ("uReach", g.reach),
            ("uNorm", g.norm.is_some() as usize),
        ] {
            self.set_i32(&prog, name, v as i32);
        }
        let acts: Vec<i32> = match g.norm {
            Some((_, _, a)) => a
                .iter()
                .map(|&a| crate::plan::map_act(a).map(act_code))
                .collect::<Result<_>>()?,
            None => Vec::new(),
        };
        if acts.len() > 4 {
            return Err(WebglError("more than 4 activations after a norm".into()));
        }
        self.set_i32(&prog, "uActs", acts.len() as i32);
        if let Some(loc) = self.uniform(&prog, "uAct") {
            let a = |i: usize| acts.get(i).copied().unwrap_or(0);
            gl.uniform4i(Some(&loc), a(0), a(1), a(2), a(3));
        }
        self.bind_target(&x, cols, rows)?;
        self.draw_banded(cols, rows, 4);
        if let Some(t) = norm_tex {
            gl.delete_texture(Some(&t));
        }
        // pass 2: the plan, its input that texture
        self.use_plan(plan_id);
        *self.gpu_inputs.borrow_mut() = vec![(input.to_string(), x)];
        let r = self.exec(plan, params);
        self.gpu_inputs.borrow_mut().clear();
        let mut tex = r?;
        let out = plan.outputs[0];
        let t = tex[out]
            .take()
            .ok_or_else(|| WebglError("the plan made no output".into()))?;
        self.free_run_textures(tex);
        Ok((t, plan.slot_dims[out].1))
    }

    /// Passes 3 and 4: slab output `a` (`[c_out, core, plane]` linear, `aw`
    /// wide) — its planes `z0..z1` — quantised into `dst` (`c_out·d` planes),
    /// and their sums on their way back.
    #[allow(clippy::too_many_arguments)]
    pub fn scatter_slab(
        &self,
        a: WebGlTexture,
        aw: usize,
        dst: &ResidentStore,
        c_out: usize,
        core: usize,
        z0: usize,
        z1: usize,
        d: usize,
    ) -> Result<PendingStats> {
        let gl = &self.gl;
        let plane = dst.plane;
        let planes = c_out * core;
        let chunks = plane.div_ceil(1024);
        // 3: per plane and chunk, then per plane
        let part = self.resident_target(chunks, planes, GL::RGBA32F, GL::RGBA)?;
        let prog = self.resident_program(|p| &mut p.partial, FS_PARTIAL)?;
        gl.use_program(Some(&prog));
        self.bind_tex_unit(&prog, "A", 0, &a);
        self.set_i32(&prog, "uAW", aw as i32);
        self.set_i32(&prog, "uPlane", plane as i32);
        self.bind_target(&part, chunks, planes)?;
        self.draw_banded(chunks, planes, 1024);
        let ps = self.resident_target(planes, 1, GL::RGBA32F, GL::RGBA)?;
        let prog = self.resident_program(|p| &mut p.plane, FS_PLANE)?;
        gl.use_program(Some(&prog));
        self.bind_tex_unit(&prog, "P", 0, &part);
        self.set_i32(&prog, "uChunks", chunks as i32);
        self.bind_target(&ps, planes, 1)?;
        self.draw();
        // 4: the values, then the ranges, a channel at a time
        let prog = self.resident_program(|p| &mut p.scatter, FS_SCATTER)?;
        gl.use_program(Some(&prog));
        self.bind_tex_unit(&prog, "A", 0, &a);
        self.bind_tex_unit(&prog, "PS", 1, &ps);
        for (name, v) in [
            ("uAW", aw),
            ("uW", dst.width),
            ("uPlane", plane),
            ("uD", d),
            ("uCore", core),
            ("uZ0", z0),
        ] {
            self.set_i32(&prog, name, v as i32);
        }
        let rows_total = (dst.planes * plane).div_ceil(dst.width);
        for c in 0..c_out {
            let (e0, e1) = ((c * d + z0) * plane, (c * d + z1) * plane);
            self.set_i32(&prog, "uC", c as i32);
            self.set_i32(&prog, "uE0", e0 as i32);
            self.set_i32(&prog, "uE1", e1 as i32);
            let (y0, y1) = (e0 / dst.width, (e1 - 1) / dst.width + 1);
            self.bind_target(&dst.tex, dst.width, rows_total)?;
            gl.enable(GL::SCISSOR_TEST);
            gl.scissor(0, y0 as i32, dst.width as i32, (y1 - y0) as i32);
            self.draw();
            gl.disable(GL::SCISSOR_TEST);
        }
        let prog = self.resident_program(|p| &mut p.ranges, FS_RANGES)?;
        gl.use_program(Some(&prog));
        self.bind_tex_unit(&prog, "PS", 0, &ps);
        for (name, v) in [
            ("uRW", dst.range_width),
            ("uD", d),
            ("uCore", core),
            ("uZ0", z0),
        ] {
            self.set_i32(&prog, name, v as i32);
        }
        let rrows = dst.planes.div_ceil(dst.range_width);
        for c in 0..c_out {
            let (p0, p1) = (c * d + z0, c * d + z1);
            self.set_i32(&prog, "uC", c as i32);
            self.set_i32(&prog, "uP0", p0 as i32);
            self.set_i32(&prog, "uP1", p1 as i32);
            let (y0, y1) = (p0 / dst.range_width, (p1 - 1) / dst.range_width + 1);
            self.bind_target(&dst.ranges, dst.range_width, rrows)?;
            gl.enable(GL::SCISSOR_TEST);
            gl.scissor(0, y0 as i32, dst.range_width as i32, (y1 - y0) as i32);
            self.draw();
            gl.disable(GL::SCISSOR_TEST);
        }
        // the sums back, without waiting
        self.bind_target(&ps, planes, 1)?;
        let buf = gl
            .create_buffer()
            .ok_or_else(|| WebglError("create_buffer".into()))?;
        gl.bind_buffer(GL::PIXEL_PACK_BUFFER, Some(&buf));
        gl.buffer_data_with_i32(GL::PIXEL_PACK_BUFFER, (planes * 16) as i32, GL::STREAM_READ);
        gl.read_pixels_with_i32(0, 0, planes as i32, 1, GL::RGBA, GL::FLOAT, 0)
            .map_err(|e| err("read_pixels(stats)", e))?;
        gl.bind_buffer(GL::PIXEL_PACK_BUFFER, None);
        gl.delete_texture(Some(&part));
        gl.delete_texture(Some(&ps));
        self.release(a);
        gl.flush();
        Ok(PendingStats { buf, planes })
    }

    /// Wait for the GPU (to time passes): a 1×1 read of the bound target,
    /// which cannot return before the work queued ahead of it is done
    /// (`finish` alone did not wait, in Chrome).
    pub fn finish(&self) {
        let mut px = [0f32; 4];
        // SAFETY: the view is filled before any JS alloc.
        unsafe {
            let view = js_sys::Float32Array::view_mut_raw(px.as_mut_ptr(), 4);
            let _ = self.gl.read_pixels_with_opt_array_buffer_view(
                0,
                0,
                1,
                1,
                GL::RGBA,
                GL::FLOAT,
                Some(&view),
            );
        }
    }

    /// A slab's per-plane `(sum, sum of squares)`, `[c_out, core]`.
    pub fn collect_stats(&self, p: PendingStats) -> Result<Vec<(f32, f32)>> {
        let gl = &self.gl;
        let mut v = vec![0f32; p.planes * 4];
        gl.bind_buffer(GL::PIXEL_PACK_BUFFER, Some(&p.buf));
        // SAFETY: the view is filled before any JS alloc.
        unsafe {
            let view = js_sys::Float32Array::view_mut_raw(v.as_mut_ptr(), v.len());
            gl.get_buffer_sub_data_with_i32_and_array_buffer_view(GL::PIXEL_PACK_BUFFER, 0, &view);
        }
        gl.bind_buffer(GL::PIXEL_PACK_BUFFER, None);
        gl.delete_buffer(Some(&p.buf));
        self.check_alive()?;
        Ok(v.chunks_exact(4).map(|c| (c[2], c[3])).collect())
    }

    /// The first `len` values of `t` (`w` wide, linear), and `t` freed.
    pub fn read_scores(&self, t: WebGlTexture, w: usize, len: usize) -> Result<Vec<f32>> {
        let h = len.div_ceil(w);
        let mut v = self.read_back(&t, w, h)?;
        v.truncate(len);
        self.release(t);
        self.check_alive()?;
        Ok(v)
    }
}
