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
// RLX — flow-map self-distillation teachers (Python `flow_map_policy` / `fmq` ESD).

use crate::graph::CompiledFlowMapAgent;
use crate::spec::{DistillationType, RlSpec};

const JVP_EPS: f32 = 1e-4;

/// Eulerian self-distillation teacher (Eq. 5, `distillation_type == "mf"`).
///
/// `teacher = v_rt + (t - r) * JVP(u; tangents=(1, 0, v_rt))`.
pub fn esd_teacher_mf(
    agent: &mut CompiledFlowMapAgent,
    state: &[f32],
    a_r: &[f32],
    r: f32,
    t: f32,
    v_rt: &[f32],
) -> Vec<f32> {
    let jvp = jvp_velocity(agent, state, a_r, r, t, 1.0, 0.0, v_rt);
    let dt = t - r;
    v_rt.iter()
        .zip(jvp.iter())
        .map(|(&v, &j)| v + dt * j)
        .collect()
}

/// Lagrangian self-distillation (`distillation_type == "lsd"`).
///
/// Returns `(a_r', r', t', target)` with `a_r' = X_{s,u}(I_s)`, `r'=t'=u`, target `dX/du`.
pub fn esd_lsd_sample(
    agent: &mut CompiledFlowMapAgent,
    state: &[f32],
    a_r: &[f32],
    r: f32,
    t: f32,
    clip: f32,
) -> (Vec<f32>, f32, f32, Vec<f32>) {
    let u = agent.velocity(state, a_r, r, t);
    let dt = t - r;
    let x_su = clip_action(
        &a_r.iter().zip(u.iter()).map(|(&a, &v)| a + dt * v).collect::<Vec<_>>(),
        clip,
    );
    let d_x_du = jvp_jump_wrt_end_time(agent, state, a_r, r, t);
    (x_su, t, t, d_x_du)
}

/// Progressive self-distillation (`distillation_type == "psd"`).
pub fn esd_teacher_psd(
    agent: &mut CompiledFlowMapAgent,
    state: &[f32],
    a_r: &[f32],
    r: f32,
    t: f32,
    gamma: f32,
    clip: f32,
) -> (Vec<f32>, Vec<f32>) {
    let w = r + gamma * (t - r);
    let v_sw = agent.velocity(state, a_r, r, w);
    let dt_sw = w - r;
    let x_sw = clip_action(
        &a_r
            .iter()
            .zip(v_sw.iter())
            .map(|(&a, &v)| a + dt_sw * v)
            .collect::<Vec<_>>(),
        clip,
    );
    let v_wu = agent.velocity(state, &x_sw, w, t);
    let dt_wu = t - w;
    let x_wu = clip_action(
        &x_sw
            .iter()
            .zip(v_wu.iter())
            .map(|(&x, &v)| x + dt_wu * v)
            .collect::<Vec<_>>(),
        clip,
    );
    let student = agent.velocity(state, a_r, r, t);
    let teacher: Vec<f32> = v_sw
        .iter()
        .zip(v_wu.iter())
        .map(|(&a, &b)| gamma * a + (1.0 - gamma) * b)
        .collect();
    let _ = x_wu;
    (student, teacher)
}

/// Build one ESD regression target and optional remapped `(a_r, r, t)`.
pub fn esd_regression_target(
    kind: DistillationType,
    agent: &mut CompiledFlowMapAgent,
    spec: &RlSpec,
    state: &[f32],
    a_r: &[f32],
    r: f32,
    t: f32,
    v_rt: &[f32],
    gamma: f32,
) -> (Vec<f32>, f32, f32, Vec<f32>) {
    match kind {
        DistillationType::Mf => {
            let target = esd_teacher_mf(agent, state, a_r, r, t, v_rt);
            (a_r.to_vec(), r, t, target)
        }
        DistillationType::Lsd => esd_lsd_sample(agent, state, a_r, r, t, spec.action_clip),
        DistillationType::Psd => {
            let (_student, teacher) = esd_teacher_psd(agent, state, a_r, r, t, gamma, spec.action_clip);
            (a_r.to_vec(), r, t, teacher)
        }
    }
}

/// JVP of `u_{r,t}(a_r|s)` with tangents `(dr, dt, da)`.
fn jvp_velocity(
    agent: &mut CompiledFlowMapAgent,
    state: &[f32],
    a_r: &[f32],
    r: f32,
    t: f32,
    dr: f32,
    dt: f32,
    da: &[f32],
) -> Vec<f32> {
    let u0 = agent.velocity(state, a_r, r, t);
    let u_r = agent.velocity(state, a_r, r + JVP_EPS * dr, t + JVP_EPS * dt);
    let a_pert: Vec<f32> = a_r
        .iter()
        .zip(da.iter())
        .map(|(&a, &d)| a + JVP_EPS * d)
        .collect();
    let u_a = agent.velocity(state, &a_pert, r, t);
    u0.iter()
        .zip(u_r.iter())
        .zip(u_a.iter())
        .map(|((&u0, &ur), &ua)| (ur - u0) / JVP_EPS * dr + (ua - u0) / JVP_EPS)
        .collect()
}

/// JVP of `X_{s,u}(I_s) = I_s + (u-s)*u_{s,u}(I_s)` w.r.t. end time `u` (tangent 1).
fn jvp_jump_wrt_end_time(
    agent: &mut CompiledFlowMapAgent,
    state: &[f32],
    a_r: &[f32],
    r: f32,
    t: f32,
) -> Vec<f32> {
    let eps = JVP_EPS;
    let u0 = agent.velocity(state, a_r, r, t);
    let x0: Vec<f32> = a_r
        .iter()
        .zip(u0.iter())
        .map(|(&a, &v)| a + (t - r) * v)
        .collect();
    let u1 = agent.velocity(state, a_r, r, t + eps);
    let x1: Vec<f32> = a_r
        .iter()
        .zip(u1.iter())
        .map(|(&a, &v)| a + (t + eps - r) * v)
        .collect();
    x1.iter()
        .zip(x0.iter())
        .map(|(&x1, &x0)| (x1 - x0) / eps)
        .collect()
}

fn clip_action(a: &[f32], clip: f32) -> Vec<f32> {
    a.iter().map(|x| x.clamp(-clip, clip)).collect()
}
