// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Orchard: the reference planner is scored on the same episodes as the
//! network, and it is far above doing nothing.

use rlx_wsp::space::play_policy;
use rlx_wsp::{Act, EnvKind, Orchard, WeightSpace};

#[test]
fn planner_and_network_see_identical_episodes() {
    // Zero weights → argmax ties → always "stay"; the same policy through
    // play_policy must give the same events on the same seed.
    let mut ws = WeightSpace::new(EnvKind::Orchard, &[32, 32], Act::Tanh, 0);
    let net = ws.evaluate(64, 11);
    let stay = play_policy(Orchard::new, 64, 11, |_| 0);
    assert_eq!(net.mean_return, stay.mean_return);
    assert_eq!(net.positive_events, stay.positive_events);
    assert_eq!(net.negative_events, stay.negative_events);
}

#[test]
fn planner_far_above_standing_still() {
    let stay = play_policy(Orchard::new, 64, 5, |_| 0);
    let plan = play_policy(Orchard::new, 64, 5, Orchard::plan_action);
    assert!(
        plan.mean_return > stay.mean_return + 10.0,
        "{} vs {}",
        plan.mean_return,
        stay.mean_return
    );
}

#[test]
fn describe_reports_each_envs_own_label_format() {
    let fmt = |env| {
        let mut ws = WeightSpace::new(env, &[4, 4], Act::Tanh, 0);
        rlx_wsp::handle_json(&mut ws, r#"{"tool":"describe"}"#)["ok"]["environment"]
            ["obs_label_format"]
            .as_str()
            .unwrap()
            .to_string()
    };
    assert_eq!(
        fmt(EnvKind::Orchard),
        "fruit<(x=X,y=Y) | fruit>(x=X,y=Y) | bomb<(x=X,y=Y) | bomb>(x=X,y=Y) | paddle(x=X)"
    );
    assert_eq!(
        fmt(EnvKind::Bounce),
        "ball<(x=X,y=Y) | ball>(x=X,y=Y) | paddle(x=X)"
    );
}

#[test]
fn gusts_hard_is_wider_busier_and_hidden() {
    use rlx_wsp::Env;
    let spec = Orchard::gusts_hard().spec();
    assert_eq!(spec.name, "gusts_hard");
    assert_eq!(spec.obs_dim, 4 * 16 * 11 + 16);
    assert!(spec.description.contains("NOT given"));
    let ws = WeightSpace::new(EnvKind::GustsHard, &[32, 32], Act::Tanh, 0);
    assert_eq!(ws.state.mlp.widths, vec![4 * 16 * 11 + 16, 32, 32, 3]);
    // The planner still beats standing still by a wide margin.
    let stay = play_policy(Orchard::gusts_hard, 32, 5, |_| 0);
    let plan = play_policy(Orchard::gusts_hard, 32, 5, Orchard::plan_action);
    assert!(plan.mean_return > stay.mean_return + 10.0);
}

#[test]
fn hard_mnist_trigger_sits_where_digits_have_ink() {
    use rlx_wsp::mnist::{SIDE, Trigger};
    let cells = Trigger::CenterChecker.cells();
    assert_eq!(cells.len(), 16);
    assert!(
        cells
            .iter()
            .all(|(i, _)| (12..16).contains(&(i / SIDE)) && (12..16).contains(&(i % SIDE)))
    );
    assert_eq!(cells.iter().filter(|(_, v)| *v == 1.0).count(), 8);
    assert_eq!(
        EnvKind::MnistFixHard.trigger(),
        Some(Trigger::CenterChecker)
    );
    assert!(EnvKind::MnistFixHard.is_classifier());
}
