// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// The two rows whose shape is not a plain call: Optimizer.step updates the
// caller's map in place, Trainer.run takes callbacks.
module rlx

import time

// step updates each params[name] in place from grads[name].
pub fn (self &Optimizer) step(mut params map[string][]f64, grads map[string][]f64) ! {
	mut names := params.keys()
	names.sort()
	mut ps := [][]f32{}
	mut gs := [][]f32{}
	for name in names {
		g := grads[name] or {
			return RlxError{
				kind:    'type'
				message: "Optimizer.step: no gradient for '${name}'"
			}
		}
		ps << to_f32s(params[name])
		gs << to_f32s(g)
	}
	pp := ps.map(ptr_of(it))
	gp := gs.map(ptr_of(it))
	lens := ps.map(usize(it.len))
	check_rc(C.rlx_optimizer_step(self.h, ptr_of(cstrs(names)), ptr_of(pp), ptr_of(gp), ptr_of(lens),
		unsafe { nil }, unsafe { nil }, usize(names.len)))!
	for i, name in names {
		params[name] = ps[i].map(f64(it))
	}
}

// TrainSummary is what Trainer.run did.
pub struct TrainSummary {
pub:
	steps            int
	stopped          bool
	first_loss       ?f64
	last_loss        ?f64
	mean_loss        ?f64
	steps_per_second f64
	grad_norm        ?f64
}

@[params]
pub struct TrainerRunOpts {
pub:
	lr      ?fn (int) f64
	on_step ?fn (int, f64) bool
	every   int = 1
}

// run trains `steps` steps, pulling inputs from batch(step) (none: data
// exhausted). lr(step) schedules the rate; on_step(step, loss) returning
// false stops early (called every `every` steps).
pub fn (self &Trainer) run(steps int, batch fn (int) ?map[string][]f64, o TrainerRunOpts) !TrainSummary {
	every := if o.every < 1 { 1 } else { o.every }
	start := self.steps()!
	mut losses := []f64{}
	mut stopped := false
	sw := time.new_stopwatch()
	for i in 0 .. steps {
		s := start + i
		b := batch(s) or { break }
		if lr := o.lr {
			self.set_lr(lr(s))!
		}
		loss := self.step(b)!
		losses << loss
		if on_step := o.on_step {
			if (i + 1) % every == 0 && !on_step(s, loss) {
				stopped = true
				break
			}
		}
	}
	secs := f64(sw.elapsed().nanoseconds()) / 1e9
	mut sum := 0.0
	for l in losses {
		sum += l
	}
	return TrainSummary{
		steps:            losses.len
		stopped:          stopped
		first_loss:       if losses.len > 0 { ?f64(losses[0]) } else { none }
		last_loss:        if losses.len > 0 { ?f64(losses.last()) } else { none }
		mean_loss:        if losses.len > 0 { ?f64(sum / losses.len) } else { none }
		steps_per_second: losses.len / (if secs > 1e-9 { secs } else { 1e-9 })
		grad_norm:        self.grad_norm()!
	}
}
