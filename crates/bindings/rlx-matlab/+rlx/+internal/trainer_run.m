% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function summary = trainer_run(self, steps, batch, varargin)
    % rlx.Trainer.run: train `steps` steps, pulling inputs from batch(step)
    % ([] ends early). Options: 'lr' (@(step) rate), 'on_step' (@(step, loss)
    % keep_going), 'every' (call on_step every n steps).
    o = rlx.internal.opts(varargin, {'lr', 'on_step', 'every'}, {[], [], 1});
    every = max(o.every, 1);
    start = self.steps();
    losses = zeros(1, 0);
    stopped = false;
    t0 = tic;
    for i = 0:steps - 1
        s = start + i;
        b = batch(s);
        if isempty(b)
            break;
        end
        if ~isempty(o.lr)
            self.set_lr(o.lr(s));
        end
        loss = self.step(b);
        losses(end + 1) = loss; %#ok<AGROW>
        if ~isempty(o.on_step) && mod(i + 1, every) == 0 && ~o.on_step(s, loss)
            stopped = true;
            break;
        end
    end
    summary = struct('steps', numel(losses), 'stopped', stopped, ...
        'first_loss', [], 'last_loss', [], 'mean_loss', [], ...
        'steps_per_second', numel(losses) / max(toc(t0), 1e-9), 'grad_norm', self.grad_norm());
    if ~isempty(losses)
        summary.first_loss = losses(1);
        summary.last_loss = losses(end);
        summary.mean_loss = mean(losses);
    end
end
