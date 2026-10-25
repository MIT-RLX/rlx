% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function example_training(device)
    % Training: fit y = x·w_true with the Trainer, checkpoint and resume it,
    % then the same fit by hand with grad and an Optimizer.
    %   example_training('cpu')
    if nargin < 1, device = 'cpu'; end
    w_true = [2 -1 0.5];
    batch = 8;
    function b = data(step)
        x = zeros(1, batch * 3);
        y = zeros(1, batch);
        for i = 0:batch - 1
            for k = 0:2
                v = mod(step * 31 + i * 7 + k * 3, 17) / 8.5 - 1;
                x(i * 3 + k + 1) = v;
                y(i + 1) = y(i + 1) + v * w_true(k + 1);
            end
        end
        b = struct('x', x, 'y', y);
    end
    fmt = @(v) sprintf('%.3f %.3f %.3f', v(1), v(2), v(3));

    g = rlx.Graph('name', 'regression');
    x = g.input('x', [batch 3]);
    y = g.input('y', [batch 1]);
    w = g.param('w', [3 1]);
    d = g.sub(g.matmul(x, w), y);
    g.set_outputs(g.mean(g.mul(d, d), [0 1]));

    opts = struct('device', device, 'optimizer', struct('kind', 'adamw', 'lr', 0.1));
    trainer = rlx.Trainer(g.clone(), {'w'}, struct('w', [0 0 0]), 'options', opts);
    first = trainer.evaluate(data(0));
    trainer.run(400, @data, 'lr', @(s) 0.1 * 0.5 * (1 + cos(pi * s / 400)), ...
        'on_step', @(~, loss) loss > 1e-6);
    fprintf('trainer loss: %.4f -> %.6f in %d steps\n', first, trainer.evaluate(data(0)), trainer.steps());
    p = trainer.params();
    fprintf('trainer w: %s\n', fmt(p('w')));

    path = fullfile(tempdir, 'rlx-example-train-matlab.gguf');
    written = trainer.save(path);
    restored = rlx.Trainer(g.clone(), {'w'}, struct('w', [0 0 0]), 'options', struct('device', device));
    info = restored.load(path);
    fprintf('checkpoint: %d tensors, restored step %d\n', written, info.step);
    delete(path);
    restored.freeze('names', {'w'});
    p = restored.params();
    before = fmt(p('w'));
    restored.step(data(1));
    p = restored.params();
    fprintf('frozen: %s unchanged: %s\n', strjoin(cellstr(restored.frozen()), ','), mat2str(strcmp(fmt(p('w')), before)));
    restored.unfreeze();

    backward = rlx.Session('device', device).compile(rlx.grad(g, w)); % outputs [loss, dw]
    opt = rlx.Optimizer('kind', 'sgd', 'options', struct('lr', 0.2));
    params = struct('w', [0 0 0]);
    loss = 0;
    for step = 0:299
        backward.set_param('w', params.w);
        feed = data(step);
        feed.d_output = 1;
        outs = backward.run(feed);
        loss = outs{1};
        params = opt.step(params, struct('w', outs{2}));
    end
    fprintf('manual loss: %.6f w: %s\n', loss, fmt(params.w));
end
