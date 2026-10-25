% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function test_rlx()
    % The MATLAB / Octave binding against the real library; errors out on the
    % first failure.
    checks = 0;
    function check(ok, what)
        checks = checks + 1;
        if ~ok, error('rlx:test', 'failed: %s', what); end
    end
    function [g, w] = loss_graph()
        g = rlx.Graph('name', 'mse');
        x = g.input('x', [1 2]);
        y = g.input('y', [1 2]);
        w = g.param('w', [2 2]);
        d = g.sub(g.matmul(x, w), y);
        g.set_outputs(g.mean(g.mul(d, d), [0 1]));
    end

    % inference
    g = loss_graph();
    m = rlx.Session().compile(g);
    check(g.is_consumed(), 'compile consumes');
    m.set_param('w', [1 0 0 1]);
    out = m.run(struct('x', [1 2], 'y', [0 0]));
    check(out{1} == 2.5, 'loss 2.5');
    check(strcmp(m.device(), 'cpu'), 'device');
    check(isequal(rlx.unflat(rlx.flat([1 2; 3 4]), [2 2]), [1 2; 3 4]) && isequal(rlx.flat([1 2; 3 4]), [1 2 3 4]), 'flat');

    % errors
    try
        rlx.Graph().relu(42);
        check(false, 'range error raised');
    catch e
        check(strcmp(e.identifier, 'rlx:range'), 'range error');
    end
    try
        rlx.Graph().softmax(0, 'axes', 1);
        check(false, 'unknown option raised');
    catch e
        check(strcmp(e.identifier, 'rlx:type'), 'unknown option');
    end

    % training
    [g, w] = loss_graph();
    bwd = rlx.Session().compile(rlx.grad(g, w));
    bwd.set_param('w', [1 0 0 1]);
    out = bwd.run(struct('x', [1 2], 'y', [0 0], 'd_output', 1));
    check(out{2}(4) == 4, 'dw');
    t = rlx.Trainer(g.clone(), {'w'}, struct('w', [0 0 0 0]), ...
        'options', struct('optimizer', struct('kind', 'adamw', 'lr', 0.1)));
    r = t.run(300, @(~) struct('x', [1 1], 'y', [1 -1]), 'on_step', @(~, l) l > 1e-6);
    check(r.last_loss < 1e-4, 'trainer converges');
    opt = rlx.Optimizer('kind', 'sgd', 'options', struct('lr', 0.5));
    p = opt.step(struct('p', [1 -1]), struct('p', [2 -2]));
    check(isequal(p.p, [0 0]), 'optimizer (struct)');
    pm = containers.Map({'p'}, {[1 -1]});
    opt.step(pm, struct('p', [2 -2]));
    check(isequal(pm('p'), [0 0]), 'optimizer (Map, in place)');

    % loading
    path = fullfile(tempdir, 'rlx-matlab-test.gguf');
    gw = rlx.GgufWriter();
    gw.add_f32('w', 4, 'F32', [0.5 -1 2 0.25]);
    check(gw.write(path, 'architecture', 'demo') == 1, 'write');
    f = rlx.Gguf.open(path);
    check(strcmp(f.architecture(), 'demo'), 'architecture');
    tw = f.tensor('w');
    check(tw(3) == 2, 'tensor');
    check(strcmp(f.info('w').dtype, 'F32'), 'info');
    delete(path);

    % backends and a one-rank mesh
    devices = cellstr(rlx.devices());
    check(strcmp(devices{1}, 'cpu') && rlx.is_available('cpu'), 'cpu');
    g = loss_graph();
    runner = rlx.Runner(g, 'policy', struct('only', {{'cpu'}}));
    runner.set_param('w', [1 0 0 1]);
    res = runner.run(struct('x', [1 2], 'y', [0 0]));
    check(strcmp(res.device, 'cpu') && res.outputs{1} == 2.5, 'runner');
    mesh = rlx.Group.connect(0, 1);
    reduced = mesh.all_reduce([2 3], 'op', 'max');
    check(reduced(2) == 3, 'all_reduce');
    fprintf('test_rlx: %d checks passed\n', checks);
end
