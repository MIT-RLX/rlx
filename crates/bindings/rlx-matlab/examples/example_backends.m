% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function example_backends()
    % Picking a backend: what this build has, what suits a graph, and the ways
    % to choose — by name, by cost model, by policy, by benchmark, per run.
    %   example_backends()
    function g = model()
        g = rlx.Graph('name', 'mlp');
        x = g.input('x', [4 16]);
        g.set_outputs(g.tanh(g.matmul(x, g.param('w', [16 16]))));
    end
    w = mod(0:255, 5) / 10 - 0.2;
    feed = struct('x', mod(0:63, 7) / 7);

    available = cellstr(rlx.devices());
    fprintf('devices: %s\n', strjoin(available, ' '));
    fprintf('metal available: %s\n', mat2str(rlx.is_available('metal')));
    fprintf('fastest for graph: %s\n', rlx.fastest_device_for(model()));
    reference = [];
    for k = 1:numel(available)
        c = rlx.Session('device', available{k}).compile(model());
        c.set_param('w', w);
        y = c.run(feed);
        if isempty(reference), reference = y{1}; end
        fprintf('session %s: max|Δ| vs first = %.2e\n', available{k}, max(abs(y{1} - reference)));
    end
    flex = rlx.FlexibleSession('policy', struct('prefer', {{'metal', 'cpu'}}));
    fprintf('flexible session picked: %s\n', flex.compile(model()).device());
    runner = rlx.Runner(model());
    runner.set_param('w', w);
    fprintf('runner warmed: %s\n', strjoin(cellstr(runner.warm_all()), ' '));
    rows = runner.benchmark(feed, 'runs', 3);
    for k = 1:numel(rows)
        fprintf('  bench %s: exec_ms=%g\n', rows(k).device, rows(k).exec_ms);
    end
    r = runner.run(feed);
    fprintf('runner ran on: %s\n', r.device);
    r = runner.run_try({'metal', 'cpu'}, feed);
    fprintf('run_try chain: %s\n', r.device);
    router = rlx.Router(model());
    router.set_param('w', w);
    r = router.run_chain(feed);
    fprintf('router ran on: %s\n', r.device);
end
