% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function example_inference(device)
    % Inference: build a graph, compile it for a device, set its weights, run it.
    %   example_inference('cpu')
    % A two-layer classifier, x [1, 4] → relu(x·w1) [1, 8] → softmax(·w2) [1, 3].
    if nargin < 1, device = 'cpu'; end
    weights = @(n, seed) mod((0:n - 1) * 7919 + seed * 104729, 1000) / 1000 - 0.5;

    g = rlx.Graph('name', 'classifier');
    x = g.input('x', [1 4]);
    hidden = g.relu(g.matmul(x, g.param('w1', [4 8])));
    g.set_outputs(g.softmax(g.matmul(hidden, g.param('w2', [8 3]))));

    model = rlx.Session('device', device).compile(g); % consumes g
    fprintf('compiled on: %s\n', model.device());
    model.set_params(struct('w1', weights(32, 1), 'w2', weights(24, 2)));
    out = model.run(struct('x', [0.5 -1 2 0.25]));
    probs = out{1};
    fprintf('probs: %s\n', strtrim(sprintf('%.4f ', probs)));
    fprintf('probs sum: %.4f\n', sum(probs));
    fprintf('output shapes: %s dtypes: %s\n', mat2str(model.output_shapes()), strjoin(cellstr(model.output_dtypes()), ','));
end
