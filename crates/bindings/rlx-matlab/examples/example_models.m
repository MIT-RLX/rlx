% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function example_models()
    % Models: any provider's model, opened by path or id and run by name. A
    % graph saved as an .rlxp package runs anywhere; a plugin's models (a
    % downstream collection's LLMs, or the reference plugin's demo:count)
    % load through RLX_MODEL_PLUGINS.
    %   example_models()
    g = rlx.Graph('name', 'affine');
    x = g.input('x', [1 2]);
    g.set_outputs(g.matmul(x, g.param('w', [2 2])));
    path = fullfile(tempdir, 'rlx-example-matlab.rlxp');
    rlx.save_model(g, path, struct('w', [2 0 0 3]), ...
        'model', struct('task', 'graph', 'family', 'affine', 'outputs', {{'y'}}));

    m = rlx.Model.open(path, 'options', struct('device', 'cpu'));
    info = m.info();
    fprintf('model: %s (%s) on %s\n', info.family, info.task, info.device);
    y = m.run(struct('x', [1 1]));
    fprintf('model run: %.3f %.3f\n', y{1}(1), y{1}(2));
    delete(m);
    delete(path);

    lm = rlx.Model.open('demo:count');
    info = lm.info();
    fprintf('plugin model: %s (%s)\n', info.family, info.task);
    fprintf('model text: %s\n', lm.generate_text('c', 'options', struct('max_tokens', 4)));
    fprintf('model ids: %s\n', strtrim(sprintf('%d ', lm.generate(2, 'max_tokens', 3))));

    % By task: a chat, speech out and back in, embeddings.
    msgs = {struct('role', 'user', 'content', 'c')};
    fprintf('chat: %s\n', lm.chat(msgs, 'options', struct('max_tokens', 4)));
    speech = rlx.Model.open('demo:voice').synthesize('hello');
    fprintf('voice: %d samples at %d Hz\n', numel(speech.audio), speech.sample_rate);
    ear = rlx.Model.open('demo:speech');
    fprintf('heard: %s\n', ear.transcribe(speech.audio, 'sample_rate', speech.sample_rate));
    rows = rlx.Model.open('demo:embed').embed({'hello', 'sky'});
    if iscell(rows)
        rows = cell2mat(cellfun(@(r) r(:)', rows(:), 'UniformOutput', false));
    end
    rows = rows.';
    fprintf('embed: %s\n', strtrim(sprintf('%d ', round(rows(:)))));
end
