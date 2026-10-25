% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function example_loading()
    % Loading: a GGUF file (written here, f32 and quantized) and a safetensors
    % file into a compiled graph, and quantizing in memory.
    %   example_loading()
    w = mod(0:63, 9) / 4 - 1;

    gguf = fullfile(tempdir, 'rlx-example-matlab.gguf');
    writer = rlx.GgufWriter();
    writer.add_f32('proj.weight', [8 8], 'F32', w);
    writer.add_f32('proj.weight_q8', [8 8], 'Q8_0', w);
    writer.write(gguf, 'architecture', 'demo', 'metadata', containers.Map({'demo.layers'}, {1}));
    f = rlx.Gguf.open(gguf);
    fprintf('gguf architecture: %s\n', f.architecture());
    fprintf('gguf tensors: %s\n', strjoin(cellstr(f.tensor_names()), ' '));
    exact = f.tensor('proj.weight');
    q8 = f.tensor('proj.weight_q8');
    fprintf('gguf f32 exact: %s q8 max|Δ|: %.4f\n', mat2str(isequal(double(exact), w)), max(abs(exact - q8)));

    g = rlx.Graph('name', 'proj');
    x = g.input('x', [1 8]);
    g.set_outputs(g.matmul(x, g.param('proj.weight', [8 8])));
    c = rlx.Session().compile(g);
    c.set_param('proj.weight', exact);
    y = c.run(struct('x', ones(1, 8)));
    fprintf('gguf -> graph: %.2f %.2f %.2f\n', y{1}(1), y{1}(2), y{1}(3));

    st = fullfile(tempdir, 'rlx-example-matlab.safetensors');
    header = '{"proj.weight":{"dtype":"F32","shape":[8,8],"data_offsets":[0,256]}}';
    header = [header repmat(' ', 1, mod(8 - mod(numel(header), 8), 8))];
    fid = fopen(st, 'w', 'ieee-le');
    fwrite(fid, numel(header), 'uint64');
    fwrite(fid, header, 'char');
    fwrite(fid, w, 'single');
    fclose(fid);
    loaded = rlx.load_weights('safetensors', st);
    fprintf('safetensors: [proj.weight] exact: %s\n', mat2str(isequal(double(loaded('proj.weight')), w)));

    packed = rlx.quantize(w, 'Q8_0');
    back = rlx.dequantize(packed, 'Q8_0', 'count', numel(w));
    fprintf('quantize q8_0: %d -> %d bytes, round trip %d values\n', numel(w) * 4, numel(packed), numel(back));
    delete(gguf);
    delete(st);
    fprintf('argmax: %d\n', rlx.argmax([0.1 2 0.3]));
end
