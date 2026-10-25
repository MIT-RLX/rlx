% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function [names, data, lens, dtypes] = typed_inputs(m)
    % Named typed tensors: each value a struct with `data` (uint8 bytes) and `dtype`.
    [names, vals] = rlx.internal.entries(m);
    data = cellfun(@(t) t.data, vals, 'UniformOutput', false);
    dtypes = cellfun(@(t) t.dtype, vals, 'UniformOutput', false);
    lens = cellfun(@numel, data);
end
