% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function s = json(v)
    % Options to JSON text: structs, containers.Map, cell arrays (lists),
    % numbers, logicals, strings.
    if isa(v, 'containers.Map')
        keys = v.keys();
        parts = cell(1, numel(keys));
        for k = 1:numel(keys)
            parts{k} = [jsonencode(keys{k}) ':' rlx.internal.json(v(keys{k}))];
        end
        s = ['{' strjoin(parts, ',') '}'];
    elseif isstruct(v) && isscalar(v)
        keys = fieldnames(v);
        parts = cell(1, numel(keys));
        for k = 1:numel(keys)
            parts{k} = [jsonencode(keys{k}) ':' rlx.internal.json(v.(keys{k}))];
        end
        s = ['{' strjoin(parts, ',') '}'];
    elseif iscell(v)
        parts = cellfun(@rlx.internal.json, v, 'UniformOutput', false);
        s = ['[' strjoin(parts, ',') ']'];
    else
        s = jsonencode(v);
    end
end
