% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function o = opts(args, names, defaults)
    % Optional parameters: name-value pairs or one struct, over `defaults`.
    o = cell2struct(defaults(:), names(:), 1);
    if isempty(args)
        return;
    end
    if numel(args) == 1 && isstruct(args{1})
        keys = fieldnames(args{1});
        vals = struct2cell(args{1});
    else
        if mod(numel(args), 2) ~= 0
            error('rlx:type', 'optional parameters are name-value pairs (takes: %s)', strjoin(names, ', '));
        end
        keys = args(1:2:end);
        vals = args(2:2:end);
    end
    for k = 1:numel(keys)
        key = char(keys{k});
        if ~any(strcmp(names, key))
            error('rlx:type', 'unknown option ''%s'' (takes: %s)', key, strjoin(names, ', '));
        end
        o.(key) = vals{k};
    end
end
