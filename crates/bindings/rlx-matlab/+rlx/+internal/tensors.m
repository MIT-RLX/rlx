% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function m = tensors(p)
    % A tensor map rlx returned, as a containers.Map name → single row vector.
    [names, values] = rlx_mex('support_take_tensors', p);
    m = containers.Map('KeyType', 'char', 'ValueType', 'any');
    for k = 1:numel(names)
        m(names{k}) = values{k};
    end
end
