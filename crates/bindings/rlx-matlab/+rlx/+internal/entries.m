% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function [names, values] = entries(m)
    % The sorted names and values of a struct or containers.Map.
    if isa(m, 'containers.Map')
        names = m.keys();
        values = m.values(names);
    elseif isstruct(m)
        names = fieldnames(m).';
        values = struct2cell(m).';
        [names, order] = sort(names);
        values = values(order);
    else
        error('rlx:type', 'expected a struct or containers.Map of tensors, got a %s', class(m));
    end
end
