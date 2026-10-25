% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function c = strs(v)
    % A list of strings as a cell array ([] is empty, one string is a list of one).
    if isempty(v)
        c = {};
    elseif ischar(v)
        c = {v};
    elseif isstring(v)
        c = cellstr(v);
    else
        c = v;
    end
end
