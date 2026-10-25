% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function s = json_or_empty(v)
    % An optional JSON argument: [] is none.
    if isempty(v) && ~isstruct(v) && ~iscell(v)
        s = [];
    else
        s = rlx.internal.json(v);
    end
end
