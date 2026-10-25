% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function n = node(v)
    % An optional node: [] is rlx.NONE.
    if isempty(v)
        n = rlx.NONE;
    else
        n = v;
    end
end
