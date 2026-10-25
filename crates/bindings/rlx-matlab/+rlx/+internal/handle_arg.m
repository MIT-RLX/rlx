% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function h = handle_arg(v)
    % An optional handle argument: [] is none.
    if isempty(v)
        h = uint64(0);
    else
        h = v.h;
    end
end
