% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function v = optional(value, has)
    % An optional result: [] when absent.
    if has
        v = value;
    else
        v = [];
    end
end
