% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function v = decode(s)
    % A JSON report to MATLAB values (jsondecode); none is [].
    if isempty(s)
        v = [];
    else
        v = jsondecode(s);
    end
end
