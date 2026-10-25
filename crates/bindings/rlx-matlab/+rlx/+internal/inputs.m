% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function [names, data, lens] = inputs(m)
    % Named tensors (struct or containers.Map of numeric arrays) for the gateway.
    [names, data] = rlx.internal.entries(m);
    lens = cellfun(@numel, data);
end
