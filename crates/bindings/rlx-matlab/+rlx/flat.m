% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function v = flat(a)
    % An N-d MATLAB array (column-major) as the flat row-major vector rlx
    % takes: rlx.flat(A) for an m×n matrix is A's rows, one after another.
    v = reshape(permute(a, ndims(a):-1:1), 1, []);
end
