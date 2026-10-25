% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function a = unflat(v, shape)
    % A flat row-major vector rlx returned, as a MATLAB array of `shape`.
    shape = double(shape(:).');
    if isscalar(shape)
        shape = [1 shape];
    end
    a = permute(reshape(v, fliplr(shape)), numel(shape):-1:1);
end
