% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function [flat, n] = rows(v, w, splat)
    % Rows of `w` values — an n×w matrix or a cell array of rows — flattened
    % row by row; with `splat`, a single value stands for all `w`.
    if iscell(v)
        n = numel(v);
        flat = zeros(1, n * w);
        for i = 1:n
            r = v{i};
            if numel(r) == w
                flat((i - 1) * w + (1:w)) = r(:).';
            elseif splat && isscalar(r)
                flat((i - 1) * w + (1:w)) = r;
            else
                error('rlx:type', 'row %d needs %d values, got %d', i, w, numel(r));
            end
        end
    elseif isempty(v)
        flat = [];
        n = 0;
    elseif size(v, 2) == w
        n = size(v, 1);
        flat = reshape(v.', 1, []);
    elseif splat && size(v, 2) == 1
        n = size(v, 1);
        flat = reshape(repmat(v, 1, w).', 1, []);
    else
        error('rlx:type', 'rows need %d values each, got a %dx%d matrix', w, size(v, 1), size(v, 2));
    end
end
