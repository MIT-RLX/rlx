% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function params = optimizer_step(self, params, grads)
    % rlx.Optimizer.step: update each params entry from grads(name). A
    % containers.Map is updated in place; a struct (a value) comes back updated.
    if isstruct(params) && nargout == 0
        error('rlx:type', 'params is a struct, copied by value: use params = opt.step(params, grads), or pass a containers.Map');
    end
    [names, values] = rlx.internal.entries(params);
    gs = cell(1, numel(names));
    for k = 1:numel(names)
        if isa(grads, 'containers.Map') && grads.isKey(names{k})
            gs{k} = grads(names{k});
        elseif isstruct(grads) && isfield(grads, names{k})
            gs{k} = grads.(names{k});
        else
            error('rlx:type', 'Optimizer step: no gradient for ''%s''', names{k});
        end
    end
    [rc, updated] = rlx_mex('support_optimizer_step', self.h, names, values, gs);
    rlx.internal.check(rc);
    for k = 1:numel(names)
        v = reshape(cast(updated{k}, class(values{k})), size(values{k}));
        if isa(params, 'containers.Map')
            params(names{k}) = v;
        else
            params.(names{k}) = v;
        end
    end
end
