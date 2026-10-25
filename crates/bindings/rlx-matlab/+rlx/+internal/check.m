% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function check(rc)
    % Raise the last rlx error when `rc` is not RLX_OK: the identifier is
    % rlx:<kind> (type, range, runtime, panic, not_found).
    if rc ~= 0
        kinds = {'type', 'range', 'runtime', 'panic', 'not_found'};
        kind = 'other';
        if rc < 0 && -rc <= numel(kinds)
            kind = kinds{-rc};
        end
        error(['rlx:' kind], '%s', rlx_mex('support_last_error'));
    end
end
