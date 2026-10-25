% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

classdef Handle < handle
    % The base of every rlx class: owns a C handle (a uint64), freed by
    % `delete(obj)` or when the last reference goes.
    properties (SetAccess = private, Hidden)
        h = uint64(0)
        free = ''
    end
    methods
        function self = Handle(h, free)
            if isempty(h) || h == 0
                error('rlx:runtime', 'rlx returned NULL');
            end
            self.h = h;
            self.free = free;
        end

        function delete(self)
            if self.h ~= 0
                rlx_mex(self.free, self.h);
                self.h = uint64(0);
            end
        end
    end
end
