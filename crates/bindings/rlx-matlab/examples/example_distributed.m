% RLX — versatile ML compiler + runtime.
% Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
% SPDX-License-Identifier: MIT OR Apache-2.0

function example_distributed(rank, world, peers)
    % Distributed: one process per rank, all with the same peers (here three
    % on loopback; across machines, one per host):
    %   octave --eval "example_distributed(1, 3, '127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502')" &
    %   octave --eval "example_distributed(2, 3, '127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502')" &
    %   octave --eval "example_distributed(0, 3, '127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502')"
    % (MATLAB: matlab -batch "example_distributed(…)".) Rank 0 coordinates:
    % collectives, pipelined inference (a stage shipped per worker),
    % data-parallel training. With no arguments it is a mesh of one.
    if nargin < 1, rank = 0; world = 1; peers = ''; end
    per_rank = 16;
    blobs = {};
    function uri = blob(name, values)
        path = fullfile(tempdir, ['rlx-dist-matlab-' name]);
        fid = fopen(path, 'w', 'ieee-le');
        fwrite(fid, values, 'single');
        fclose(fid);
        blobs{end + 1} = path;
        uri = ['file://' path];
    end

    options = struct('device', 'cpu');
    if ~isempty(peers)
        options.peers = strsplit(peers, ',');
    end
    g = rlx.Group.connect(rank, world, 'options', options);
    total = g.all_reduce(rank + 1, 'op', 'sum');
    gathered = g.all_gather(rank + 1);
    shared = g.broadcast(0, (rank == 0) * 42); % the root's length
    if rank == 0
        fprintf('mesh: rank %d of %d, leader: %s\n', g.rank(), g.world(), mat2str(g.is_leader()));
        fprintf('all_reduce sum: %g\n', total);
        fprintf('all_gather: %s\n', strjoin(arrayfun(@num2str, gathered, 'UniformOutput', false), ','));
        fprintf('broadcast: %g\n', shared);
    end
    if world == 1
        fprintf('a mesh of one: pass rank, world and peers to start more ranks\n');
        return;
    end
    if rank ~= 0
        g.serve_worker('options', struct('steps', 2));
        r = g.serve_trainer();
        g.send(0, 1, r.params.w); % rank 0 compares
        return;
    end

    stage_w = {mod((0:15) * 5, 7) / 7 - 0.3, mod((0:15) * 3, 5) / 5 - 0.2};
    caps = g.collect_caps();
    names = arrayfun(@(c) sprintf('rank %d on %s', c.rank, c.platform), caps, 'UniformOutput', false);
    fprintf('workers: %s\n', strjoin(names(:).', ', '));
    stage = rlx.Graph('name', 'stage');
    stage.set_outputs(stage.relu(stage.matmul(stage.input('x', [1 4]), stage.param('w', [4 4]))));
    for r = 1:world - 1
        uri = blob(sprintf('w%d.bin', r), stage_w{mod(r - 1, 2) + 1});
        g.ship_stage(r, stage, struct('device', 'cpu', 'weights', struct('w', uri)));
    end
    xs = {[1 -1 0.5 2], [0.25 0.5 -0.75 1]};
    for k = 1:numel(xs)
        g.send_activation(1, xs{k});
        y = g.recv_activation(world - 1);
        fprintf('pipeline: %.3f %.3f %.3f %.3f\n', y(1), y(2), y(3), y(4));
    end

    g.collect_caps();
    lr = rlx.Graph('name', 'lr');
    X = lr.input('X', [4 3]);
    Y = lr.input('y', [4 1]);
    wv = lr.param('w', [3 1]);
    d = lr.sub(lr.matmul(X, wv), Y);
    lr.set_outputs(lr.mean(lr.mul(d, d), [0 1]));
    bwd = rlx.grad(lr, wv);
    w_true = [1.5 -2 0.5];
    n = per_rank * world;
    x_all = mod((0:n * 3 - 1) * 37, 23) / 11.5 - 1;
    y_all = reshape(x_all, 3, n).' * w_true.';
    uris = struct('X', blob('X.bin', x_all), 'y', blob('y.bin', y_all), 'w', blob('w0.bin', [0 0 0]));
    function s = spec(r)
        s = struct('params', {{struct('name', 'w', 'uri', uris.w)}}, ...
            'data', {{struct('input', 'X', 'uri', uris.X, 'elem', 3, 'shard_start', r * per_rank, 'shard_len', per_rank), ...
                      struct('input', 'y', 'uri', uris.y, 'elem', 1, 'shard_start', r * per_rank, 'shard_len', per_rank)}}, ...
            'batch', 4, 'lr_per_epoch', repmat(0.2, 1, 40), 'device', 'cpu');
    end
    for r = 1:world - 1
        g.ship_train(r, bwd, spec(r));
    end
    mine = g.run_train(bwd, spec(0));
    fprintf('training loss: %.4f -> %.2e\n', mine.metrics.first_loss, mine.metrics.last_loss);
    w0 = mine.params.w;
    fprintf('rank 0 w: %.3f %.3f %.3f (true: 1.5 -2 0.5)\n', w0(1), w0(2), w0(3));
    for r = 1:world - 1
        theirs = g.recv(r, 1);
        fprintf('rank %d w agrees: %s\n', r, mat2str(all(abs(double(theirs(:)) - w0(:)) < 1e-5)));
    end
    for k = 1:numel(blobs)
        delete(blobs{k});
    end
end
