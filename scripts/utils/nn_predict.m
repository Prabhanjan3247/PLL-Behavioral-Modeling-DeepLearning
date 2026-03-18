function [Y_hat, cache] = nn_predict(net, X)
%NN_PREDICT  Unified forward pass for custom and toolbox-based networks.
%
%  INPUT
%    net - either:
%            (1) legacy custom struct with net.W / net.b
%            (2) Deep Learning Toolbox network object
%    X   - [n_in x N]  (features as rows, samples as columns)
%
%  OUTPUT
%    Y_hat - [n_out x N]
%    cache - forward-pass cache for backprop (legacy custom net only)

    cache = [];

    % ------------------------------------------------------------------ %
    %  Legacy custom network path
    % ------------------------------------------------------------------ %
    if isstruct(net) && isfield(net, 'W') && isfield(net, 'b')
        n_layers = numel(net.W);
        A = X;

        collect_cache = (nargout > 1);
        if collect_cache
            cache.A = cell(1, n_layers + 1);
            cache.Z = cell(1, n_layers);
            cache.A{1} = X;
        end

        for L = 1:n_layers-1
            Z = net.W{L} * A + net.b{L};
            A = max(0, Z);       % ReLU
            if collect_cache
                cache.Z{L} = Z;
                cache.A{L+1} = A;
            end
        end

        % Linear output layer
        Z = net.W{n_layers} * A + net.b{n_layers};
        Y_hat = Z;

        if collect_cache
            cache.Z{n_layers} = Z;
            cache.A{n_layers+1} = Z;
        end
        return
    end

    % ------------------------------------------------------------------ %
    %  Deep Learning Toolbox path
    % ------------------------------------------------------------------ %
    if isa(net, 'SeriesNetwork') || isa(net, 'DAGNetwork') || isa(net, 'dlnetwork')
        X_obs = X';
        Y_obs = predict(net, X_obs);
        Y_hat = Y_obs';
        return
    end

    error('Unsupported network type in nn_predict.');
end
