function Y_hat = nn_predict(net, X)
%NN_PREDICT  Forward pass through the custom PLL neural network.
%
%  INPUT
%    net – struct returned by train_pll_model:
%            net.W  – cell array of weight matrices
%            net.b  – cell array of bias column vectors
%    X   – [n_in × N]  (features as rows, samples as columns)
%
%  OUTPUT
%    Y_hat – [n_out × N]

    n_layers = numel(net.W);
    A = X;

    for L = 1:n_layers-1
        Z = net.W{L} * A + net.b{L};
        A = max(0, Z);       % ReLU
    end

    % Linear output layer
    Y_hat = net.W{n_layers} * A + net.b{n_layers};
end
