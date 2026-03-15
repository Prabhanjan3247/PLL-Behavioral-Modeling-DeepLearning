function [net, mu_X, sigma_X, mu_Y, sigma_Y, train_info] = train_pll_model(X, Y, model_save_path)
%TRAIN_PLL_MODEL  Train a deep neural network to predict PLL metrics.
%
%  Pure MATLAB implementation — requires NO additional toolboxes.
%  Uses mini-batch Adam optimisation with ReLU hidden layers.
%
%  Architecture:  9 → 128 → 256 → 128 → 64 → 3
%
%  INPUTS
%    X               – [N × 9] raw input features
%    Y               – [N × 3] raw output values
%    model_save_path – path to save trained model (.mat)  [optional]
%
%  OUTPUTS
%    net        – struct with fields W (weights), b (biases), arch
%    mu_X/sigma_X – input normalisation stats
%    mu_Y/sigma_Y – output normalisation stats
%    train_info   – struct with TrainingLoss and ValidationLoss per epoch

    if nargin < 3, model_save_path = ''; end

    fprintf('\n========================================\n');
    fprintf('  PLL Deep-Learning Model Training\n');
    fprintf('  (Pure MATLAB - No Toolbox Required)\n');
    fprintf('========================================\n');

    % ------------------------------------------------------------------ %
    %  Log-transform inputs/outputs spanning many decades
    % ------------------------------------------------------------------ %
    log_cols_X = [1, 3, 4, 5, 6, 7, 9];
    log_cols_Y = [1, 3];

    X_t = X;
    X_t(:, log_cols_X) = log10(abs(X(:, log_cols_X)) + eps);
    X_t(:, 2)          = log10(abs(X(:, 2)) + 1e-10);   % Idb: soft log

    Y_t = Y;
    Y_t(:, log_cols_Y) = log10(abs(Y(:, log_cols_Y)) + eps);

    % ------------------------------------------------------------------ %
    %  Z-score normalisation
    % ------------------------------------------------------------------ %
    mu_X    = mean(X_t);
    sigma_X = std(X_t)  + eps;
    X_norm  = (X_t - mu_X) ./ sigma_X;

    mu_Y    = mean(Y_t);
    sigma_Y = std(Y_t)  + eps;
    Y_norm  = (Y_t - mu_Y) ./ sigma_Y;

    % ------------------------------------------------------------------ %
    %  Train / Validation / Test split  (70 / 15 / 15)
    % ------------------------------------------------------------------ %
    N_total = size(X_norm, 1);
    rng(0);
    idx     = randperm(N_total);
    n_train = round(0.70 * N_total);
    n_val   = round(0.15 * N_total);

    idx_tr  = idx(1 : n_train);
    idx_val = idx(n_train+1 : n_train+n_val);
    idx_te  = idx(n_train+n_val+1 : end);

    % Store as  features × samples  (columns = observations)
    Xtr = X_norm(idx_tr,  :)';    Ytr = Y_norm(idx_tr,  :)';
    Xva = X_norm(idx_val, :)';    Yva = Y_norm(idx_val, :)';
    Xte = X_norm(idx_te,  :)';    Yte = Y_norm(idx_te,  :)';

    fprintf('Samples → Train: %d | Val: %d | Test: %d\n', ...
            n_train, n_val, numel(idx_te));

    % ------------------------------------------------------------------ %
    %  Network initialisation  (He initialisation for ReLU)
    % ------------------------------------------------------------------ %
    arch = [9, 128, 256, 128, 64, 3];
    n_layers = numel(arch) - 1;

    net.arch = arch;
    net.W    = cell(1, n_layers);
    net.b    = cell(1, n_layers);

    for L = 1:n_layers
        fan_in      = arch(L);
        fan_out     = arch(L+1);
        net.W{L}    = randn(fan_out, fan_in) * sqrt(2 / fan_in);
        net.b{L}    = zeros(fan_out, 1);
    end

    fprintf('\nArchitecture: %s\n', strjoin(arrayfun(@num2str, arch, ...
            'UniformOutput', false), ' → '));

    % ------------------------------------------------------------------ %
    %  Hyper-parameters
    % ------------------------------------------------------------------ %
    lr          = 3e-4;      % Adam learning rate
    beta1       = 0.9;
    beta2       = 0.999;
    epsilon     = 1e-8;
    lambda      = 1e-4;      % L2 weight decay
    n_epochs    = 500;
    batch_size  = 256;
    patience    = 30;        % early-stopping patience (epochs)

    % Adam moment buffers
    mW = cell(1, n_layers);  vW = cell(1, n_layers);
    mb = cell(1, n_layers);  vb = cell(1, n_layers);
    for L = 1:n_layers
        mW{L} = zeros(size(net.W{L}));  vW{L} = zeros(size(net.W{L}));
        mb{L} = zeros(size(net.b{L}));  vb{L} = zeros(size(net.b{L}));
    end

    % ------------------------------------------------------------------ %
    %  Training loop
    % ------------------------------------------------------------------ %
    train_loss_hist = zeros(1, n_epochs);
    val_loss_hist   = zeros(1, n_epochs);

    best_val_loss = Inf;
    best_net      = net;
    no_improve    = 0;
    t_adam        = 0;          % Adam time step

    n_tr = size(Xtr, 2);
    fprintf('\nTraining for up to %d epochs (early stop patience=%d) …\n\n', ...
            n_epochs, patience);
    fprintf('%6s  %12s  %12s\n', 'Epoch', 'Train-MSE', 'Val-MSE');
    fprintf('%s\n', repmat('-', 1, 34));

    for epoch = 1:n_epochs
        % Shuffle training data
        perm = randperm(n_tr);
        Xtr_s = Xtr(:, perm);
        Ytr_s = Ytr(:, perm);

        epoch_loss = 0;
        n_batches  = 0;

        for b_start = 1 : batch_size : n_tr
            b_end = min(b_start + batch_size - 1, n_tr);
            Xb = Xtr_s(:, b_start:b_end);
            Yb = Ytr_s(:, b_start:b_end);

            % Forward pass
            [Y_hat, cache] = nn_forward(net, Xb);

            % Loss (MSE)
            diff = Y_hat - Yb;
            batch_loss = mean(diff(:).^2);
            epoch_loss = epoch_loss + batch_loss;

            % Backward pass
            grads = nn_backward(net, cache, Yb, lambda);

            % Adam update
            t_adam = t_adam + 1;
            for L = 1:n_layers
                mW{L} = beta1*mW{L} + (1-beta1)*grads.dW{L};
                vW{L} = beta2*vW{L} + (1-beta2)*grads.dW{L}.^2;
                mb{L} = beta1*mb{L} + (1-beta1)*grads.db{L};
                vb{L} = beta2*vb{L} + (1-beta2)*grads.db{L}.^2;

                mW_hat = mW{L} / (1 - beta1^t_adam);
                vW_hat = vW{L} / (1 - beta2^t_adam);
                mb_hat = mb{L} / (1 - beta1^t_adam);
                vb_hat = vb{L} / (1 - beta2^t_adam);

                net.W{L} = net.W{L} - lr * mW_hat ./ (sqrt(vW_hat) + epsilon);
                net.b{L} = net.b{L} - lr * mb_hat ./ (sqrt(vb_hat) + epsilon);
            end
            n_batches = n_batches + 1;
        end

        epoch_loss = epoch_loss / n_batches;

        % Validation loss
        Yva_hat   = nn_forward(net, Xva);
        diff_val  = Yva_hat - Yva;
        val_loss  = mean(diff_val(:).^2);

        train_loss_hist(epoch) = epoch_loss;
        val_loss_hist(epoch)   = val_loss;

        % Console log every 25 epochs
        if mod(epoch, 25) == 0 || epoch == 1
            fprintf('%6d  %12.6f  %12.6f\n', epoch, epoch_loss, val_loss);
        end

        % Early stopping / best-model tracking
        if val_loss < best_val_loss - 1e-8
            best_val_loss = val_loss;
            best_net      = net;
            no_improve    = 0;
        else
            no_improve = no_improve + 1;
        end

        if no_improve >= patience
            fprintf('\nEarly stopping at epoch %d  (best val MSE = %.6f)\n', ...
                    epoch, best_val_loss);
            n_epochs = epoch;
            break
        end
    end

    net = best_net;   % restore best weights

    % ------------------------------------------------------------------ %
    %  Test-set performance
    % ------------------------------------------------------------------ %
    Yte_hat   = nn_forward(net, Xte);
    Yte_pred_t = (Yte_hat' .* sigma_Y) + mu_Y;
    Yte_true_t = (Yte'     .* sigma_Y) + mu_Y;

    Y_pred_raw = Yte_pred_t;
    Y_true_raw = Yte_true_t;
    Y_pred_raw(:, log_cols_Y) = 10 .^ Yte_pred_t(:, log_cols_Y);
    Y_true_raw(:, log_cols_Y) = 10 .^ Yte_true_t(:, log_cols_Y);

    output_names = {'LockTime_s', 'PhaseNoise_dBcHz', 'Fout_Hz'};
    fprintf('\n--- Test-Set Performance ---\n');
    for k = 1:3
        ss_res = sum((Y_true_raw(:,k) - Y_pred_raw(:,k)).^2);
        ss_tot = sum((Y_true_raw(:,k) - mean(Y_true_raw(:,k))).^2);
        r2   = 1 - ss_res / max(ss_tot, eps);
        rmse = sqrt(mean((Y_true_raw(:,k) - Y_pred_raw(:,k)).^2));
        fprintf('  %-22s  R² = %.4f   RMSE = %.4e\n', output_names{k}, r2, rmse);
    end

    % Pack training history
    train_info.TrainingLoss   = train_loss_hist(1:n_epochs);
    train_info.ValidationLoss = val_loss_hist(1:n_epochs);
    train_info.log_cols_X     = log_cols_X;
    train_info.log_cols_Y     = log_cols_Y;

    % ------------------------------------------------------------------ %
    %  Save model
    % ------------------------------------------------------------------ %
    log_cols_in  = log_cols_X;   %#ok<NASGU>
    log_cols_out = log_cols_Y;   %#ok<NASGU>

    if ~isempty(model_save_path)
        if ~isempty(fileparts(model_save_path)) && ...
                ~exist(fileparts(model_save_path), 'dir')
            mkdir(fileparts(model_save_path));
        end
        save(model_save_path, 'net', 'mu_X', 'sigma_X', 'mu_Y', 'sigma_Y', ...
             'log_cols_in', 'log_cols_out', 'output_names', 'train_info');
        fprintf('\nModel saved to: %s\n', model_save_path);
    end
end

%% ======================================================================
%  Local helper: forward pass
%% ======================================================================
function [Y_hat, cache] = nn_forward(net, X)
%NN_FORWARD  Forward pass through the network.
%  X: [n_in × N]   Y_hat: [n_out × N]
    n_layers = numel(net.W);
    cache.A  = cell(1, n_layers+1);
    cache.Z  = cell(1, n_layers);
    cache.A{1} = X;

    A = X;
    for L = 1:n_layers-1
        Z = net.W{L} * A + net.b{L};
        A = max(0, Z);       % ReLU
        cache.Z{L} = Z;
        cache.A{L+1} = A;
    end
    % Linear output layer
    L = n_layers;
    Z = net.W{L} * A + net.b{L};
    cache.Z{L} = Z;
    cache.A{L+1} = Z;
    Y_hat = Z;
end

%% ======================================================================
%  Local helper: backward pass
%% ======================================================================
function grads = nn_backward(net, cache, Y, lambda)
%NN_BACKWARD  Backpropagation with L2 regularisation.
    n_layers = numel(net.W);
    N = size(Y, 2);
    grads.dW = cell(1, n_layers);
    grads.db = cell(1, n_layers);

    % Output layer gradient (MSE loss, linear activation)
    dZ = (cache.A{n_layers+1} - Y) / N;

    for L = n_layers:-1:1
        grads.dW{L} = dZ * cache.A{L}' + lambda * net.W{L};
        grads.db{L} = sum(dZ, 2);
        if L > 1
            dA = net.W{L}' * dZ;
            dZ = dA .* (cache.Z{L-1} > 0);   % ReLU derivative
        end
    end
end
