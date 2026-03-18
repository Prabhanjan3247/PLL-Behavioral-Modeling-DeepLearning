function [net, mu_X, sigma_X, mu_Y, sigma_Y, train_info] = train_pll_model(X, Y, model_save_path)
%TRAIN_PLL_MODEL  Train a deep neural network to predict PLL metrics.
%
%  Deep Learning Toolbox implementation using trainNetwork.
%  Architecture: 9 -> 128 -> 256 -> 128 -> 64 -> 3 (regression)
%
%  INPUTS
%    X               - [N x 9] raw input features
%    Y               - [N x 3] raw output values
%    model_save_path - path to save trained model (.mat)  [optional]
%
%  OUTPUTS
%    net           - trained Deep Learning Toolbox network object
%    mu_X/sigma_X  - input normalisation stats
%    mu_Y/sigma_Y  - output normalisation stats
%    train_info    - struct with loss histories and metadata

    if nargin < 3
        model_save_path = '';
    end

    fprintf('\n========================================\n');
    fprintf('  PLL Deep-Learning Model Training\n');
    fprintf('  (Deep Learning Toolbox)\n');
    fprintf('========================================\n');

    % ------------------------------------------------------------------ %
    %  Preprocessing metadata and log transforms
    % ------------------------------------------------------------------ %
    preproc.log_cols_in      = [1, 3, 4, 5, 6, 7, 9];
    preproc.log_cols_out     = [1, 3];
    preproc.idb_col          = 2;
    preproc.log_floor_input  = 1e-12;
    preproc.log_floor_output = 1e-12;
    preproc.log_floor_idb    = 1e-10;

    X_t = X;
    X_t(:, preproc.log_cols_in) = log10(abs(X(:, preproc.log_cols_in)) + preproc.log_floor_input);
    X_t(:, preproc.idb_col)     = log10(abs(X(:, preproc.idb_col)) + preproc.log_floor_idb);

    Y_t = Y;
    Y_t(:, preproc.log_cols_out) = log10(abs(Y(:, preproc.log_cols_out)) + preproc.log_floor_output);

    % ------------------------------------------------------------------ %
    %  Z-score normalisation
    % ------------------------------------------------------------------ %
    mu_X    = mean(X_t, 1);
    sigma_X = std(X_t, 0, 1) + preproc.log_floor_input;
    X_norm  = (X_t - mu_X) ./ sigma_X;

    mu_Y    = mean(Y_t, 1);
    sigma_Y = std(Y_t, 0, 1) + preproc.log_floor_output;
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
    idx_val = idx(n_train + 1 : n_train + n_val);
    idx_te  = idx(n_train + n_val + 1 : end);

    Xtr = X_norm(idx_tr, :);
    Ytr = Y_norm(idx_tr, :);
    Xva = X_norm(idx_val, :);
    Yva = Y_norm(idx_val, :);
    Xte = X_norm(idx_te, :);
    Yte = Y_norm(idx_te, :);

    fprintf('Samples -> Train: %d | Val: %d | Test: %d\n', n_train, n_val, numel(idx_te));

    % ------------------------------------------------------------------ %
    %  Network definition
    % ------------------------------------------------------------------ %
    n_in  = size(X, 2);
    n_out = size(Y, 2);

    layers = [ ...
        featureInputLayer(n_in, 'Normalization', 'none', 'Name', 'input')
        fullyConnectedLayer(128, 'Name', 'fc1', 'WeightsInitializer', 'he')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(256, 'Name', 'fc2', 'WeightsInitializer', 'he')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(128, 'Name', 'fc3', 'WeightsInitializer', 'he')
        reluLayer('Name', 'relu3')
        fullyConnectedLayer(64, 'Name', 'fc4', 'WeightsInitializer', 'he')
        reluLayer('Name', 'relu4')
        fullyConnectedLayer(n_out, 'Name', 'fc_out', 'WeightsInitializer', 'he')
        regressionLayer('Name', 'regression') ...
    ];

    fprintf('\nArchitecture: 9 -> 128 -> 256 -> 128 -> 64 -> 3\n');

    % ------------------------------------------------------------------ %
    %  Training options
    % ------------------------------------------------------------------ %
    max_epochs = 500;
    mini_batch = 256;
    val_freq   = max(1, floor(size(Xtr, 1) / mini_batch));

    options = trainingOptions('adam', ...
        'InitialLearnRate', 3e-4, ...
        'GradientDecayFactor', 0.9, ...
        'SquaredGradientDecayFactor', 0.999, ...
        'Epsilon', 1e-8, ...
        'L2Regularization', 1e-4, ...
        'MaxEpochs', max_epochs, ...
        'MiniBatchSize', mini_batch, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {Xva, Yva}, ...
        'ValidationFrequency', val_freq, ...
        'ValidationPatience', 30, ...
        'ExecutionEnvironment', 'auto', ...
        'Verbose', false, ...
        'Plots', 'none');

    fprintf('\nTraining with Deep Learning Toolbox ...\n\n');

    [net, info] = trainNetwork(Xtr, Ytr, layers, options);

    % ------------------------------------------------------------------ %
    %  Test-set performance
    % ------------------------------------------------------------------ %
    Yte_hat = predict(net, Xte, 'ExecutionEnvironment', 'auto');

    Yte_pred_t = (Yte_hat .* sigma_Y) + mu_Y;
    Yte_true_t = (Yte    .* sigma_Y) + mu_Y;

    Y_pred_raw = Yte_pred_t;
    Y_true_raw = Yte_true_t;
    Y_pred_raw(:, preproc.log_cols_out) = 10 .^ Yte_pred_t(:, preproc.log_cols_out);
    Y_true_raw(:, preproc.log_cols_out) = 10 .^ Yte_true_t(:, preproc.log_cols_out);

    output_names = {'LockTime_s', 'PhaseNoise_dBcHz', 'Fout_Hz'};
    fprintf('\n--- Test-Set Performance ---\n');
    for k = 1:3
        ss_res = sum((Y_true_raw(:, k) - Y_pred_raw(:, k)).^2);
        ss_tot = sum((Y_true_raw(:, k) - mean(Y_true_raw(:, k))).^2);
        r2     = 1 - ss_res / max(ss_tot, preproc.log_floor_output);
        rmse   = sqrt(mean((Y_true_raw(:, k) - Y_pred_raw(:, k)).^2));
        fprintf('  %-22s  R^2 = %.4f   RMSE = %.4e\n', output_names{k}, r2, rmse);
    end

    % Pack training info
    train_loss = info.TrainingLoss;
    val_loss   = info.ValidationLoss;
    train_loss = train_loss(~isnan(train_loss));
    val_loss   = val_loss(~isnan(val_loss));

    train_info.TrainingLoss   = train_loss(:)';
    train_info.ValidationLoss = val_loss(:)';
    train_info.log_cols_X     = preproc.log_cols_in;
    train_info.log_cols_Y     = preproc.log_cols_out;
    train_info.preprocessing  = preproc;
    train_info.split.train_idx = idx_tr;
    train_info.split.val_idx   = idx_val;
    train_info.split.test_idx  = idx_te;
    train_info.framework = 'Deep Learning Toolbox';
    train_info.training_info = info;

    % ------------------------------------------------------------------ %
    %  Save model
    % ------------------------------------------------------------------ %
    log_cols_in  = preproc.log_cols_in;   %#ok<NASGU>
    log_cols_out = preproc.log_cols_out;  %#ok<NASGU>

    if ~isempty(model_save_path)
        save_dir = fileparts(model_save_path);
        if ~isempty(save_dir) && ~exist(save_dir, 'dir')
            mkdir(save_dir);
        end
        save(model_save_path, 'net', 'mu_X', 'sigma_X', 'mu_Y', 'sigma_Y', ...
            'log_cols_in', 'log_cols_out', 'preproc', 'output_names', 'train_info');
        fprintf('\nModel saved to: %s\n', model_save_path);
    end
end
