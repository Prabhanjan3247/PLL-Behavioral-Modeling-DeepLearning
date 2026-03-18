function metrics = evaluate_model(net, X, Y, mu_X, sigma_X, mu_Y, sigma_Y, output_names, preproc, train_info)
%EVALUATE_MODEL  Compute regression metrics on a dataset.
%
%  INPUTS
%    net, mu_X, sigma_X, mu_Y, sigma_Y – as returned by train_pll_model
%    X         – [N × 9]  raw input samples
%    Y         – [N × 3]  raw output samples
%    output_names – 1×3 cell of metric labels (optional)

    if nargin < 9 || isempty(preproc)
        preproc.log_cols_in      = [1, 3, 4, 5, 6, 7, 9];
        preproc.log_cols_out     = [1, 3];
        preproc.idb_col          = 2;
        preproc.log_floor_input  = 1e-12;
        preproc.log_floor_output = 1e-12;
        preproc.log_floor_idb    = 1e-10;
    end

    if nargin < 10
        train_info = [];
    end

    if nargin < 8 || isempty(output_names)
        output_names = {'LockTime_s', 'PhaseNoise_dBcHz', 'Fout_Hz'};
    end

    log_cols_X = preproc.log_cols_in;
    log_cols_Y = preproc.log_cols_out;

    % ---- Pre-process inputs ------------------------------------------ %
    X_t = X;
    X_t(:, log_cols_X)      = log10(abs(X(:, log_cols_X)) + preproc.log_floor_input);
    X_t(:, preproc.idb_col) = log10(abs(X(:, preproc.idb_col)) + preproc.log_floor_idb);
    X_norm = (X_t - mu_X) ./ sigma_X;

    % ---- Forward pass ------------------------------------------------- %
    Y_pred_norm = nn_predict(net, X_norm')';      % N × 3

    % ---- Denormalise -------------------------------------------------- %
    Y_pred_t   = Y_pred_norm .* sigma_Y + mu_Y;
    Y_pred_raw = Y_pred_t;
    Y_pred_raw(:, log_cols_Y) = 10 .^ Y_pred_t(:, log_cols_Y);

    Y_true_raw = Y;   % raw Y is already in original (physical) units

    % ---- Regression metrics ------------------------------------------ %
    n_out = size(Y, 2);
    metrics = struct();

    fprintf('\n==============================\n');
    fprintf('       Model Evaluation\n');
    fprintf('==============================\n');
    fprintf('%-22s  %8s  %12s  %10s  %10s\n', 'Metric', 'R²', 'RMSE', 'MAE', 'MAPE(%)');
    fprintf('%s\n', repmat('-', 1, 68));

    for k = 1:n_out
        yt = Y_true_raw(:, k);
        yp = Y_pred_raw(:, k);

        ss_res = sum((yt - yp).^2);
        ss_tot = sum((yt - mean(yt)).^2);
        r2   = 1 - ss_res / max(ss_tot, preproc.log_floor_output);
        rmse = sqrt(mean((yt - yp).^2));
        mae  = mean(abs(yt - yp));
        mape = mean(abs((yt - yp) ./ (abs(yt) + preproc.log_floor_output))) * 100;

        fname = output_names{k};
        metrics.(fname).R2   = r2;
        metrics.(fname).RMSE = rmse;
        metrics.(fname).MAE  = mae;
        metrics.(fname).MAPE = mape;

        fprintf('%-22s  %8.4f  %12.4e  %10.4e  %10.2f\n', ...
                fname, r2, rmse, mae, mape);
    end
    fprintf('%s\n', repmat('-', 1, 68));

    % ---- Save results ------------------------------------------------ %
    results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
    if ~exist(results_dir, 'dir'), mkdir(results_dir); end
    save(fullfile(results_dir, 'evaluation_results.mat'), ...
         'Y_true_raw', 'Y_pred_raw', 'metrics', 'output_names');
    fprintf('Evaluation results saved to: %s\n', results_dir);

    % ---- Plots ------------------------------------------------------- %
    fig_dir = fullfile(results_dir, 'figures');
    plot_results(Y_true_raw, Y_pred_raw, output_names, fig_dir, train_info);
end
