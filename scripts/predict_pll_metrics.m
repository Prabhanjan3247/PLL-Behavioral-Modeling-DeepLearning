function predictions = predict_pll_metrics(params_input, model_path)
%PREDICT_PLL_METRICS  Predict PLL performance using the trained DL model.
%
%  INPUTS
%    params_input – struct with fields: fref, Idb, Icp, Ileak, R, C,
%                   Kvco, PN_vco, N   (single sample)
%                   OR [M × 9] numeric matrix (batch of samples)
%    model_path   – path to saved model .mat
%                   (default: <project_root>/models/pll_dl_model.mat)
%
%  OUTPUT
%    predictions – struct with fields:
%      .LockTime_s        [s]
%      .PhaseNoise_dBcHz  [dBc/Hz @ 1 MHz]
%      .Fout_Hz           [Hz]

    % ---- Locate model ------------------------------------------------- %
    if nargin < 2 || isempty(model_path)
        script_dir = fileparts(mfilename('fullpath'));
        model_path = fullfile(script_dir, '..', 'models', 'pll_dl_model.mat');
    end

    if ~isfile(model_path)
        error('Model file not found:\n  %s\nRun main.m first to train the model.', ...
              model_path);
    end

    mdl     = load(model_path);
    net     = mdl.net;
    mu_X    = mdl.mu_X;
    sigma_X = mdl.sigma_X;
    mu_Y    = mdl.mu_Y;
    sigma_Y = mdl.sigma_Y;

    log_cols_Y = [1, 3];

    % ---- Parse inputs ------------------------------------------------- %
    if isstruct(params_input)
        fields = {'fref','Idb','Icp','Ileak','R','C','Kvco','PN_vco','N'};
        X_raw = zeros(1, 9);
        for k = 1:9
            X_raw(1, k) = params_input.(fields{k});
        end
    elseif isnumeric(params_input)
        X_raw = params_input;
        if size(X_raw, 2) ~= 9
            error('Expected 9 columns; got %d.', size(X_raw, 2));
        end
    else
        error('params_input must be a struct or [M × 9] numeric matrix.');
    end

    % ---- Pre-process -------------------------------------------------- %
    X_t = X_raw;
    X_t(:, [1,3,4,5,6,7,9]) = log10(abs(X_raw(:, [1,3,4,5,6,7,9])) + eps);
    X_t(:, 2)                = log10(abs(X_raw(:, 2)) + 1e-10);
    X_norm = (X_t - mu_X) ./ sigma_X;

    % ---- Predict ------------------------------------------------------ %
    Y_pred_norm = nn_predict(net, X_norm')';    % M × 3

    % ---- Denormalise -------------------------------------------------- %
    Y_pred_t   = Y_pred_norm .* sigma_Y + mu_Y;
    Y_pred_raw = Y_pred_t;
    Y_pred_raw(:, log_cols_Y) = 10 .^ Y_pred_t(:, log_cols_Y);

    % ---- Pack outputs ------------------------------------------------- %
    predictions.LockTime_s       = Y_pred_raw(:, 1);
    predictions.PhaseNoise_dBcHz = Y_pred_raw(:, 2);
    predictions.Fout_Hz          = Y_pred_raw(:, 3);

    % ---- Display ------------------------------------------------------ %
    M = size(X_raw, 1);
    fprintf('\n===== PLL Metric Predictions (%d sample(s)) =====\n', M);
    for i = 1:M
        fprintf('  Lock Time   : %.4e s\n',      predictions.LockTime_s(i));
        fprintf('  Phase Noise : %.2f dBc/Hz\n', predictions.PhaseNoise_dBcHz(i));
        fprintf('  Output Freq : %.4e Hz  (%.3f MHz)\n', ...
                predictions.Fout_Hz(i), predictions.Fout_Hz(i)/1e6);
    end

    % ---- Cross-check with analytical model (struct input only) -------- %
    if isstruct(params_input)
        fprintf('\n  [Analytical cross-check]\n');
        [lt, pn, fo] = pll_behavioral_model(params_input);
        fprintf('  Lock Time   : %.4e s\n', lt);
        fprintf('  Phase Noise : %.2f dBc/Hz\n', pn);
        fprintf('  Output Freq : %.4e Hz\n', fo);
    end
end
