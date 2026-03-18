%% MAIN.M — Behavioral Modeling of PLL using Deep Learning
%  =========================================================
%  Capstone Project
%  Topic  : Behavioral Modeling of Phase-Locked Loop (PLL)
%           using Deep Learning Techniques
%  =========================================================
%
%  Pipeline:
%    1. Generate PLL training data via the analytical behavioral model
%    2. Train a deep feedforward network (Deep Learning Toolbox)
%    3. Evaluate model accuracy on a held-out test set
%    4. Demonstrate real-time prediction with example PLL specifications
%
%  Requirements:
%    - MATLAB R2019b or later
%    - Deep Learning Toolbox
%
%  Usage:
%    Simply run this script.  All outputs are saved to:
%      data/    – generated dataset (.mat)
%      models/  – trained network (.mat)
%      results/ – evaluation metrics & figures

clc; clear; close all;

% Set to true to re-generate data and re-train even if .mat files exist.
% Useful when the behavioral model has been updated.
FORCE_REGENERATE = false;   % set true to force re-generation and re-training

%% -----------------------------------------------------------------------
%  0.  Path setup
%% -----------------------------------------------------------------------
project_root = fileparts(mfilename('fullpath'));
addpath(fullfile(project_root, 'scripts'));
addpath(fullfile(project_root, 'scripts', 'utils'));

data_path  = fullfile(project_root, 'data',   'pll_dataset.mat');
model_path = fullfile(project_root, 'models', 'pll_dl_model.mat');

fprintf('=================================================\n');
fprintf('  PLL Deep-Learning Behavioral Model\n');
fprintf('=================================================\n\n');

%% -----------------------------------------------------------------------
%  1.  Generate (or load) dataset
%% -----------------------------------------------------------------------
N_SAMPLES = 6000;   % increase for better accuracy (at cost of time)

if isfile(data_path) && ~FORCE_REGENERATE
    fprintf('[Step 1]  Loading existing dataset: %s\n', data_path);
    load(data_path, 'X', 'Y', 'feature_names', 'output_names');
    fprintf('          Loaded %d samples.\n', size(X, 1));
else
    fprintf('[Step 1]  Generating %d simulation samples …\n', N_SAMPLES);
    [X, Y, feature_names, output_names] = generate_pll_data(N_SAMPLES, data_path);
end

% Print dataset summary
fprintf('\nDataset summary:\n');
fprintf('  Inputs  (%d features): %s\n', numel(feature_names), ...
        strjoin(feature_names, ', '));
fprintf('  Outputs (%d targets) : %s\n', numel(output_names),  ...
        strjoin(output_names, ', '));
fprintf('  Total samples : %d\n\n', size(X,1));

%% -----------------------------------------------------------------------
%  2.  Train the deep learning model
%% -----------------------------------------------------------------------
if isfile(model_path) && ~FORCE_REGENERATE
    fprintf('[Step 2]  Loading existing model: %s\n', model_path);
        mdl = load(model_path);
        net = mdl.net;
        mu_X = mdl.mu_X;
        sigma_X = mdl.sigma_X;
        mu_Y = mdl.mu_Y;
        sigma_Y = mdl.sigma_Y;
        if isfield(mdl, 'train_info')
                train_info = mdl.train_info;
        else
                train_info = struct();
        end
        if isfield(mdl, 'preproc')
                preproc = mdl.preproc;
        else
                preproc.log_cols_in      = [1, 3, 4, 5, 6, 7, 9];
                preproc.log_cols_out     = [1, 3];
                preproc.idb_col          = 2;
                preproc.log_floor_input  = 1e-12;
                preproc.log_floor_output = 1e-12;
                preproc.log_floor_idb    = 1e-10;
        end

        % Enforce Deep Learning Toolbox model for compliance
        if isstruct(net) && isfield(net, 'W')
                fprintf('          Legacy custom model detected. Retraining with Deep Learning Toolbox...\n\n');
                [net, mu_X, sigma_X, mu_Y, sigma_Y, train_info] = train_pll_model(X, Y, model_path);
                preproc.log_cols_in      = [1, 3, 4, 5, 6, 7, 9];
                preproc.log_cols_out     = [1, 3];
                preproc.idb_col          = 2;
                preproc.log_floor_input  = 1e-12;
                preproc.log_floor_output = 1e-12;
                preproc.log_floor_idb    = 1e-10;
        else
                fprintf('          Model loaded.\n\n');
        end
else
    fprintf('[Step 2]  Training deep neural network …\n\n');
    [net, mu_X, sigma_X, mu_Y, sigma_Y, train_info] = train_pll_model(X, Y, model_path);
        preproc.log_cols_in      = [1, 3, 4, 5, 6, 7, 9];
        preproc.log_cols_out     = [1, 3];
        preproc.idb_col          = 2;
        preproc.log_floor_input  = 1e-12;
        preproc.log_floor_output = 1e-12;
        preproc.log_floor_idb    = 1e-10;
end

%% -----------------------------------------------------------------------
%  3.  Evaluate on full dataset (model internally holds split info)
%% -----------------------------------------------------------------------
fprintf('\n[Step 3]  Evaluating model …\n');
metrics = evaluate_model(net, X, Y, mu_X, sigma_X, mu_Y, sigma_Y, output_names, preproc, train_info);

%% -----------------------------------------------------------------------
%  4.  Example predictions with new PLL specifications
%% -----------------------------------------------------------------------
fprintf('\n[Step 4]  Predicting for example PLL designs …\n');

% --- Example 1: Low-power 2.4 GHz RF PLL ------------------------------ %
ex1.fref    = 20e6;      % 20 MHz reference
ex1.Idb     = 5e-6;      % 5 µA dead-band compensation
ex1.Icp     = 500e-6;    % 500 µA charge pump
ex1.Ileak   = 1e-9;      % 1 nA leakage
ex1.R       = 20e3;      % 20 kΩ
ex1.C       = 200e-12;   % 200 pF
ex1.Kvco    = 500e6;     % 500 MHz/V
ex1.PN_vco  = -110;      % -110 dBc/Hz @ 1 MHz
ex1.N       = 120;       % Fout = 2.4 GHz

fprintf('\n--- Example 1: 2.4 GHz RF PLL ---\n');
pred1 = predict_pll_metrics(ex1, model_path);

% --- Example 2: High-frequency 5G mmWave PLL -------------------------- %
ex2.fref    = 100e6;     % 100 MHz reference
ex2.Idb     = 10e-6;     % 10 µA
ex2.Icp     = 2e-3;      % 2 mA charge pump
ex2.Ileak   = 5e-9;      % 5 nA leakage
ex2.R       = 5e3;       % 5 kΩ
ex2.C       = 50e-12;    % 50 pF
ex2.Kvco    = 1500e6;    % 1.5 GHz/V
ex2.PN_vco  = -95;       % -95 dBc/Hz @ 1 MHz
ex2.N       = 280;       % Fout = 28 GHz

fprintf('\n--- Example 2: 28 GHz mmWave PLL ---\n');
pred2 = predict_pll_metrics(ex2, model_path);

% --- Example 3: Low-jitter clock-generation PLL ----------------------- %
ex3.fref    = 50e6;      % 50 MHz
ex3.Idb     = 2e-6;
ex3.Icp     = 1e-3;
ex3.Ileak   = 0.5e-9;
ex3.R       = 10e3;
ex3.C       = 100e-12;
ex3.Kvco    = 200e6;
ex3.PN_vco  = -118;
ex3.N       = 20;        % Fout = 1 GHz

fprintf('\n--- Example 3: 1 GHz Clock Synthesis PLL ---\n');
pred3 = predict_pll_metrics(ex3, model_path);

%% -----------------------------------------------------------------------
%  5.  Display comparison table
%% -----------------------------------------------------------------------
fprintf('\n\n==================================================================\n');
fprintf('                  Prediction Summary Table\n');
fprintf('==================================================================\n');
fprintf('%-28s  %12s  %18s  %14s\n', 'Design', 'Lock Time (µs)', ...
        'Phase Noise (dBc/Hz)', 'Fout (MHz)');
fprintf('%s\n', repmat('-', 1, 78));
fprintf('%-28s  %12.3f  %18.2f  %14.2f\n', ...
        'Ex1: 2.4 GHz RF PLL',   pred1.LockTime_s*1e6,       pred1.PhaseNoise_dBcHz,  pred1.Fout_Hz/1e6);
fprintf('%-28s  %12.3f  %18.2f  %14.2f\n', ...
        'Ex2: 28 GHz mmWave PLL', pred2.LockTime_s*1e6,      pred2.PhaseNoise_dBcHz,  pred2.Fout_Hz/1e6);
fprintf('%-28s  %12.3f  %18.2f  %14.2f\n', ...
        'Ex3: 1 GHz Clock Gen',  pred3.LockTime_s*1e6,       pred3.PhaseNoise_dBcHz,  pred3.Fout_Hz/1e6);
fprintf('%s\n', repmat('=', 1, 78));

%% -----------------------------------------------------------------------
%  6.  Feature importance (sensitivity analysis)
%% -----------------------------------------------------------------------
fprintf('\n[Step 5]  Sensitivity analysis on trained model …\n');

mdl_sens = load(model_path);
fields9  = {'fref','Idb','Icp','Ileak','R','C','Kvco','PN_vco','N'};

if isfield(mdl_sens, 'preproc')
        preproc_sens = mdl_sens.preproc;
else
        preproc_sens = preproc;
end

% Inline normaliser: struct → normalised row vector
base_params = ex1;
base_xr = [base_params.fref, base_params.Idb, base_params.Icp, ...
           base_params.Ileak, base_params.R,   base_params.C, ...
           base_params.Kvco,  base_params.PN_vco, base_params.N];
base_xt = base_xr;
base_xt(preproc_sens.log_cols_in) = log10(abs(base_xr(preproc_sens.log_cols_in)) + preproc_sens.log_floor_input);
base_xt(preproc_sens.idb_col) = log10(abs(base_xr(preproc_sens.idb_col)) + preproc_sens.log_floor_idb);
base_xn  = (base_xt - mdl_sens.mu_X) ./ mdl_sens.sigma_X;
base_out = nn_predict(mdl_sens.net, base_xn')';
base_lt  = 10^(base_out(1)*mdl_sens.sigma_Y(1) + mdl_sens.mu_Y(1));

sensitivity = zeros(1, 9);
fprintf('  Feature importance (±10%% perturbation → lock-time change):\n');
for k = 1:9
    % +10% perturbation
    xp = base_xr;  xp(k) = base_xr(k) * 1.10;
    xtp = xp;
        xtp(preproc_sens.log_cols_in) = log10(abs(xp(preproc_sens.log_cols_in)) + preproc_sens.log_floor_input);
        xtp(preproc_sens.idb_col) = log10(abs(xp(preproc_sens.idb_col)) + preproc_sens.log_floor_idb);
    op = nn_predict(mdl_sens.net, ((xtp - mdl_sens.mu_X)./mdl_sens.sigma_X)')';

    % -10% perturbation
    xm = base_xr;  xm(k) = base_xr(k) * 0.90;
    xtm = xm;
        xtm(preproc_sens.log_cols_in) = log10(abs(xm(preproc_sens.log_cols_in)) + preproc_sens.log_floor_input);
        xtm(preproc_sens.idb_col) = log10(abs(xm(preproc_sens.idb_col)) + preproc_sens.log_floor_idb);
    om = nn_predict(mdl_sens.net, ((xtm - mdl_sens.mu_X)./mdl_sens.sigma_X)')';

    lt_p = 10^(op(1)*mdl_sens.sigma_Y(1) + mdl_sens.mu_Y(1));
    lt_m = 10^(om(1)*mdl_sens.sigma_Y(1) + mdl_sens.mu_Y(1));
        sensitivity(k) = abs(lt_p - lt_m) / max(base_lt, preproc_sens.log_floor_output) * 100;
    fprintf('    %-12s : %.3f %%\n', fields9{k}, sensitivity(k));
end

% Bar chart
fig = figure('Visible', 'off');
bar(sensitivity, 'FaceColor', [0.2 0.5 0.8]);
set(gca, 'XTickLabel', fields9, 'XTick', 1:9);
xtickangle(40);
ylabel('Lock-Time Sensitivity (%)');
title('Feature Sensitivity to Lock Time Prediction');
grid on;
saveas(fig, fullfile(project_root, 'results', 'figures', 'feature_sensitivity.png'));
close(fig);

%% -----------------------------------------------------------------------
%  Done
%% -----------------------------------------------------------------------
fprintf('\n=================================================\n');
fprintf('  Project complete.  Check results/ for outputs.\n');
fprintf('=================================================\n');
