function [X, Y, feature_names, output_names] = generate_pll_data(N_samples, save_path)
%GENERATE_PLL_DATA  Monte-Carlo sweep of PLL input parameters.
%
%  Runs the PLL behavioral model with N_samples randomised parameter sets
%  and returns the resulting input feature matrix X and output matrix Y.
%
%  INPUTS
%    N_samples  – Number of Monte-Carlo samples (default 5000)
%    save_path  – Path to save the .mat dataset (optional)
%
%  OUTPUTS
%    X            – [N_samples × 9]  input feature matrix  (raw values)
%    Y            – [N_samples × 3]  output matrix (raw values)
%    feature_names – cell array of feature names
%    output_names  – cell array of output names

    if nargin < 1 || isempty(N_samples), N_samples = 5000; end
    if nargin < 2, save_path = ''; end

    rng(42);   % reproducibility

    % ------------------------------------------------------------------ %
    %  Parameter ranges (physically motivated)
    % ------------------------------------------------------------------ %
    %   fref   : 1 MHz – 200 MHz
    %   Idb    : 0     – 50 µA
    %   Icp    : 100 µA – 5 mA
    %   Ileak  : 0.1 nA – 10 nA
    %   R      : 1 kΩ  – 100 kΩ
    %   C      : 10 pF – 10 nF
    %   Kvco   : 100 MHz/V – 2 GHz/V
    %   PN_vco : -120 – -80 dBc/Hz @ 1 MHz
    %   N      : 10 – 1000

    % Log-uniform distributions for quantities spanning decades
    log_fref  = log10([1e6,  200e6]);
    log_Icp   = log10([100e-6, 5e-3]);
    log_Ileak = log10([0.1e-9, 10e-9]);
    log_R     = log10([1e3,   100e3]);
    log_C     = log10([10e-12, 10e-9]);
    log_Kvco  = log10([100e6, 2000e6]);
    log_N     = log10([10, 1000]);

    fref  = 10 .^ (log_fref(1)  + rand(N_samples,1)*(log_fref(2)  - log_fref(1)));
    Icp   = 10 .^ (log_Icp(1)   + rand(N_samples,1)*(log_Icp(2)   - log_Icp(1)));
    Ileak = 10 .^ (log_Ileak(1) + rand(N_samples,1)*(log_Ileak(2) - log_Ileak(1)));
    R     = 10 .^ (log_R(1)     + rand(N_samples,1)*(log_R(2)     - log_R(1)));
    C     = 10 .^ (log_C(1)     + rand(N_samples,1)*(log_C(2)     - log_C(1)));
    Kvco  = 10 .^ (log_Kvco(1)  + rand(N_samples,1)*(log_Kvco(2)  - log_Kvco(1)));
    N_div = round(10 .^ (log_N(1) + rand(N_samples,1)*(log_N(2) - log_N(1))));

    % Uniform distributions
    Idb    = rand(N_samples, 1) * 50e-6;           % 0 – 50 µA
    PN_vco = -120 + rand(N_samples, 1) * 40;        % -120 – -80 dBc/Hz

    % ------------------------------------------------------------------ %
    %  Simulate every sample
    % ------------------------------------------------------------------ %
    X = [fref, Idb, Icp, Ileak, R, C, Kvco, PN_vco, double(N_div)];
    Y = zeros(N_samples, 3);

    fprintf('Generating %d PLL simulation samples ...\n', N_samples);
    t_start = tic;

    for i = 1:N_samples
        p.fref   = fref(i);
        p.Idb    = Idb(i);
        p.Icp    = Icp(i);
        p.Ileak  = Ileak(i);
        p.R      = R(i);
        p.C      = C(i);
        p.Kvco   = Kvco(i);
        p.PN_vco = PN_vco(i);
        p.N      = N_div(i);

        [lt, pn, fo] = pll_behavioral_model(p);

        % Clamp to physical limits
        lt = max(min(lt, 1e-2), 1e-9);   % 1 ns – 10 ms
        pn = max(min(pn, -40), -180);     % dBc/Hz range (realistic PLL: -40 to -180)

        Y(i, :) = [lt, pn, fo];

        if mod(i, 500) == 0
            fprintf('  %d / %d samples done  (%.1f s elapsed)\n', ...
                    i, N_samples, toc(t_start));
        end
    end

    % Remove any rows with NaN/Inf
    valid = all(isfinite(X), 2) & all(isfinite(Y), 2);
    X = X(valid, :);
    Y = Y(valid, :);
    fprintf('Valid samples: %d / %d\n', sum(valid), N_samples);

    % ------------------------------------------------------------------ %
    %  Metadata
    % ------------------------------------------------------------------ %
    feature_names = {'fref_Hz', 'Idb_A', 'Icp_A', 'Ileak_A', ...
                     'R_Ohm', 'C_F', 'Kvco_Hz_per_V', 'PN_vco_dBcHz', 'N'};
    output_names  = {'LockTime_s', 'PhaseNoise_dBcHz', 'Fout_Hz'};

    fprintf('Data generation complete.\n');
    fprintf('  X: %d x %d   Y: %d x %d\n', size(X), size(Y));

    % ------------------------------------------------------------------ %
    %  Optional save
    % ------------------------------------------------------------------ %
    if ~isempty(save_path)
        if ~exist(fileparts(save_path), 'dir') && ~isempty(fileparts(save_path))
            mkdir(fileparts(save_path));
        end
        save(save_path, 'X', 'Y', 'feature_names', 'output_names');
        fprintf('Dataset saved to: %s\n', save_path);
    end
end
