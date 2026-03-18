function [lock_time, phase_noise_out, f_out, status] = pll_behavioral_model(params)
%PLL_BEHAVIORAL_MODEL  Compute PLL performance metrics from circuit parameters.
%
%  Implements a 2nd-order Type-II charge-pump PLL with a passive lead-lag
%  loop filter.  All equations are derived from Gardner's PLL textbook.
%
%  INPUT  params – struct with fields:
%    fref    Reference (input) clock frequency       [Hz]
%    Idb     PFD dead-band compensation current      [A]
%    Icp     Charge-pump input current               [A]
%    Ileak   Charge-pump leakage current             [A]
%    R       Loop-filter series resistance           [Ohm]
%    C       Loop-filter capacitance                 [F]
%    Kvco    VCO voltage sensitivity                 [Hz/V]
%    PN_vco  VCO phase noise at 1 MHz offset         [dBc/Hz]
%    N       Clock-divider integer value             [-]
%
%  OUTPUT
%    lock_time       Estimated PLL lock time         [s]
%    phase_noise_out Output phase noise @ 1 MHz off  [dBc/Hz]
%    f_out           Output (VCO) frequency          [Hz]
%    status          struct with validity metadata

    status.isValid = true;
    status.reason = '';

    lock_time = NaN;
    phase_noise_out = NaN;
    f_out = NaN;

    fref    = params.fref;
    Icp     = params.Icp;
    Ileak   = params.Ileak;
    Idb     = params.Idb;
    R       = params.R;
    C       = params.C;
    Kvco_Hz = params.Kvco;       % Hz/V  → convert below
    PN_vco  = params.PN_vco;     % dBc/Hz @ 1 MHz offset
    N       = params.N;

    % Basic physical sanity checks
    if any(~isfinite([fref, Icp, Ileak, Idb, R, C, Kvco_Hz, PN_vco, N]))
        status.isValid = false;
        status.reason = 'Non-finite parameter detected.';
        return
    end

    if fref <= 0 || Icp <= 0 || R <= 0 || C <= 0 || Kvco_Hz <= 0 || N <= 0
        status.isValid = false;
        status.reason = 'Non-positive physical parameter detected.';
        return
    end

    % ------------------------------------------------------------------ %
    %  Derived quantities
    % ------------------------------------------------------------------ %
    Kvco  = 2 * pi * Kvco_Hz;      % rad/s/V

    f_out = N * fref;              % VCO output frequency  [Hz]

    % ------------------------------------------------------------------ %
    %  Loop parameters (2nd-order Type-II PLL)
    % ------------------------------------------------------------------ %
    % Natural frequency
    omega_n = sqrt((Icp * Kvco) / (2 * pi * N * C));

    % Damping factor  ζ = (R·C·ωn) / 2
    zeta = (R * C * omega_n) / 2;

    % Guard against degenerate cases
    if omega_n <= 0 || zeta <= 0
        status.isValid = false;
        status.reason = 'Degenerate loop dynamics (omega_n<=0 or zeta<=0).';
        return
    end

    % Loop bandwidth (–3 dB of closed-loop magnitude, rad/s)
    omega_c = omega_n * (2 * zeta + sqrt(4 * zeta^2 + 1));

    % ------------------------------------------------------------------ %
    %  Lock time (Gardner estimate for 1 % settling after a freq. step)
    % ------------------------------------------------------------------ %
    T_lock_phase = 4.6 / (zeta * omega_n);

    % Extra settling due to CP leakage (current mismatch pulls the filter)
    T_lock_leak  = (Ileak / Icp) * (10 / omega_n);

    % Dead-band compensation effect on first-cycle phase error
    T_lock_db    = (Idb  / Icp) * (5  / omega_n);

    lock_time = max(T_lock_phase + T_lock_leak + T_lock_db, 1e-12);

    % ------------------------------------------------------------------ %
    %  Phase noise at the output @ 1 MHz offset
    % ------------------------------------------------------------------ %
    f_offset     = 1e6;                % Hz
    omega_offset = 2 * pi * f_offset;
    jw           = 1j * omega_offset;

    % Loop-filter transfer function  F(s) = (1 + s·R·C) / (s·C)
    H_filter = (1 + jw * R * C) / (jw * C);

    % Open-loop gain  G(s) = [Icp/(2π)] · F(s) · [Kvco/s] · (1/N)
    G_open = (Icp / (2 * pi)) * H_filter * (Kvco / jw) / N;

    % Closed-loop noise transfer functions
    H_ref_to_out = N * G_open / (1 + G_open);   % reference phase → output phase
    H_vco_to_out = 1        / (1 + G_open);      % VCO phase       → output phase

    % ---- Reference oscillator phase noise at 1 MHz offset (dBc/Hz) ---
    % Crystal/TCXO thermal floor: ~−160 dBc/Hz @ 1 MHz for a 10 MHz ref.
    % Degrades ~10 dB per decade as carrier frequency rises (lower resonator Q).
    PN_ref = -160 + 10 * log10(max(fref, 1e6) / 10e6);

    % ---- Charge-pump shot noise (correct formula, negligible contributor) --
    % S_I = 2·q·Icp  [A²/Hz]
    % Transfer: S_phi_cp = S_I · |Z_LF|² · (Kvco/ω)² / N² / |1+G|²  [rad²/Hz]
    q           = 1.6e-19;                          % electron charge [C]
    S_I         = 2 * q * Icp;                      % shot-noise PSD  [A²/Hz]
    Z_LF_mag2   = abs(H_filter)^2;                  % |Z_LF|²          [Ω²]
    H_cp_phase  = sqrt(S_I * Z_LF_mag2) * (Kvco / omega_offset) / N / abs(1 + G_open);
    PN_cp_dBcHz = 10 * log10(max(H_cp_phase^2 / 2, 1e-30));

    % ---- Sum contributions (linear domain, then convert back to dBc/Hz) ---
    PN_ref_contrib = PN_ref + 20 * log10(max(abs(H_ref_to_out), 1e-15));
    PN_vco_contrib = PN_vco + 20 * log10(max(abs(H_vco_to_out), 1e-15));

    PN_total = 10^(PN_ref_contrib / 10) + ...
               10^(PN_vco_contrib / 10) + ...
               10^(PN_cp_dBcHz   / 10);

    phase_noise_out = 10 * log10(max(PN_total, 1e-30));

    if ~isfinite(lock_time) || ~isfinite(phase_noise_out) || ~isfinite(f_out)
        status.isValid = false;
        status.reason = 'Non-finite output computed.';
    end
end
