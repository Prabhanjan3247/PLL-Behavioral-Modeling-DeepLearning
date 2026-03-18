function plot_results(Y_true, Y_pred, output_names, save_dir, train_info)
%PLOT_RESULTS   Scatter-and-regression plots for predicted vs true values.
%
%  Y_true     – [N × 3]  ground-truth outputs
%  Y_pred     – [N × 3]  model predictions
%  output_names – 1×3 cell of strings (metric labels)
%  save_dir   – folder to save the figures (created if necessary)
%  train_info – optional struct with TrainingLoss/ValidationLoss

    if nargin < 4 || isempty(save_dir)
        save_dir = fullfile(fileparts(mfilename('fullpath')), ...
                            '..', '..', 'results', 'figures');
    end
    if ~exist(save_dir, 'dir'), mkdir(save_dir); end

    n_out = size(Y_true, 2);

    % ------------------------------------------------------------------ %
    %  Per-output scatter plot
    % ------------------------------------------------------------------ %
    for k = 1:n_out
        yt = Y_true(:, k);
        yp = Y_pred(:, k);

        fig = figure('Visible', 'off');
        scatter(yt, yp, 18, 'filled', 'MarkerFaceAlpha', 0.5);
        hold on;
        lo = min([yt; yp]); hi = max([yt; yp]);
        plot([lo hi], [lo hi], 'r--', 'LineWidth', 1.5);
        xlabel(['True – ', output_names{k}]);
        ylabel(['Predicted – ', output_names{k}]);
        title(['Predicted vs True: ', output_names{k}]);
        grid on; axis tight;

        % R² annotation
        ss_res = sum((yt - yp).^2);
        ss_tot = sum((yt - mean(yt)).^2);
        r2 = 1 - ss_res / max(ss_tot, 1e-12);
        text(lo + 0.05*(hi-lo), hi - 0.1*(hi-lo), ...
             sprintf('R^2 = %.4f', r2), 'FontSize', 10, 'Color', 'k');

        fname = fullfile(save_dir, sprintf('scatter_%s.png', ...
                sanitise(output_names{k})));
        saveas(fig, fname);
        close(fig);
        fprintf('  Saved: %s\n', fname);
    end

    % ------------------------------------------------------------------ %
    %  Residual histogram (all outputs combined)
    % ------------------------------------------------------------------ %
    fig = figure('Visible', 'off');
    residuals = Y_pred - Y_true;
    for k = 1:n_out
        subplot(1, n_out, k);
        histogram(residuals(:, k), 30, 'Normalization', 'pdf');
        xlabel(output_names{k});
        title(sprintf('Residuals – %s', output_names{k}));
        grid on;
    end
    sgtitle('Prediction Residuals (predicted – true)');
    saveas(fig, fullfile(save_dir, 'residuals_histogram.png'));
    close(fig);

    % ------------------------------------------------------------------ %
    %  Training error curve (if supplied)
    % ------------------------------------------------------------------ %
    if nargin >= 5 && isstruct(train_info) && isfield(train_info, 'TrainingLoss')
        fig = figure('Visible', 'off');
        semilogy(train_info.TrainingLoss, 'b-', 'LineWidth', 1.5); hold on;
        if isfield(train_info, 'ValidationLoss')
            semilogy(train_info.ValidationLoss, 'r--', 'LineWidth', 1.5);
            legend('Training', 'Validation');
        end
        xlabel('Epoch'); ylabel('MSE Loss');
        title('Training / Validation Loss Curve');
        grid on;
        saveas(fig, fullfile(save_dir, 'training_loss.png'));
        close(fig);
    end

    fprintf('All plots saved to: %s\n', save_dir);
end

% ------------------------------------------------------------------ %
function s = sanitise(s)
    s = regexprep(s, '[^a-zA-Z0-9_]', '_');
end
