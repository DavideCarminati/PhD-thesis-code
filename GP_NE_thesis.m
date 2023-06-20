%% Noise Estimation in GP as a function
function out = GP_NE_thesis(x_train, y_train, x_test, y_test, hyp_init, hyp_guessing, surface_plot)

set(0,'defaultTextInterpreter','latex');
set(0, 'defaultAxesTickLabelInterpreter','latex');
set(0, 'defaultLegendInterpreter','latex');
rng(1);

if size(hyp_init, 2) ~= 1
    disp("hyp_init must by a 3x1 vector. Exiting...");
    return;
end

SURFACE = surface_plot;
TRAIN_POINTS = size(x_train, 1); % Servono un tot di training points per avere i giusti valori!
VERBOSE = false;

x = x_train; %100
% y_real = @(x) x .* sin(2 * x);
% y_real = @(x) 0.8 + ( x + 0.2 ) .* ( 1 - 5 ./ (1 + exp(-2 * x)));
% y_real = @(x) sin(x.^2 / 2); % Now it works! There was an error in building K_tmp - was using logged params...
% y_real = @(x) sin(x .* sqrt(abs(x)));
y = y_train;
% ampl = 1;
l0 = hyp_init(1); % initial length scale
x_star = x_test;

% Adam sembra più robusto degli altri due, ma per qualche motivo, quando
% funziona, vanilla GD ci mette meno iterazioni! In realtà FGD ci mette
% meno iterazioni, ma adam rimane il più robusto quando cambiano le
% condizioni iniziali (che è solo l in questo caso)

regr = figure();
plot(x, y, 'o-', 'LineWidth', 1);
grid on, hold on;

nu = 0.6; % Derivative order
mu = 1e-3; % learning rate
c = [ 0.01; 0.01; 0.0001 ]; % Starting point for integration
epsilon = 1e-6; % Jitter to avoid negative denom
lambda_old = log(hyp_init); % [ l, alpha, sigma ]
% lambda_old = log([ 10; 10; 1e-1 ]); % [ l, alpha, sigma ]
lambda_old_vanilla = lambda_old;
lambda_old_adam = lambda_old;
lambda_old_qn = lambda_old;
lambda_old_bfgs = lambda_old;
% sigma = sqrt(1e-3);%0.0618;
iter_max = 5000;
grad_err_tol = 1e-2; % Leggere il paper dei cinesi per capire come far spiccare il fractional in velocità... adesso va leggermente più veloce :(

l_span = linspace(0.1, 20, 20);
ampl_span = linspace(0.1, 20, 20);
[X1, X2] = ndgrid(l_span, ampl_span);

K = @(l, ampl) ampl^2 * exp(-0.5 * pdist2(x, x).^2 / l^2) + 1e-6 * eye(size(x,1), size(x,1)); % added some jitter...
% K_tilde = @(l, ampl, sigma) ampl^2 * exp(-0.5 * pdist2(x, x).^2 / l^2) + sigma^2 * eye(size(x,1), size(x,1)); % Kernel with noise
K_tilde = @(l, ampl, sigma) K(l, ampl) + sigma^2 * eye(size(x,1), size(x,1)); % Kernel with noise
K_star = @(l,ampl) ampl^2 * exp(-0.5 * pdist2(x_star, x).^2 / l^2);
K_starstar = @(l,ampl) ampl^2 * exp(-0.5 * pdist2(x_star, x_star).^2 / l^2);
dK_l = @(l,ampl) pdist2(x, x).^2 / l^3 .* K(l, ampl);
dK_ampl = @(l, ampl) 2 / ampl * K(l, ampl);
dK_sigma = @(sigma) 2 * sigma * eye(size(x,1), size(x,1));
% ddK_l = @(l) -3 * pdist2(x, x) / l^4 .* K(l) + pdist2(x, x) / l^3 .* dK_l(l);
% var_K = kernel(x, x, lambda_old(2), lambda_old(1)) + sigma^2 * eye(size(x,1), size(x,1));
% det(var_K)

NLL = @(l, ampl, sigma) 0.5 * (log(det(K_tilde(l, ampl, sigma))) +...
    y' * inv(K_tilde(l, ampl, sigma)) * y + TRAIN_POINTS*log(2 * pi));

grad_err_fig = figure();
% hold on, grid on;
% ylabel("$\| \nabla f(x) \|_\infty$");
% xlabel("Iteration")

%% Noise estimation

if hyp_guessing == true

    % eta = sigma_noise^2 / sigma_cov^2


    % Asymptote of LogLH 1st derivative
    m = 3; % Number of basis functions
    X = zeros(TRAIN_POINTS, m);
%     X = [ ones(TRAIN_POINTS, 1), x, x.^2, x.^3, x.^4, x.^5, x.^6, x.^7, x.^8 ];      % Design Matrix obtained from basis fun phi(x(1:end))
    X = [ ones(TRAIN_POINTS, 1), x, x.^2 ];      % Design Matrix obtained from basis fun phi(x(1:end))
%     % Gaussian basis function
%     mu_rbf = linspace(min(x), max(x), m);
%     for jj = 1:TRAIN_POINTS
%         X(jj, :) = [ exp(-pdist2(x(jj), mu_rbf').^2 / (2*std(y))) ];
%     end
    % Check how the selected basis function (= the priori) describes the training set
    beta_NE = X \ y;
    figure
    plot(x, y, 'b--')
    hold on, grid on;
    plot(x, X * beta_NE, 'r')
    legend('True funct', 'Using priori');
    str_NE = ['RMSE = ', num2str(sqrt(sum((y - X * beta_NE).^2) / length(y))) ];
    annotation('textbox','String', str_NE,'FitBoxToText','on');
    Q = eye(TRAIN_POINTS) - X * pinv(X);
    K_tmp = K(exp(lambda_old(1)), exp(lambda_old(2)));
    N = K_tmp * Q;
    y_tilde = y ./ sqrt(y' * Q * y); % Q-norm of vector y
    A0 = -Q * ( trace(N) / (TRAIN_POINTS - m) * eye(TRAIN_POINTS) - N);
    A1 = Q * ( trace(N*N) / (TRAIN_POINTS - m) * eye(TRAIN_POINTS) + trace(N) / (TRAIN_POINTS - m) * N - 2*N*N);
    A2 = -A1 * N;
    A3 = (A1 + A0 * N) * N*N;
    a0 = y_tilde' * A0 * y_tilde;
    a1 = y_tilde' * A1 * y_tilde;
    a2 = y_tilde' * A2 * y_tilde;
    a3 = y_tilde' * A3 * y_tilde;
    p = [ a0, a1, a2, a3 ]; % Polynomial part of asymptote of the derivative of LogLH
    roots_loglh = roots(p);
    roots_loglh = real(roots_loglh);
    roots_loglh = max(0, roots_loglh);

    % Checking if found eta roots produce a negative second derivative to find
    % the arg max
    second_der = d2LogLH_deta2(p, roots_loglh, TRAIN_POINTS, m);
    eta_opt = roots_loglh( second_der <= 0 );
    eta_opt = eta_opt(1); % Selecting only the first optimal root

    % In this case, sigma^2 is 1 (so that it doesn't modify the kernel) and
    % thus sigma0^2 = eta o forse no?

    inv_K_eta = inv(K_tmp + eye(TRAIN_POINTS) * eta_opt);
    M1_eta = inv_K_eta - inv_K_eta * X * inv(X' * inv_K_eta * X ) * X' * inv_K_eta;
    sigma_opt = 1 / (TRAIN_POINTS - m) * y' * M1_eta * y; % sigma^2 optimal
    sigma0_opt = eta_opt * sigma_opt; % sigma_noise^2 optimal

    % Update the initial parameters
    lambda_old = log([ l0; sqrt(sigma_opt); sqrt(sigma0_opt) ]);
end
    lambda_old_adam = lambda_old;
    lambda_old_vanilla = lambda_old;
    lambda_old_qn = lambda_old;
    lambda_old_bfgs = lambda_old;
    lambda_old_nm = lambda_old;

%% Using Matlab BFGS

[opt_sol, out_struct, grad_err_vec_BFGS, opt_sol_NM] = trainGP(lambda_old_bfgs, x, y);

function [opt_sol, out_struct, grad_err_vec_BFGS, opt_sol_NM] = trainGP(lambda_old_bfgs, x, y)

    options = optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient',true, ...
            'OptimalityTolerance', 0.5e-3, 'PlotFcn', @outFnc);

    grad_err_vec_BFGS = zeros(iter_max, 1);
    function stop = outFnc(x, optimValues,state)
        stop = false;
        grad_err_vec_BFGS(optimValues.iteration+1) = norm(optimValues.gradient, 'inf');
    end
    
    [opt_sol, fval, ~, out_struct] = fminunc(@cost_fun, lambda_old_bfgs, options);

    fprintf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Nelder-Mead  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n");
    options_NM = optimset('Display','iter','MaxIter', 10000);
    [opt_sol_NM, fval_NM, ~, out_struct_NM] = fminsearch(@cost_fun, lambda_old_nm, options_NM);
    disp(out_struct_NM)
    
    function [nll, dnll] = cost_fun(lambda_old)
    
        K_bfgs = @(l, ampl) ampl^2 * exp(-0.5 * pdist2(x, x).^2 / l^2) + 1e-6 * eye(size(x,1), size(x,1)); % added some jitter...
        K_tilde_bfgs = @(l, ampl, sigma) K_bfgs(l, ampl) + sigma^2 * eye(size(x,1), size(x,1)); % Kernel with noise
        dK_l_bfgs = @(l,ampl) pdist2(x, x).^2 / l^3 .* K_bfgs(l, ampl);
        dK_ampl_bfgs = @(l, ampl) 2 / ampl * K_bfgs(l, ampl);
        dK_sigma_bfgs = @(sigma) 2 * sigma * eye(size(x,1), size(x,1));
        NLL_bfgs = @(l, ampl, sigma) 0.5 * (log(det(K_tilde_bfgs(l, ampl, sigma))) +...
            y' * inv(K_tilde_bfgs(l, ampl, sigma)) * y + TRAIN_POINTS*log(2 * pi));

    % Transforming hyperparams to exp
        lambda_old = exp(lambda_old);
        K_tilde_tmp_bfgs = K_tilde_bfgs(lambda_old(1), lambda_old(2), lambda_old(3));
        dK_l_tmp_bfgs = dK_l_bfgs(lambda_old(1), lambda_old(2));
        dK_ampl_tmp_bfgs = dK_ampl_bfgs(lambda_old(1), lambda_old(2));
        dK_sigma_tmp_bfgs = dK_sigma_bfgs(lambda_old(3));
        dNLL_l_bfgs = 0.5 * trace(K_tilde_tmp_bfgs \ dK_l_tmp_bfgs) - 0.5 * y' * ((K_tilde_tmp_bfgs \ dK_l_tmp_bfgs) / K_tilde_tmp_bfgs) * y;
        dNLL_ampl_bfgs = 0.5 * trace(K_tilde_tmp_bfgs \ dK_ampl_tmp_bfgs) - 0.5 * y' * ((K_tilde_tmp_bfgs \ dK_ampl_tmp_bfgs) / K_tilde_tmp_bfgs) * y;
        dNLL_sigma_bfgs = 0.5 * trace(K_tilde_tmp_bfgs \ dK_sigma_tmp_bfgs) - 0.5 * y' * ((K_tilde_tmp_bfgs \ dK_sigma_tmp_bfgs) / K_tilde_tmp_bfgs) * y;
%         h_bfgs = zeros(3,1);
%         h_bfgs(1) = dNLL_l_bfgs / gamma(2 - nu) * (abs(lambda_old(1) - c(1)) + epsilon)^(1 - nu);
%         h_bfgs(2) = dNLL_ampl_bfgs / gamma(2 - nu) * (abs(lambda_old(2) - c(2)) + epsilon)^(1 - nu) ;
%         h_bfgs(3) = dNLL_sigma_bfgs / gamma(2 - nu) * (abs(lambda_old(3) - c(3)) + epsilon)^(1 - nu);

        nll = NLL_bfgs( lambda_old(1), lambda_old(2), lambda_old(3));

        % Transforming hyperparams to log
%         lambda_old = log(lambda_old);
%         lambda_old = lambda_old - mu * h;
        dnll = [ dNLL_l_bfgs ; dNLL_ampl_bfgs; dNLL_sigma_bfgs ];
        
    end
    
end

iterBFGS = out_struct.iterations;
grad_errBFGS = out_struct.firstorderopt;
disp(out_struct)
lambda_old_bfgs = opt_sol;
lambda_old_nm = opt_sol_NM;

figure(grad_err_fig)
subplot(3,1,1)
hold on, grid on;
ylabel("$\| \nabla f(x) \|_\infty$");
xlabel("Iteration")
plot(grad_err_vec_BFGS(1:iterBFGS));
xlim([1 inf])

% Transforming hyperparams to exp
lambda_old_bfgs = exp(lambda_old_bfgs);
y_star = K_star(lambda_old_bfgs(1), lambda_old_bfgs(2)) / K_tilde(lambda_old_bfgs(1), lambda_old_bfgs(2), lambda_old_bfgs(3)) * y;
cov_star = K_starstar(lambda_old_bfgs(1), lambda_old_bfgs(2)) - K_star(lambda_old_bfgs(1), lambda_old_bfgs(2)) / ...
    K_tilde(lambda_old_bfgs(1), lambda_old_bfgs(2), lambda_old_bfgs(3)) * K_star(lambda_old_bfgs(1), lambda_old_bfgs(2))';
cov_star_vec = diag(cov_star);
rmse_bfgs = sqrt(sum((y_test - y_star).^2) / length(y_test));
figure(regr);
plot(x_star, y_star, 'r', 'LineWidth', 2)
patch([x_star; flipud(x_star)], [y_star + 2 * sqrt(cov_star_vec); flipud(y_star - 2*sqrt(cov_star_vec))], 'r', 'FaceAlpha', 0.25);
str_BFGS = ["RMSE BFGS = " + num2str(rmse_bfgs)];

% Transforming hyperparams to exp
lambda_old_nm = exp(lambda_old_nm);
y_star = K_star(lambda_old_nm(1), lambda_old_nm(2)) / K_tilde(lambda_old_nm(1), lambda_old_nm(2), lambda_old_nm(3)) * y;
cov_star = K_starstar(lambda_old_nm(1), lambda_old_nm(2)) - K_star(lambda_old_nm(1), lambda_old_nm(2)) / ...
    K_tilde(lambda_old_nm(1), lambda_old_nm(2), lambda_old_nm(3)) * K_star(lambda_old_nm(1), lambda_old_nm(2))';
cov_star_vec = diag(cov_star);
rmse_nm = sqrt(sum((y_test - y_star).^2) / length(y_test));
figure(regr);
plot(x_star, y_star, 'r', 'LineWidth', 2)
patch([x_star; flipud(x_star)], [y_star + 2 * sqrt(cov_star_vec); flipud(y_star - 2*sqrt(cov_star_vec))], 'r', 'FaceAlpha', 0.25);
str_NM = ["RMSE NM = " + num2str(rmse_nm)];
disp(str_NM);


%% Using Fractional GD
iter = 0;
grad_err = 1;
N = iter_max;
k_init = 1;
lambda_k_vec = zeros(k_init, 1);
sol_change = 1;

l_span_ext = linspace(0.1, 5, 200)';
ampl_span_ext = linspace(0.1, 5, 200)';
sigma0_span_ext = linspace(0.001, 0.5, 200)';
if SURFACE == true
    % Transforming hyperparams to exp
    lambda_old = exp(lambda_old);
    nll = zeros(size(l_span_ext,1),3);
    for idx = 1:size(l_span_ext,1)
        nll(idx, 1) = NLL(l_span_ext(idx), lambda_old(2), lambda_old(3));
        nll(idx, 2) = NLL(lambda_old(1), ampl_span_ext(idx), lambda_old(3));
        nll(idx, 3) = NLL(lambda_old(1), lambda_old(2), sigma0_span_ext(idx));
    end
    nll_fig = figure();
    subplot(2, 2, 1);
    plot(l_span_ext, nll(:,1), 'b');
    grid on;
    xlabel('Length scale $l$');
    ylabel('NLL');
    ylim([-inf, 100]);
    hold on;
    q = quiver(lambda_old(1), NLL(lambda_old(1), lambda_old(2), lambda_old(3)), 0, 0, 'k', 'LineWidth', 2);
    point = plot(lambda_old(1), NLL(lambda_old(1), lambda_old(2), lambda_old(3)), '.m', 'MarkerSize', 10);
    subplot(2, 2, 2);
    plot(ampl_span_ext, nll(:,2), 'b');
    grid on;
    xlabel('Amplitude $\alpha$');
    ylabel('NLL');
    ylim([-inf, 100]);
    hold on;
    subplot(2, 2, 3);
    plot(sigma0_span_ext, nll(:,3), 'b');
    grid on;
    xlabel('$\sigma_0$');
    ylabel('NLL');
    ylim([-inf, 100]);
    hold on;
    % Transforming hyperparams to log
    lambda_old = log(lambda_old);
end

disp('Fractional GD:');
grad_err_vect_FGD = zeros(N, 1);
frac_grad_err_vect_FGD = zeros(N, 1);
sigmas_hist = zeros(N, 2);
mu = 1e-3;
tic
while sol_change > 1e-60 && grad_err > grad_err_tol && iter < N

    % Transforming hyperparams to exp
    lambda_old = exp(lambda_old);
    sol_old = lambda_old;
    K_tilde_tmp = K_tilde(lambda_old(1), lambda_old(2), lambda_old(3));
    dK_l_tmp = dK_l(lambda_old(1), lambda_old(2));
    dK_ampl_tmp = dK_ampl(lambda_old(1), lambda_old(2));
    dK_sigma_tmp = dK_sigma(lambda_old(3));
    dNLL_l = 0.5 * trace(K_tilde_tmp \ dK_l_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_l_tmp) / K_tilde_tmp) * y;
    dNLL_ampl = 0.5 * trace(K_tilde_tmp \ dK_ampl_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_ampl_tmp) / K_tilde_tmp) * y;
    dNLL_sigma = 0.5 * trace(K_tilde_tmp \ dK_sigma_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_sigma_tmp) / K_tilde_tmp) * y;
    h = zeros(3,1);
    h(1) = dNLL_l / gamma(2 - nu) * (abs(lambda_old(1) - c(1)) + epsilon)^(1 - nu);
    h(2) = dNLL_ampl / gamma(2 - nu) * (abs(lambda_old(2) - c(2)) + epsilon)^(1 - nu) ;
    h(3) = dNLL_sigma / gamma(2 - nu) * (abs(lambda_old(3) - c(3)) + epsilon)^(1 - nu);
    % Transforming hyperparams to log
    lambda_old = log(lambda_old);
    lambda_old = lambda_old - mu * h;
    dNLL = h;% [ dNLL_l ; dNLL_ampl; dNLL_sigma ];
    grad_err = norm(dNLL, "inf");
    grad_err_vect_FGD(iter+1) = grad_err ;
    frac_grad_err_vect_FGD(iter+1) = norm(h, "inf");
    % Transforming hyperparams to exp
    lambda_old = exp(lambda_old);
    fun_value = NLL( lambda_old(1), lambda_old(2), lambda_old(3));
    % Updating diff order!
%     nu = 1 / cosh(0.005 * fun_value);
    sol_new = lambda_old;
%     sol_change = max(abs(sol_new - sol_old));
    sol_change = norm(sol_new - sol_old);
    
    if SURFACE == true && mod(iter, 50) == 0
        % Plotting path
        figure(nll_fig)
        title('Fractional GD');
        nll = zeros(size(l_span_ext));
        for idx = 1:size(l_span_ext,1)
            nll(idx, 1) = NLL(l_span_ext(idx), lambda_old(2), lambda_old(3));
            nll(idx, 2) = NLL(lambda_old(1), ampl_span_ext(idx), lambda_old(3));
            nll(idx, 3) = NLL(lambda_old(1), lambda_old(2), sigma0_span_ext(idx));
        end
        subplot(2, 2, 1);
        plot(l_span_ext, nll(:,1), 'b');
        plot(lambda_old(1), NLL(lambda_old(1), lambda_old(2), lambda_old(3)), '.m', 'MarkerSize', 10);
        q.XData = lambda_old(1); q.YData = NLL(lambda_old(1), lambda_old(2), lambda_old(3));
        q.UData = -dNLL(1)*cos(atan(dNLL(1))); q.VData = -dNLL(1)*sin(atan(dNLL(1)));
        subplot(2, 2, 2);
        plot(ampl_span_ext, nll(:,2), 'b');
        plot(lambda_old(2), NLL(lambda_old(1), lambda_old(2), lambda_old(3)), '.m', 'MarkerSize', 10);
        subplot(2, 2, 3);
        plot(sigma0_span_ext, nll(:,3), 'b');
        plot(lambda_old(3), NLL(lambda_old(1), lambda_old(2), lambda_old(3)), '.m', 'MarkerSize', 10);
        drawnow;
%         pause(0.1);
    end
    
    % Transforming hyperparams to log
    lambda_old = log(lambda_old);
    iter = iter + 1;
    if VERBOSE == true
        disp(["[" + num2str(iter) + "] argmin = " + num2str(lambda_old, '%.10f') + " with err: " + num2str(grad_err) + " (NLL = " + ...
            num2str(fun_value) + ")"]);
    end
end
elapsedFGD = toc;
iterFGD = iter;
grad_errFGD = grad_err;

% Transforming hyperparams to exp
lambda_old = exp(lambda_old);
y_star = K_star(lambda_old(1), lambda_old(2)) / K_tilde(lambda_old(1), lambda_old(2), lambda_old(3)) * y;
cov_star = K_starstar(lambda_old(1), lambda_old(2)) - K_star(lambda_old(1), lambda_old(2)) / ...
    K_tilde(lambda_old(1), lambda_old(2), lambda_old(3)) * K_star(lambda_old(1), lambda_old(2))';
cov_star_vec = diag(cov_star);
rmse_fgd = sqrt(sum((y_test - y_star).^2) / length(y_test));
figure(regr);
plot(x_star, y_star, 'r', 'LineWidth', 2)
patch([x_star; flipud(x_star)], [y_star + 2 * sqrt(cov_star_vec); flipud(y_star - 2*sqrt(cov_star_vec))], 'r', 'FaceAlpha', 0.25);
str_FGD = ["RMSE FGD = " + num2str(rmse_fgd)];
% ann = annotation('textbox','String', str_FGD, 'FitBoxToText','on');

% figure
% plot(grad_err_vect_FGD);
% grid on;
% title('Gradient error');

% figure(grad_err_fig)
% plot(1:iterFGD, grad_err_vect_FGD);

% figure
% plot(sigmas_hist(:,1), 'b');
% hold on, grid on;
% plot(sigmas_hist(:,2), 'r');
% legend('$\alpha$', '$\sigma_0$', 'Interpreter', 'latex');
% title('Guessed hyperparameters')

%% Using ADAM optimizer
iter = 0;
grad_err = 1;
N = iter_max;
sol_change = 1;
beta1 = 0.9;
beta2 = 0.999;
mu = 1e-2; % Forse con adam mi posso permettere un passo più grande
m_old = zeros(size(lambda_old_adam));
v_old = m_old;
disp('ADAM GD');

if SURFACE == true
    % Transforming hyperparams to exp
    lambda_old_adam = exp(lambda_old_adam);
    nll = zeros(size(X1(:)));
    for idx = 1:size(X1(:),1)
        nll(idx) = NLL(X1(idx), X2(idx), lambda_old_adam(3));
    end
    nll = reshape(nll, size(X1,1), size(X1,1));
    nll_fig_adam = figure();
%     surf_FGD = surf(X1, X2, nll, 'LineStyle', 'none', 'FaceColor', 'interp');
    [ contourM, surf_FGD ] = contourf(X1, X2, nll, 100, 'LineStyle', 'none');
    cbar = colorbar();
    xlabel('Length scale $l$');
    ylabel('Amplitude $\alpha$');
%     zlabel('NLL');
    % zlim([-inf, 5000]);
    hold on;
%     q = quiver3(lambda_old_adam(1), lambda_old_adam(2), max(max(nll)), 0, 0, 0, 'k', 'LineWidth', 2);
    q = quiver(lambda_old_adam(1), lambda_old_adam(2), 0, 0, 'k', 'LineWidth', 2);
    point = plot(lambda_old_adam(1), lambda_old_adam(2), '.m', 'MarkerSize', 10);
    view(2);
    % Transforming hyperparams to log
    lambda_old_adam = log(lambda_old_adam);
end

grad_err_vect_ADAM = zeros(N, 1);
tic
while grad_err > grad_err_tol && iter < N && sol_change > 1e-60
    % Transforming hyperparams to exp
    lambda_old_adam = exp(lambda_old_adam);
    sol_old = lambda_old_adam;
    K_tilde_tmp = K_tilde(lambda_old_adam(1), lambda_old_adam(2), lambda_old_adam(3));
    dK_l_tmp = dK_l(lambda_old_adam(1), lambda_old_adam(2));
    dK_ampl_tmp = dK_ampl(lambda_old_adam(1), lambda_old_adam(2));
    dK_sigma_tmp = dK_sigma(lambda_old_adam(3));
    dNLL_l = 0.5 * trace(K_tilde_tmp \ dK_l_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_l_tmp) / K_tilde_tmp) * y;
    dNLL_ampl = 0.5 * trace(K_tilde_tmp \ dK_ampl_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_ampl_tmp) / K_tilde_tmp) * y;
    dNLL_sigma = 0.5 * trace(K_tilde_tmp \ dK_sigma_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_sigma_tmp) / K_tilde_tmp) * y;
    dNLL = zeros(3,1);
    dNLL(1) = dNLL_l; dNLL(2) = dNLL_ampl; dNLL(3) = dNLL_sigma;
    
    h = zeros(3,1);
    h(1) = dNLL_l;% / gamma(2 - nu) * (abs(lambda_old_adam(1) - c(1)) + epsilon)^(1 - nu);
    h(2) = dNLL_ampl;% / gamma(2 - nu) * (abs(lambda_old_adam(2) - c(2)) + epsilon)^(1 - nu);
    h(3) = dNLL_sigma;% / gamma(2 - nu) * (abs(lambda_old_adam(3) - c(3)) + epsilon)^(1 - nu);
    
    % Transforming hyperparams to log
    lambda_old_adam = log(lambda_old_adam);
    m = beta1 * m_old + (1 - beta1) * h;
    v = beta2 * v_old + (1 - beta2) * h.*h;
    m_hat = m / (1 - beta1^(iter + 1));
    v_hat = v / (1 - beta2^(iter + 1));
    lambda_old_adam = lambda_old_adam - mu * m_hat ./ (sqrt(v_hat) + epsilon);
    
    m_old = m;
    v_old = v;
    
    grad_err = norm(dNLL, "inf");
    grad_err_vect_ADAM(iter+1) = grad_err ;
    % Transforming hyperparams to exp
    lambda_old_adam = exp(lambda_old_adam);
    fun_value = NLL( lambda_old_adam(1), lambda_old_adam(2), lambda_old_adam(3));
    sol_new = lambda_old_adam;
%     sol_change = max(abs(sol_new - sol_old));
    sol_change = norm(sol_new - sol_old);
    
    if SURFACE == true && mod(iter, 50) == 0
        % Plotting path
        figure(nll_fig_adam)
        nll = zeros(size(X1(:)));
        for idx = 1:size(X1(:),1)
            nll(idx) = NLL(X1(idx), X2(idx), lambda_old_adam(3));
        end
        nll = reshape(nll, size(X1, 1), size(X1, 1));
        if (max(max(nll)) - fun_value) > 100
            % Limiting the max of the LH so that contour lines are more
            % dense
            nll = min(nll, fun_value + 100);
        end
        surf_FGD.ZData = nll;
        surf_FGD.LevelListMode = 'auto';
        colorbar;
%         plot3(lambda_old_adam(1), lambda_old_adam(2), NLL( lambda_old_adam(1), lambda_old_adam(2), lambda_old_adam(3)), 'om', 'MarkerSize', 5);
%         plot(lambda_old_adam(1), lambda_old_adam(2), 'om', 'MarkerSize', 5);
        point.XData = lambda_old_adam(1); point.YData = lambda_old_adam(2);
        q.XData = lambda_old_adam(1); q.YData = lambda_old_adam(2);
        q.UData = -dNLL(1)/norm(dNLL(1:2)); q.VData = -dNLL(2)/norm(dNLL(1:2));
        drawnow;
    %     pause(0.5);
    end
    
    % Transforming hyperparams to log
    lambda_old_adam = log(lambda_old_adam);
    iter = iter + 1;
    if VERBOSE == true
        disp(["[" + num2str(iter) + "] argmin = " + num2str(lambda_old_adam, '%.10f') + " with err: " + num2str(grad_err)+ " (NLL = " + ...
            num2str(fun_value) + ")"]);
    end
end
elapsedADAM = toc;
iterADAM = iter;
grad_errADAM = grad_err;

figure(grad_err_fig)
subplot(3,1,2)
hold on, grid on;
ylabel("$\| \nabla f(x) \|_\infty$");
xlabel("Iteration")
plot(grad_err_vect_ADAM(1:iterADAM));
xlim([1 inf])

% Transforming hyperparams to exp
lambda_old_adam = exp(lambda_old_adam);
y_star_adam = K_star(lambda_old_adam(1), lambda_old_adam(2)) / ...
    K_tilde(lambda_old_adam(1), lambda_old_adam(2), lambda_old_adam(3)) * y;
cov_star_adam = K_starstar(lambda_old_adam(1), lambda_old_adam(2)) - K_star(lambda_old_adam(1), lambda_old_adam(2)) / ...
    K_tilde(lambda_old_adam(1), lambda_old_adam(2), lambda_old_adam(3)) * K_star(lambda_old_adam(1), lambda_old_adam(2))';
cov_star_vec_adam = diag(cov_star_adam);
rmse_adam = sqrt(sum((y_test - y_star_adam).^2) / length(y_test));
figure(regr);
plot(x_star, y_star_adam, 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 2)
patch([x_star; flipud(x_star)], [y_star_adam + 2*sqrt(cov_star_vec_adam);...
    flipud(y_star_adam - 2*sqrt(cov_star_vec_adam))],  [0.9290 0.6940 0.1250], 'FaceAlpha', 0.25);
str_ADAM = ["RMSE ADAM = " + num2str(rmse_adam)];

%% Quasi-Newton method
iter = 0;
func_count = 0;
grad_err = 1;
N = iter_max;
sol_change = 1;
B = eye(3) * 1e1; % Initial approx Hessian
max_radius = 100; % Max trust region radius
radius = 1; % Initial trust region radius

disp('Quasi-Newton optimizer');

% if SURFACE == true
%     % Transforming hyperparams to exp
%     lambda_old_qn = exp(lambda_old_qn);
%     nll = zeros(size(X1(:)));
%     for idx = 1:size(X1(:),1)
%         nll(idx) = NLL(X1(idx), X2(idx), lambda_old_qn(3));
%     end
%     nll = reshape(nll, size(X1,1), size(X1,1));
%     nll_fig_qn = figure();
%     [ ~, surf_FGD ] = contourf(X1, X2, nll, 100, 'LineStyle', 'none');
%     colorbar();
%     xlabel('Length scale $l$');
%     ylabel('Amplitude $\alpha$');
%     hold on;
%     q = quiver(lambda_old_qn(1), lambda_old_qn(2), 0, 0, 'k', 'LineWidth', 2);
%     point = plot(lambda_old_qn(1), lambda_old_qn(2), '.m', 'MarkerSize', 10);
%     view(2);
%     % Transforming hyperparams to log
%     lambda_old_qn = log(lambda_old_qn);
% end

if  true
    % Transforming hyperparams to exp
    lambda_old_qn = exp(lambda_old_qn);
    l_span_ext = linspace(0.1, lambda_old_qn(1)*2, 200)';
    ampl_span_ext = linspace(0.1, lambda_old_qn(2)*2, 200)';
    sigma0_span_ext = linspace(0.001, lambda_old_qn(3)*2, 200)';
    nll = zeros(size(l_span_ext,1),3);
    for idx = 1:size(l_span_ext,1)
        nll(idx, 1) = NLL(l_span_ext(idx), lambda_old_qn(2), lambda_old_qn(3));
        nll(idx, 2) = NLL(lambda_old_qn(1), ampl_span_ext(idx), lambda_old_qn(3));
        nll(idx, 3) = NLL(lambda_old_qn(1), lambda_old_qn(2), sigma0_span_ext(idx));
    end
    nll_tmp = NLL(lambda_old_qn(1), lambda_old_qn(2), lambda_old_qn(3));
    nll_fig = figure();

    subplot(1, 3, 1);
    plot(l_span_ext, nll(:,1), 'b');
    grid on;
    xlabel('$l$');
    ylabel('NLL');
    ylim([-inf, nll_tmp + 0.5*abs(nll_tmp)]);
    hold on;
    q = quiver(lambda_old_qn(1), NLL(lambda_old_qn(1), lambda_old_qn(2), lambda_old_qn(3)), 0, 0, 'k', 'LineWidth', 2);
    point = plot(lambda_old_qn(1), NLL(lambda_old_qn(1), lambda_old_qn(2), lambda_old_qn(3)), '.m', 'MarkerSize', 20);
    hold off
    axis square;

    subplot(1, 3, 2);
    plot(ampl_span_ext, nll(:,2), 'b');
    grid on;
    xlabel('$\alpha$');
    ylabel('NLL');
%     ylim([-inf, 100]);
    ylim([-inf, nll_tmp + 0.5*abs(nll_tmp)]);
    hold on;
    plot(lambda_old_qn(2), NLL(lambda_old_qn(1), lambda_old_qn(2), lambda_old_qn(3)), '.m', 'MarkerSize', 20);
    hold off
    axis square;

    subplot(1, 3, 3);
    plot(sigma0_span_ext, nll(:,3), 'b');
    grid on;
    xlabel('$\sigma_n$');
    ylabel('NLL');
%     ylim([-inf, 100]);
    ylim([-inf, nll_tmp + 0.5*abs(nll_tmp)]);
    hold on;
    plot(lambda_old_qn(3), NLL(lambda_old_qn(1), lambda_old_qn(2), lambda_old_qn(3)), '.m', 'MarkerSize', 20);
    hold off
    axis square;
    pause()
    % Transforming hyperparams to log
    lambda_old_qn = log(lambda_old_qn);
end

grad_err_vect_QN = zeros(N, 1);
tic
while grad_err > 1e-3 && iter < N && sol_change > 1e-60
    % Transforming hyperparams to exp
    lambda_old_qn = exp(lambda_old_qn);
    fun_value_old = NLL( lambda_old_qn(1), lambda_old_qn(2), lambda_old_qn(3));
    K_tilde_tmp = K_tilde(lambda_old_qn(1), lambda_old_qn(2), lambda_old_qn(3));
    dK_l_tmp = dK_l(lambda_old_qn(1), lambda_old_qn(2));
    dK_ampl_tmp = dK_ampl(lambda_old_qn(1), lambda_old_qn(2));
    dK_sigma_tmp = dK_sigma(lambda_old_qn(3));
    dNLL_l = 0.5 * trace(K_tilde_tmp \ dK_l_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_l_tmp) / K_tilde_tmp) * y;
    dNLL_ampl = 0.5 * trace(K_tilde_tmp \ dK_ampl_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_ampl_tmp) / K_tilde_tmp) * y;
    dNLL_sigma = 0.5 * trace(K_tilde_tmp \ dK_sigma_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_sigma_tmp) / K_tilde_tmp) * y;
    dNLL = zeros(3,1);
    dNLL(1) = dNLL_l; dNLL(2) = dNLL_ampl; dNLL(3) = dNLL_sigma;
    
    eps_tol = min(0.5, sqrt(norm(dNLL))) * norm(dNLL);
    z = zeros(3,1);
    r = dNLL;
    d = -r;
    subproblem_iter = 0;
    
    while subproblem_iter < 5
        min_fun = inf;
        if d'*B*d <= 0
            fun = @(tau) dNLL' * (z + tau * d) + 0.5 * (z + tau * d)' * B * (z + tau * d);
            polyn = [ d' * d, 2 * d' * z, z' * z - radius^2 ];
            tau_sol = real(roots(polyn));
            for ii = 1:length(tau_sol)
                if fun(tau_sol(ii)) < min_fun
                    tau_min = tau_sol(ii);
                    min_fun = fun(tau_sol(ii));
                end
            end
            s_opt = z + tau_min * d;
            break;
        end
        alpha = r' * r / (d' * B * d);
        z = z + alpha * d;
        if norm(z) > radius
%             fun = @(tau) dNLL' * (z + tau * d) + 0.5 * (z + tau * d)' * B * (z + tau * d);
            polyn = [ d' * d, 2 * d' * z, z' * z - radius^2 ];
            tau_sol = real(roots(polyn));
            for ii = 1:length(tau_sol)
                if 1% tau_sol(ii) >= 0
                    s_opt = z + max(tau_sol) * d;
                    break;
                end
            end
            break;
        end
        r_new = r + alpha * B * d;
        if norm(r_new) < eps_tol
            s_opt = z;
            break;
        end
        beta = r_new' * r_new / (r' * r);
        d = -r_new + beta * d;
        r = r_new;
        subproblem_iter = subproblem_iter + 1;
    end
    
    % SR1 trust region method
    % Transforming hyperparams to log
    lambda_old_qn = log(lambda_old_qn);
    lambda_tmp = lambda_old_qn + s_opt; % Candidate next solution
    % Transforming hyperparams to exp
    lambda_old_qn = exp(lambda_old_qn);
    lambda_tmp = exp(lambda_tmp);
    % Computing function and derivative value using candidate solution
    fun_value_tmp = NLL( lambda_tmp(1), lambda_tmp(2), lambda_tmp(3)); % Candidate function value
    K_tilde_tmp = K_tilde(lambda_tmp(1), lambda_tmp(2), lambda_tmp(3));
    dK_l_tmp = dK_l(lambda_tmp(1), lambda_tmp(2));
    dK_ampl_tmp = dK_ampl(lambda_tmp(1), lambda_tmp(2));
    dK_sigma_tmp = dK_sigma(lambda_tmp(3));
    dNLL_l = 0.5 * trace(K_tilde_tmp \ dK_l_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_l_tmp) / K_tilde_tmp) * y;
    dNLL_ampl = 0.5 * trace(K_tilde_tmp \ dK_ampl_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_ampl_tmp) / K_tilde_tmp) * y;
    dNLL_sigma = 0.5 * trace(K_tilde_tmp \ dK_sigma_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_sigma_tmp) / K_tilde_tmp) * y;
    dNLL_tmp = zeros(3,1);
    dNLL_tmp(1) = dNLL_l; dNLL_tmp(2) = dNLL_ampl; dNLL_tmp(3) = dNLL_sigma;
    yk = dNLL_tmp - dNLL;
    
    actual_red = fun_value_old - fun_value_tmp;
    predicted_red = -(dNLL' * s_opt + 0.5 * s_opt' * B * s_opt);
    % Transforming hyperparams to log
    lambda_old_qn = log(lambda_old_qn);
    lambda_tmp = log(lambda_tmp);
    
    rho = actual_red / predicted_red;
    update = false;
    if rho > 1e-6
        lambda_old_qn = lambda_tmp;
        update = true;
        iter = iter + 1;
%     else
%         continue; % Do not update lambda since the actual cost fun reduction is smaller than thought
    end
    
    if rho < 0.25
        radius = 0.25 * radius;
    else
        if rho > 3/4 && (norm(s_opt) - radius) <= 1e-6
            radius = min(2 * radius, max_radius);
        end
    end

    if radius < 1e-6
        fprintf("Radius is too small! (r = %f) Stopping.../n", radius);
        break;
    end
    
    % Updating the approximate Hessian
    cond_on_B = abs(s_opt' * (yk - B * s_opt)) >= 1e-8 * norm(s_opt) * norm(yk - B * s_opt);
    if cond_on_B == true
        B = B + (yk - B * s_opt) * (yk - B * s_opt)' / ( (yk - B * s_opt)' * s_opt);
%     else
%         continue; % do not update B
    end
    
    grad_err = norm(dNLL, "inf");
    grad_err_vect_QN(iter+1) = grad_err ;
    % Transforming hyperparams to exp
    lambda_old_qn = exp(lambda_old_qn);
    fun_value = NLL( lambda_old_qn(1), lambda_old_qn(2), lambda_old_qn(3));
    sol_new = lambda_old_qn;
    sol_change = norm(sol_new - sol_old);
    
    
%     if SURFACE == true && mod(iter, 50) == 0
%         % Plotting path
%         figure(nll_fig_qn)
%         nll = zeros(size(X1(:)));
%         for idx = 1:size(X1(:),1)
%             nll(idx) = NLL(X1(idx), X2(idx), lambda_old_qn(3));
%         end
%         nll = reshape(nll, size(X1, 1), size(X1, 1));
%         if (max(max(nll)) - fun_value) > 100
%             % Limiting the max of the LH so that contour lines are more
%             % dense
%             nll = min(nll, fun_value + 100);
%         end
%         surf_FGD.ZData = nll;
%         surf_FGD.LevelListMode = 'auto';
%         colorbar;
%         point.XData = lambda_old_qn(1); point.YData = lambda_old_qn(2);
%         q.XData = lambda_old_qn(1); q.YData = lambda_old_qn(2);
%         q.UData = -dNLL(1)/norm(dNLL(1:2)); q.VData = -dNLL(2)/norm(dNLL(1:2));
%         drawnow;
%     %     pause(0.5);
%     end

    if true && mod(iter, 5) == 0
        % Plotting path
        l_span_ext = linspace(0.1, lambda_old_qn(1)*2, 200)';
        ampl_span_ext = linspace(0.1, lambda_old_qn(2)*2, 200)';
        sigma0_span_ext = linspace(0.001, lambda_old_qn(3)*2, 200)';
        nll_tmp = NLL(lambda_old_qn(1), lambda_old_qn(2), lambda_old_qn(3));
        figure(nll_fig)
%         title('Fractional GD');
        nll = zeros(size(l_span_ext));
        for idx = 1:size(l_span_ext,1)
            nll(idx, 1) = NLL(l_span_ext(idx), lambda_old_qn(2), lambda_old_qn(3));
            nll(idx, 2) = NLL(lambda_old_qn(1), ampl_span_ext(idx), lambda_old_qn(3));
            nll(idx, 3) = NLL(lambda_old_qn(1), lambda_old_qn(2), sigma0_span_ext(idx));
        end
        subplot(1, 3, 1);
        cla
        hold on, grid on;
        plot(l_span_ext, nll(:,1), 'b');
        plot(lambda_old_qn(1), NLL(lambda_old_qn(1), lambda_old_qn(2), lambda_old_qn(3)), '.m', 'MarkerSize', 20);
%         q.XData = lambda_old_qn(1); q.YData = NLL(lambda_old_qn(1), lambda_old_qn(2), lambda_old_qn(3));
%         q.UData = -dNLL(1)*cos(atan(dNLL(1)))/dNLL(1); q.VData = -dNLL(1)*sin(atan(dNLL(1)))/dNLL(1);
        hold off
        ylim([-inf, nll_tmp + 0.5*abs(nll_tmp)]);

        subplot(1, 3, 2);
        cla
        hold on, grid on
        plot(ampl_span_ext, nll(:,2), 'b');
        plot(lambda_old_qn(2), NLL(lambda_old_qn(1), lambda_old_qn(2), lambda_old_qn(3)), '.m', 'MarkerSize', 20);
        hold off
        ylim([-inf, nll_tmp + 0.5*abs(nll_tmp)]);

        subplot(1, 3, 3);
        cla
        hold on, grid on
        plot(sigma0_span_ext, nll(:,3), 'b');
        plot(lambda_old_qn(3), NLL(lambda_old_qn(1), lambda_old_qn(2), lambda_old_qn(3)), '.m', 'MarkerSize', 20);
        hold off
        ylim([-inf, nll_tmp + 0.5*abs(nll_tmp)]);

        drawnow;
        pause(0.5);
    end
    
    % Transforming hyperparams to log
    lambda_old_qn = log(lambda_old_qn);
%     iter = iter + 1;
    func_count = func_count + 1;
    if VERBOSE == true
        disp(["[" + num2str(iter) + "] argmin = " + num2str(lambda_old_qn, '%.10f') + " (updated: " + update + ") with err: " + num2str(grad_err)+ " (NLL = " + ...
            num2str(fun_value) + ")"]);
    end
end
elapsedQN = toc;
iterQN = iter;
grad_errQN= grad_err;

figure(grad_err_fig)
subplot(3,1,3)
hold on, grid on;
ylabel("$\| \nabla f(x) \|_\infty$");
xlabel("Iteration")
plot(grad_err_vect_QN(1:iterQN));
xlim([1 inf])

% Transforming hyperparams to exp
lambda_old_qn = exp(lambda_old_qn);
y_star_qn = K_star(lambda_old_qn(1), lambda_old_qn(2)) / ...
    K_tilde(lambda_old_qn(1), lambda_old_qn(2), lambda_old_qn(3)) * y;
cov_star_qn = K_starstar(lambda_old_qn(1), lambda_old_qn(2)) - K_star(lambda_old_qn(1), lambda_old_qn(2)) / ...
    K_tilde(lambda_old_qn(1), lambda_old_qn(2), lambda_old_qn(3)) * K_star(lambda_old_qn(1), lambda_old_qn(2))';
cov_star_vec_qn = diag(cov_star_qn);
rmse_qn = sqrt(sum((y_test - y_star_qn).^2) / length(y_test));
figure(regr);
plot(x_star, y_star_qn, 'k', 'LineWidth', 2)
patch([x_star; flipud(x_star)], [y_star_qn + 2*sqrt(cov_star_vec_qn);...
    flipud(y_star_qn - 2*sqrt(cov_star_vec_qn))],  'k', 'FaceAlpha', 0.15);

str_QN = ["RMSE QN = " + num2str(rmse_qn)];


%% Now using vanilla GD
iter = 0;
grad_err = 1;
N = iter_max;
mu = 1e-3;
sol_change = 1;
disp('Vanilla GD');

if SURFACE == true
    % Transforming hyperparams to exp
    lambda_old_vanilla = exp(lambda_old_vanilla);
    nll = zeros(size(X1(:)));
    for idx = 1:size(X1(:),1)
        nll(idx) = NLL(X1(idx), X2(idx), lambda_old_vanilla(3));
    end
    nll = reshape(nll, size(X1,1), size(X1,1));
    nll_fig_vanilla = figure();
    title('Vanilla_GD');
%     surf_FGD = surf(X1, X2, nll, 'LineStyle', 'none', 'FaceColor', 'interp');
    [ contourM, surf_FGD ] = contourf(X1, X2, nll, 100, 'LineStyle', 'none');
    cbar = colorbar();
    xlabel('Length scale $l$');
    ylabel('Amplitude $\alpha$');
%     zlabel('NLL');
    % zlim([-inf, 5000]);
    hold on;
%     q = quiver3(lambda_old_adam(1), lambda_old_adam(2), max(max(nll)), 0, 0, 0, 'k', 'LineWidth', 2);
    q = quiver(lambda_old_vanilla(1), lambda_old_vanilla(2), 0, 0, 'k', 'LineWidth', 2);
    point = plot(lambda_old_vanilla(1), lambda_old_vanilla(2), '.m', 'MarkerSize', 10);
    view(2);
    % Transforming hyperparams to log
    lambda_old_vanilla = log(lambda_old_vanilla);
end

grad_err_vect_GD = zeros(N, 1);
tic
while grad_err > grad_err_tol && iter < N && sol_change > 1e-60
    % Transforming hyperparams to exp
    lambda_old_vanilla = exp(lambda_old_vanilla);
    sol_old = lambda_old_vanilla;
    K_tilde_tmp = K_tilde(lambda_old_vanilla(1), lambda_old_vanilla(2), lambda_old_vanilla(3));
    dK_l_tmp = dK_l(lambda_old_vanilla(1), lambda_old_vanilla(2));
    dK_ampl_tmp = dK_ampl(lambda_old_vanilla(1), lambda_old_vanilla(2));
    dK_sigma_tmp = dK_sigma(lambda_old_vanilla(3));
    dNLL_l = 0.5 * trace(K_tilde_tmp \ dK_l_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_l_tmp) / K_tilde_tmp) * y;
    dNLL_ampl = 0.5 * trace(K_tilde_tmp \ dK_ampl_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_ampl_tmp) / K_tilde_tmp) * y;
    dNLL_sigma = 0.5 * trace(K_tilde_tmp \ dK_sigma_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_sigma_tmp) / K_tilde_tmp) * y;
    dNLL = zeros(3,1);
    dNLL(1) = dNLL_l; dNLL(2) = dNLL_ampl; dNLL(3) = dNLL_sigma;
    
    % Transforming hyperparams to log
    lambda_old_vanilla = log(lambda_old_vanilla);
    lambda_old_vanilla = lambda_old_vanilla - mu * dNLL;
    grad_err = norm(dNLL, "inf");
    grad_err_vect_GD(iter+1) = grad_err ;
    % Transforming hyperparams to exp
    lambda_old_vanilla = exp(lambda_old_vanilla);
    fun_value = NLL( lambda_old_vanilla(1), lambda_old_vanilla(2), lambda_old_vanilla(3));
    sol_new = lambda_old_vanilla;
%     sol_change = max(abs(sol_new - sol_old));
    sol_change = norm(sol_new - sol_old);
    
    if SURFACE == true && mod(iter, 50) == 0
        % Plotting path
        figure(nll_fig_vanilla)
        nll = zeros(size(X1(:)));
        for idx = 1:size(X1(:),1)
            nll(idx) = NLL(X1(idx), X2(idx), lambda_old_vanilla(3));
        end
        nll = reshape(nll, size(X1, 1), size(X1, 1));
        if (max(max(nll)) - fun_value) > 100
            % Limiting the max of the LH so that contour lines are more
            % dense
            nll = min(nll, fun_value + 100);
        end
        surf_FGD.ZData = nll;
        surf_FGD.LevelListMode = 'auto';
        colorbar;
%         plot3(lambda_old_adam(1), lambda_old_adam(2), NLL( lambda_old_adam(1), lambda_old_adam(2), lambda_old_adam(3)), 'om', 'MarkerSize', 5);
%         plot(lambda_old_adam(1), lambda_old_adam(2), 'om', 'MarkerSize', 5);
        point.XData = lambda_old_vanilla(1); point.YData = lambda_old_vanilla(2);
        q.XData = lambda_old_vanilla(1); q.YData = lambda_old_vanilla(2);
        q.UData = -dNLL(1)/norm(dNLL(1:2)); q.VData = -dNLL(2)/norm(dNLL(1:2));
        drawnow;
    %     pause(0.5);
    end
    
    % Transforming hyperparams to log
    lambda_old_vanilla = log(lambda_old_vanilla);
    iter = iter + 1;
    if VERBOSE == true
        disp(["[" + num2str(iter) + "] argmin = " + num2str(lambda_old_vanilla, '%.10f') + " with err: " + num2str(grad_err)+ " (NLL = " + ...
            num2str(fun_value) + ")"]);
    end
end
elapsedGD = toc;

figure(grad_err_fig)
% plot(1:iterGD, grad_err_vect_GD);
% legend('BFGS', 'ADAM', 'QN', 'Location', 'best')

% Transforming hyperparams to exp
lambda_old_vanilla = exp(lambda_old_vanilla);
y_star_vanilla = K_star(lambda_old_vanilla(1), lambda_old_vanilla(2)) / ...
    K_tilde(lambda_old_vanilla(1), lambda_old_vanilla(2), lambda_old_vanilla(3)) * y;
cov_star_vanilla = K_starstar(lambda_old_vanilla(1), lambda_old_vanilla(2)) - K_star(lambda_old_vanilla(1), lambda_old_vanilla(2)) / ...
    K_tilde(lambda_old_vanilla(1), lambda_old_vanilla(2), lambda_old_vanilla(3)) * K_star(lambda_old_vanilla(1), lambda_old_vanilla(2))';
cov_star_vec_vanilla = diag(cov_star_vanilla);
rmse_gd = sqrt(sum((y_test - y_star_vanilla).^2) / length(y_test));
figure(regr);
plot(x_star, y_star_vanilla, 'g', 'LineWidth', 2)
patch([x_star; flipud(x_star)], [y_star_vanilla + 2*sqrt(cov_star_vec_vanilla);...
    flipud(y_star_vanilla - 2*sqrt(cov_star_vec_vanilla))], 'g', 'FaceAlpha', 0.1);
str_GD = ["RMSE GD = " + num2str(rmse_gd)];
str = [ str_FGD; str_GD; str_ADAM; str_QN; str_BFGS];
ann = annotation('textbox','String', str,'FitBoxToText','on');
leg = legend('Real', 'BFGS', 'Confidence Interval BFGS', 'FGD', 'Confidence Interval FGD', 'ADAM', 'Confidence Interval ADAM', 'QN', 'Confidence Interval QN',...
    'GD', 'Confidence Interval GD');


str_hyp = ['l    '; 'alpha'; 'sigma'];
disp("Comparison between the two methods:");
disp(["BFGS: " + str_hyp + " = " + num2str(lambda_old_bfgs, '%.10f') + " in " + num2str(iterBFGS) + " iterations (took " + num2str(elapsedFGD) + "s)."...
    + "Gradient error = " + num2str(grad_errBFGS, '%.4e')] );
disp(["Nelder-Mead: " + str_hyp + " = " + num2str(lambda_old_nm, '%.10f') + " in " + num2str(iterBFGS) + " iterations (took " + num2str(elapsedFGD) + "s)."...
    + "Gradient error = N/A"] );
disp(["FGD: " + str_hyp + " = " + num2str(lambda_old, '%.10f') + " in " + num2str(iterFGD) + " iterations (took " + num2str(elapsedFGD) + "s)."...
    + "Gradient error = " + num2str(grad_errFGD, '%.4e')] );
disp(["FADAM: " + str_hyp + " = " + num2str(lambda_old_adam, '%.10f') + " in " + num2str(iterADAM) + " iterations (took " + num2str(elapsedADAM) + "s). "...
    + "Gradient error = " + num2str(grad_errADAM, '%.4e')]);
disp(["QN: " + str_hyp + " = " + num2str(lambda_old_qn, '%.10f') + " in " + num2str(iterQN) + " iterations and " + num2str(func_count) + " fun evals (took " + num2str(elapsedQN) + "s). "...
    + "Gradient error = " + num2str(grad_errQN, '%.4e')]);
disp(["GD: " + str_hyp + " = " + num2str(lambda_old_vanilla, '%.10f') + " in " + num2str(iter) + " iterations (took " + num2str(elapsedGD) + "s). "...
    + "Gradient error = " + num2str(grad_err, '%.4e')]);
% disp(["Difference in solution (Vanilla <> Fractional): " + num2str(abs(lambda_old_vanilla - lambda_old), '%.5e')]);
% disp(["Difference in solution: (ADAM <> Fractional): " + num2str(abs(lambda_old_adam - lambda_old), '%.5e')]);

out = true;
end

function [ sigma_opt, sigma0_opt ] = guess_hyp(x, y, initial_K, lambda, fun_NLL)
% eta = sigma_noise^2 / sigma_cov^2


    % Asymptote of LogLH 1st derivative
    m = 4; % Number of basis functions
    n = size(x, 1); % Number of training points
    X = [ ones(n, 1), x, x.^2, x.^3 ];      % Design Matrix obtained from basis fun phi(x(1:end))
%     X = [ ones(n, 1) ];      % Design Matrix obtained from basis fun phi(x(1:end))
    Q = eye(n) - X * pinv(X);
%     K_tmp = K(exp(lambda_old(1)), exp(lambda_old(2)));
    N = initial_K * Q;
    y_tilde = y ./ sqrt(y' * Q * y); % Q-norm of vector y
    A0 = -Q * ( trace(N) / (n - m) * eye(n) - N);
    A1 = Q * ( trace(N*N) / (n - m) * eye(n) + trace(N) / (n - m) * N - 2*N*N);
    A2 = -A1 * N;
    A3 = (A1 + A0 * N) * N*N;
    a0 = y_tilde' * A0 * y_tilde;
    a1 = y_tilde' * A1 * y_tilde;
    a2 = y_tilde' * A2 * y_tilde;
    a3 = y_tilde' * A3 * y_tilde;
    p = [ a0, a1, a2, a3 ]; % Polynomial part of asymptote of the derivative of LogLH
    roots_loglh = roots(p);
    roots_loglh = real(roots_loglh);
    roots_loglh = max(0, roots_loglh);

    % Checking if found eta roots produce a negative second derivative to find
    % the arg max
    second_der = d2LogLH_deta2(p, roots_loglh, n, m);
    eta_opt = roots_loglh( second_der <= 0 );
%     eta_opt = eta_opt(1); % Selecting only the first optimal root

    % In this case, sigma^2 is 1 (so that it doesn't modify the kernel) and
    % thus sigma0^2 = eta o forse no?

    for ii = 1:length(eta_opt)
        inv_K_eta = inv(initial_K + eye(n) * eta_opt(ii));
        M1_eta = inv_K_eta - inv_K_eta * X * inv(X' * inv_K_eta * X) * X' * inv_K_eta;
        sigma_opt_tmp(ii) = 1 / (n - m) * y' * M1_eta * y; % sigma^2 optimal
        sigma0_opt_tmp(ii) = eta_opt(ii) * sigma_opt_tmp(ii); % sigma_noise^2 optimal
        nll_value(ii) = -fun_NLL( lambda(1), sqrt(sigma_opt_tmp(ii)), sqrt(sigma0_opt_tmp(ii)));
    end
    sigma_opt = sigma_opt_tmp(find(nll_value == max(nll_value)));
    sigma_opt = sigma_opt(1);
    sigma0_opt = sigma0_opt_tmp(find(nll_value == max(nll_value)));
    sigma0_opt = sigma0_opt(1);
end


function out = d2LogLH_deta2 (p_coeff, eta_sol, n, m)

    out = zeros(size(eta_sol));
    for idx = 1:length(eta_sol)
        out(idx) = 0.5 * (n - m) / eta_sol(idx) * ( 2 * p_coeff(1) + 3 * p_coeff(2) / eta_sol(idx) + 4 * p_coeff(3) / eta_sol(idx)^2 ...
            + 5 * p_coeff(4) / eta_sol(idx)^3);
    end
end
        

