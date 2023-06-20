%% Noise Estimation in GP
clear
close all
% clc
set(0,'defaultTextInterpreter','latex');
set(0, 'defaultAxesTickLabelInterpreter','latex');
set(0, 'defaultLegendInterpreter','latex');
warning('off', 'all');
rng(1);

% load('real_samples.mat');
% x = optsamples1(:, 1:2);
% y = optsamples1(:, 3);
% y_noisefree = y;

% load("merged_cloud.mat");
load("artificial_cloud_1.mat");
circleRadius = [ 0.6; 0.85; 0.5 ] * 4 / 5; 
circleCenter = [    1, 4, 2.25;
                    2.5, 3.5, 4.5 ] * 4 / 5;

% load('x_GPIS2.mat');
% load('y_GPIS2.mat');
% load('y_noisefree_GPIS2.mat');

% load('x_GPIS3.mat');
% load('y_GPIS3.mat');
% load('y_noisefree_GPIS3.mat');

% [ X1_train, X2_train ] = meshgrid(linspace(0, 1, 10), linspace(0, 1, 10));
% x = [ X1_train(:), X2_train(:) ];
% y_noisefree = sin(pi * x(:,1)) + sin(pi * x(:,2));
% y = y_noisefree + randn(size(y_noisefree)) * 1e-2;

PLOT_FREQ = 1;
VERBOSE = false;
% 
% f_user = figure;
% 
% figure(f_user)
% [x1, x2, button] = ginput(300);
% TRAIN_POINTS = size(x1, 1);
% % mouse left: space; 
% %       middle: countour; 
% %       right: obstacle.
% x = 4*[x1'; x2']';
% y_noisefree = button - 2;
% y = button - 2 + randn(TRAIN_POINTS,1) * 1e-2;
% % 
% save('x_GPIS3.mat', 'x');
% save('y_GPIS3.mat', 'y');
% save('y_noisefree_GPIS3.mat', 'y_noisefree');

%%
% % Adding some more free points to training dataset...
% add_pts = 6;
% add_x = [ linspace(0, 4, add_pts)', zeros(add_pts,1); linspace(0, 4, add_pts)', 4*ones(add_pts,1);...
%         zeros(add_pts,1), linspace(0, 4, add_pts)'; 4*ones(add_pts,1), linspace(0, 4, add_pts)' ];
% % add_x = [ linspace(0, 1, add_pts)', -0.6*ones(add_pts,1); linspace(0, 1, add_pts)', 0.6*ones(add_pts,1);...
% %         -0.1*ones(add_pts,1), linspace(-0.6, 0.6, add_pts)'; 1*ones(add_pts,1), linspace(-0.6, 0.6, add_pts)' ];
% x = [ x; add_x ]; 
% add_y = -ones(size(add_x, 1), 1);
% y = [ y; add_y + randn(size(add_y)) * 1e-1 ];
% y_noisefree = [ y_noisefree; add_y ];

TRAIN_POINTS = size(x, 1);
% l0 = [0.5;0.5];
l0 = [ 2;2 ];
% l0 = [1; 1];
[ X1_test, X2_test ] = meshgrid(linspace(0, 4, 50), linspace(0, 4, 50));
% [ X1_test, X2_test ] = meshgrid(linspace(0, 1, 50), linspace(0, 1, 50));
% [ X1_test, X2_test ] = meshgrid(linspace(0, 1, 50), linspace(-0.5, 0.5, 50));
x_star = [ X1_test(:), X2_test(:) ];

% % Trying to add some geometrical (spherical) priors for GPIS2 dataset
% A_1 = diag([0.5^-2; 0.5^-2]);
% A_2 = diag([0.75^-2; 0.75^-2]);
% center1 = [ 1, 2.5 ];
% center2 = [ 3.5, 2.25 ];
% % w1 = 0.5 - tanh(
% mf = zeros(size(x, 1),1);
% for ii = 1:size(x, 1)
%     mf(ii) = 0.5 / 2 * ( (x(ii, :) - center1) * A_1 * (x(ii, :) - center1)' - 1 ) + ...
%         0.75 / 2 * ( (x(ii, :) - center2) * A_2 * (x(ii, :) - center2)' - 1 );
% end
% y = y - mf;
% 
% for ii = 1:size(x_star, 1)
%     mf_test(ii) = max(0, -0.5 / 2 * ( (x_star(ii, :) - center1) * A_1 * (x_star(ii, :) - center1)' - 1 )) + ...
%         max( 0, -0.75 / 2 * ( (x_star(ii, :) - center2) * A_2 * (x_star(ii, :) - center2)' - 1 ));
% end
% 
% figure
% surf(X1_test, X2_test, reshape(mf_test, size(X1_test)));

% Trying to add some geometrical (spherical) priors for a set with ONLY
% ONE spherical obstacle
% A_1 = diag([0.5^-2; 0.5^-2]);
% center1 = [ 1, 2.5 ];
% 
% mf = zeros(size(x, 1),1);
% for ii = 1:size(x, 1)
%     mf(ii) = 0.5 / 2 * ( (x(ii, :) - center1) * A_1 * (x(ii, :) - center1)' - 1 );
% end
% y = y - mf;
% y = y + mf;
% 
% for ii = 1:size(x_star, 1)
%     mf_test(ii) = max(0, -0.5 / 2 * ( (x_star(ii, :) - center1) * A_1 * (x_star(ii, :) - center1)' - 1 ));
% end
% 
% figure
% surf(X1_test, X2_test, reshape(mf_test, size(X1_test)));

% x = linspace(-5, 5, TRAIN_POINTS)'; %100
% % y_real = @(x) x .* sin(2 * x);
% % y_real = @(x) 0.8 + ( x + 0.2 ) .* ( 1 - 5 ./ (1 + exp(-2 * x)));
% y_real = @(x) sin(x.^2 / 2); % Now it works! There was an error in building K_tmp - was using logged params...
% % y_real = @(x) sin(x .* sqrt(abs(x)));
% y = y_real(x) + randn(TRAIN_POINTS,1) * 1e-1;
% l0 = 2; % initial length scale
% x_star = linspace(-5, 5, 200)';

% Adam sembra più robusto degli altri due, ma per qualche motivo, quando
% funziona, vanilla GD ci mette meno iterazioni! In realtà FGD ci mette
% meno iterazioni, ma adam rimane il più robusto quando cambiano le
% condizioni iniziali (che è solo l in questo caso)

% regr = figure();
% plot(x, y, 'o-', 'LineWidth', 1);
% grid on, hold on;

% lambda_old = log([ l0; 3; 1e-1 ]); % [ l, alpha, sigma ]
lambda_old = log([ l0; 1; 1e-1 ]); % [ l, alpha, sigma ]

lambda_old_qn = lambda_old;
% sigma = sqrt(1e-3);%0.0618;
iter_max = 200;
grad_err_tol = 1e-2; % Leggere il paper dei cinesi per capire come far spiccare il fractional in velocità... adesso va leggermente più veloce :(
% c = lambda_old;

radius0 = 0.1;

l_span = linspace(0.1, 5, 20);
ampl_span = linspace(0.1, 5, 20);
[X1, X2] = ndgrid(l_span, l_span);

K = @(l, ampl) ampl^2 * exp(-0.5 * pdist2(x, x, 'mahalanobis', diag(l.^2)).^2) + 1e-6 * eye(size(x,1), size(x,1)); % added some jitter...
% K_tilde = @(l, ampl, sigma) ampl^2 * exp(-0.5 * pdist2(x, x).^2 / l^2) + sigma^2 * eye(size(x,1), size(x,1)); % Kernel with noise
K_tilde = @(l, ampl, sigma) K(l, ampl) + sigma^2 * eye(size(x,1), size(x,1)); % Kernel with noise
K_star = @(l,ampl) ampl^2 * exp(-0.5 * pdist2(x_star, x, 'mahalanobis', diag(l.^2)).^2);
K_starstar = @(l,ampl) ampl^2 * exp(-0.5 * pdist2(x_star, x_star,'mahalanobis', diag(l.^2)).^2);
dK_l1 = @(l,ampl) pdist2(x(:,1), x(:,1)).^2 / l(1)^3 .* K(l, ampl);
dK_l2 = @(l,ampl) pdist2(x(:,2), x(:,2)).^2 / l(2)^3 .* K(l, ampl);
dK_ampl = @(l, ampl) 2 / ampl * K(l, ampl);
dK_sigma = @(sigma) 2 * sigma * eye(size(x,1), size(x,1));
% ddK_l = @(l) -3 * pdist2(x, x) / l^4 .* K(l) + pdist2(x, x) / l^3 .* dK_l(l);
% var_K = kernel(x, x, lambda_old(2), lambda_old(1)) + sigma^2 * eye(size(x,1), size(x,1));
% det(var_K)

% NLL = @(l, ampl, sigma) 0.5 * (log(det(K_tilde(l, ampl, sigma))) +...
%     y' * inv(K_tilde(l, ampl, sigma)) * y + TRAIN_POINTS*log(2 * pi));
NLL = @(l, ampl, sigma) sum(log(diag(chol(K_tilde(l, ampl, sigma))))) +...
            0.5 * y' / K_tilde(l, ampl, sigma) * y + TRAIN_POINTS/2*log(2 * pi);

%% Noise estimation

% eta = sigma_noise^2 / sigma_cov^2


% Asymptote of LogLH 1st derivative
m = 6; % Number of basis functions
X = [ ones(TRAIN_POINTS, 1), x(:,1), x(:,2), x(:,1).^2, x(:,1) .* x(:,2), x(:,2).^2];%, ...
%             x(:,1).^3, x(:,1).^2 .* x(:,2), x(:,1) .* x(:,2).^2, x(:,2).^3 ];      % Design Matrix obtained from basis fun phi(x(1:end))
Q = eye(TRAIN_POINTS) - X * pinv(X);
K_tmp = K(exp(lambda_old(1:2)), exp(lambda_old(3)));
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
M1_eta = inv_K_eta - inv_K_eta * X * inv(X' * inv_K_eta * X) * X' * inv_K_eta;
sigma_opt = 1 / (TRAIN_POINTS - m) * y' * M1_eta * y; % sigma^2 optimal
sigma0_opt = eta_opt * sigma_opt; % sigma_noise^2 optimal

% Update the initial parameters
lambda_old = log([ l0; sqrt(sigma_opt); sqrt(sigma0_opt) ]);
lambda_old_qn = lambda_old;
lambda_old_fqn = lambda_old;
lambda_old_bfgs = lambda_old;
lambda_old_nm = lambda_old;
lambda_old_adam = lambda_old;

%% Using ADAM optimizer
iter = 0;
grad_err = 1;
N = 2000; %iter_max;
sol_change = 1;
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-6;
mu = 1e-2; % Forse con adam mi posso permettere un passo più grande
m_old = zeros(size(lambda_old_adam));
v_old = m_old;
disp('ADAM GD');

grad_err_vect_ADAM = zeros(N, 1);
tic
while grad_err > grad_err_tol && iter < N && sol_change > 1e-60
    % Transforming hyperparams to exp
    lambda_old_adam = exp(lambda_old_adam);
    fun_value_old = NLL( lambda_old_adam(1:2), lambda_old_adam(3), lambda_old_adam(4));
    K_tilde_tmp = K_tilde(lambda_old_adam(1:2), lambda_old_adam(3), lambda_old_adam(4));
    dK_l1_tmp = dK_l1(lambda_old_adam(1:2), lambda_old_adam(3));
    dK_l2_tmp = dK_l2(lambda_old_adam(1:2), lambda_old_adam(3));
    dK_ampl_tmp = dK_ampl(lambda_old_adam(1:2), lambda_old_adam(3));
    dK_sigma_tmp = dK_sigma(lambda_old_adam(4));
    dNLL_l1 = 0.5 * trace(K_tilde_tmp \ dK_l1_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_l1_tmp) / K_tilde_tmp) * y;
    dNLL_l2 = 0.5 * trace(K_tilde_tmp \ dK_l2_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_l2_tmp) / K_tilde_tmp) * y;
    dNLL_ampl = 0.5 * trace(K_tilde_tmp \ dK_ampl_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_ampl_tmp) / K_tilde_tmp) * y;
    dNLL_sigma = 0.5 * trace(K_tilde_tmp \ dK_sigma_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_sigma_tmp) / K_tilde_tmp) * y;
    dNLL = [ dNLL_l1; dNLL_l2; dNLL_ampl; dNLL_sigma ];
    
    % Transforming hyperparams to log
    lambda_old_adam = log(lambda_old_adam);
    m = beta1 * m_old + (1 - beta1) * dNLL;
    v = beta2 * v_old + (1 - beta2) * dNLL.*dNLL;
    m_hat = m / (1 - beta1^(iter + 1));
    v_hat = v / (1 - beta2^(iter + 1));
    lambda_old_adam = lambda_old_adam - mu * m_hat ./ (sqrt(v_hat) + epsilon);
    
    m_old = m;
    v_old = v;
    
    grad_err = norm(dNLL, "inf");
    grad_err_vect_ADAM(iter+1) = grad_err ;
    % Transforming hyperparams to exp
    lambda_old_adam = exp(lambda_old_adam);
    fun_value = NLL( lambda_old_adam(1:2), lambda_old_adam(3), lambda_old_adam(4));
    sol_new = lambda_old_adam;
%     sol_change = max(abs(sol_new - sol_old));
%     sol_change = norm(sol_new - sol_old);
    
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
fun_valueADAM = fun_value;

% Transforming hyperparams to exp
lambda_old_adam = exp(lambda_old_adam);
y_star_adam = K_star(lambda_old_adam(1:2), lambda_old_adam(3)) / K_tilde(lambda_old_adam(1:2), lambda_old_adam(3), lambda_old_adam(4)) * y;
cov_star_adam = K_starstar(lambda_old_adam(1:2), lambda_old_adam(3)) - K_star(lambda_old_adam(1:2), lambda_old_adam(3)) / ...
    K_tilde(lambda_old_adam(1:2), lambda_old_adam(3), lambda_old_adam(4)) * K_star(lambda_old_adam(1:2), lambda_old_adam(3))';
cov_star_vec_adam = diag(cov_star_adam);
% rmse_qn = sqrt(immse(y_test, y_star_adam));

f1 = figure;
figure(f1)
% subplot(1, 2, 1)
cla
axis('square')
surface(X1_test, X2_test, reshape(y_star_adam, 50, 50) - max(y_star_adam), 'FaceColor','interp','EdgeColor','interp');
hold on
plot(x(y_noisefree==1,1), x(y_noisefree==1,2), '.','markersize',28,'color',[.8 0 0]); %Interior points
plot(x(y_noisefree==0,1), x(y_noisefree==0,2), '.','markersize',28,'color',[.8 .4 0]); %Border points
plot(x(y_noisefree==-1,1), x(y_noisefree==-1,2), '.','markersize',28,'color',[0 .6 0]); %Exterior points
M_ADAM = contour(X1_test, X2_test, reshape(y_star_adam, 50, 50), [0,0]);
colormap('gray');
% title('Environment modeling QN')
xlabel('$x_1$ [m]', 'Interpreter', 'latex')
ylabel('$x_2$ [m]', 'Interpreter', 'latex')
colorbar('Ticks', [min(y_star_adam)-max(y_star_adam), 0], 'TickLabels', {'Free', 'Obstacle'});
hold off

% subplot(1, 2, 2);
figure
cla
% contourf(X1_test, X2_test, reshape(cov_star_vec_adam, 50, 50), 'LineStyle', 'None');
surface(X1_test, X2_test, reshape(cov_star_vec_adam, 50, 50) - max(cov_star_vec_adam), 'FaceColor','interp','EdgeColor','interp');
hold on
plot(x(y_noisefree==1,1), x(y_noisefree==1,2), '.','markersize',28,'color',[.8 0 0]); %Interior points
plot(x(y_noisefree==0,1), x(y_noisefree==0,2), '.','markersize',28,'color',[.8 .4 0]); %Border points
plot(x(y_noisefree==-1,1), x(y_noisefree==-1,2), '.','markersize',28,'color',[0 .6 0]); %Exterior points
contour(X1_test, X2_test, reshape(y_star_adam, 50, 50), [0,0], 'w');
colormap('gray');
% title('Environment modeling QN (Variance)')
xlabel('$x_1$ [m]', 'Interpreter', 'latex')
ylabel('$x_2$ [m]', 'Interpreter', 'latex')
colorbar('Ticks', [min(cov_star_vec_adam) - max(cov_star_vec_adam), 0], 'TickLabels', {'Low', 'High'})
axis('square')
hold off

param_ADAM = max(rmse_contour(M_ADAM, circleCenter, circleRadius));% / iterADAM;

%% SR1 solver
tic
[opt_sol, fun_valueSR1, out_struct_SR1] = trainGP_SR1(lambda_old_qn, x, y);
% [opt_sol, fun_valueSR1, out_struct_SR1] = trainGP_SR1(opt_sol + randn(4,1)*0.1, x, y);

elapsedQN = toc;
iterQN = out_struct_SR1.iterations;
grad_errQN= out_struct_SR1.gradients(iterQN);

% Transforming hyperparams to exp
lambda_old_qn = exp(opt_sol);
y_star_qn = K_star(lambda_old_qn(1:2), lambda_old_qn(3)) / K_tilde(lambda_old_qn(1:2), lambda_old_qn(3), lambda_old_qn(4)) * y;
cov_star_qn = K_starstar(lambda_old_qn(1:2), lambda_old_qn(3)) - K_star(lambda_old_qn(1:2), lambda_old_qn(3)) / ...
    K_tilde(lambda_old_qn(1:2), lambda_old_qn(3), lambda_old_qn(4)) * K_star(lambda_old_qn(1:2), lambda_old_qn(3))';
cov_star_vec_qn = diag(cov_star_qn);
% rmse_qn = sqrt(immse(y_test, y_star_qn));

f2 = figure;
figure(f2)
% subplot(1, 2, 1)
cla
surface(X1_test, X2_test, reshape(y_star_qn, 50, 50) - max(y_star_qn), 'FaceColor','interp','EdgeColor','interp');
hold on
plot(x(y_noisefree==1,1), x(y_noisefree==1,2), '.','markersize',28,'color',[.8 0 0]); %Interior points
plot(x(y_noisefree==0,1), x(y_noisefree==0,2), '.','markersize',28,'color',[.8 .4 0]); %Border points
plot(x(y_noisefree==-1,1), x(y_noisefree==-1,2), '.','markersize',28,'color',[0 .6 0]); %Exterior points
M_SR1 = contour(X1_test, X2_test, reshape(y_star_qn, 50, 50), [0,0]);
colormap('gray');
% title('Environment modeling QN')
xlabel('$x_1$ [m]', 'Interpreter', 'latex')
ylabel('$x_2$ [m]', 'Interpreter', 'latex')
colorbar('Ticks', [min(y_star_qn)-max(y_star_qn), 0], 'TickLabels', {'Free', 'Obstacle'})
axis('square')
hold off

% subplot(1, 2, 2);
figure
cla
% contourf(X1_test, X2_test, reshape(cov_star_vec_qn, 50, 50), 'LineStyle', 'None');
surface(X1_test, X2_test, reshape(cov_star_vec_qn, 50, 50) - max(cov_star_vec_qn), 'FaceColor','interp','EdgeColor','interp');
hold on
plot(x(y_noisefree==1,1), x(y_noisefree==1,2), '.','markersize',28,'color',[.8 0 0]); %Interior points
plot(x(y_noisefree==0,1), x(y_noisefree==0,2), '.','markersize',28,'color',[.8 .4 0]); %Border points
plot(x(y_noisefree==-1,1), x(y_noisefree==-1,2), '.','markersize',28,'color',[0 .6 0]); %Exterior points
contour(X1_test, X2_test, reshape(y_star_qn, 50, 50), [0,0], 'w');
colormap('gray');
% title('Environment modeling QN (Variance)')
xlabel('$x_1$ [m]', 'Interpreter', 'latex')
ylabel('$x_2$ [m]', 'Interpreter', 'latex')
colorbar('Ticks', [min(cov_star_vec_qn) - max(cov_star_vec_qn), 0], 'TickLabels', {'Low', 'High'})
axis('square')
hold off

param_QN = max(rmse_contour(M_SR1, circleCenter, circleRadius));% / iterQN;
% str_QN = ["RMSE QN = " + num2str(rmse_qn)];

%% Using Matlab BFGS

[opt_sol, fun_valueBFGS, out_struct, grad_err_vec_BFGS] = trainGP_BFGS(lambda_old_bfgs, x, y);

iterBFGS = out_struct.iterations;
grad_errBFGS = out_struct.firstorderopt;
disp(out_struct)
lambda_old_bfgs = opt_sol;

% Transforming hyperparams to exp
lambda_old_bfgs = exp(lambda_old_bfgs);
y_star_bfgs = K_star(lambda_old_bfgs(1:2), lambda_old_bfgs(3)) / K_tilde(lambda_old_bfgs(1:2), lambda_old_bfgs(3), lambda_old_bfgs(4)) * y;
cov_star_bfgs = K_starstar(lambda_old_bfgs(1:2), lambda_old_bfgs(3)) - K_star(lambda_old_bfgs(1:2), lambda_old_bfgs(3)) / ...
    K_tilde(lambda_old_bfgs(1:2), lambda_old_bfgs(3), lambda_old_bfgs(4)) * K_star(lambda_old_bfgs(1:2), lambda_old_bfgs(3))';
cov_star_vec_bfgs = diag(cov_star_bfgs);
% rmse_qn = sqrt(immse(y_test, y_star_bfgs));

f3 = figure;
figure(f3)
% subplot(1, 2, 1)
cla
surface(X1_test, X2_test, reshape(y_star_bfgs, 50, 50) - max(y_star_bfgs), 'FaceColor','interp','EdgeColor','interp');
hold on
plot(x(y_noisefree==1,1), x(y_noisefree==1,2), '.','markersize',28,'color',[.8 0 0]); %Interior points
plot(x(y_noisefree==0,1), x(y_noisefree==0,2), '.','markersize',28,'color',[.8 .4 0]); %Border points
plot(x(y_noisefree==-1,1), x(y_noisefree==-1,2), '.','markersize',28,'color',[0 .6 0]); %Exterior points
M_BFGS = contour(X1_test, X2_test, reshape(y_star_bfgs, 50, 50), [0,0]);
colormap('gray');
% title('Environment modeling QN')
xlabel('$x_1$ [m]', 'Interpreter', 'latex')
ylabel('$x_2$ [m]', 'Interpreter', 'latex')
colorbar('Ticks', [min(y_star_bfgs)-max(y_star_bfgs), 0], 'TickLabels', {'Free', 'Obstacle'})
axis('square')
hold off

% subplot(1, 2, 2);
figure
cla
% contourf(X1_test, X2_test, reshape(cov_star_vec_bfgs, 50, 50), 'LineStyle', 'None');
surface(X1_test, X2_test, reshape(cov_star_vec_bfgs, 50, 50) - max(cov_star_vec_bfgs), 'FaceColor','interp','EdgeColor','interp');
hold on
plot(x(y_noisefree==1,1), x(y_noisefree==1,2), '.','markersize',28,'color',[.8 0 0]); %Interior points
plot(x(y_noisefree==0,1), x(y_noisefree==0,2), '.','markersize',28,'color',[.8 .4 0]); %Border points
plot(x(y_noisefree==-1,1), x(y_noisefree==-1,2), '.','markersize',28,'color',[0 .6 0]); %Exterior points
contour(X1_test, X2_test, reshape(y_star_bfgs, 50, 50), [0,0], 'w');
colormap('gray');
% title('Environment modeling QN (Variance)')
xlabel('$x_1$ [m]', 'Interpreter', 'latex')
ylabel('$x_2$ [m]', 'Interpreter', 'latex')
colorbar('Ticks', [min(cov_star_vec_bfgs) - max(cov_star_vec_bfgs), 0], 'TickLabels', {'Low', 'High'})
axis('square')
hold off

param_BFGS = max(rmse_contour(M_BFGS, circleCenter, circleRadius));% / iterBFGS;

%% Using Matlab Nelder-Mead

[opt_sol, fun_valueNM, out_struct] = trainGP_NM(lambda_old_nm, x, y);
lambda_old_nm = opt_sol;
iterNM = out_struct.iterations;
disp(out_struct)

% Transforming hyperparams to exp
lambda_old_nm = exp(lambda_old_nm);
y_star_nm = K_star(lambda_old_nm(1:2), lambda_old_nm(3)) / K_tilde(lambda_old_nm(1:2), lambda_old_nm(3), lambda_old_nm(4)) * y;
cov_star_nm = K_starstar(lambda_old_nm(1:2), lambda_old_nm(3)) - K_star(lambda_old_nm(1:2), lambda_old_nm(3)) / ...
    K_tilde(lambda_old_nm(1:2), lambda_old_nm(3), lambda_old_nm(4)) * K_star(lambda_old_nm(1:2), lambda_old_nm(3))';
cov_star_vec_nm = diag(cov_star_nm);
% rmse_qn = sqrt(immse(y_test, y_star_nm));

f4 = figure;
figure(f4)
% subplot(1, 2, 1)
cla
surface(X1_test, X2_test, reshape(y_star_nm, 50, 50) - max(y_star_nm), 'FaceColor','interp','EdgeColor','interp');
hold on
plot(x(y_noisefree==1,1), x(y_noisefree==1,2), '.','markersize',28,'color',[.8 0 0]); %Interior points
plot(x(y_noisefree==0,1), x(y_noisefree==0,2), '.','markersize',28,'color',[.8 .4 0]); %Border points
plot(x(y_noisefree==-1,1), x(y_noisefree==-1,2), '.','markersize',28,'color',[0 .6 0]); %Exterior points
M_NM = contour(X1_test, X2_test, reshape(y_star_nm, 50, 50), [0,0]);
colormap('gray');
% title('Environment modeling QN')
xlabel('$x_1$ [m]', 'Interpreter', 'latex')
ylabel('$x_2$ [m]', 'Interpreter', 'latex')
colorbar('Ticks', [min(y_star_nm)-max(y_star_nm), 0], 'TickLabels', {'Free', 'Obstacle'})
axis('square')
hold off

% subplot(1, 2, 2);
figure
cla
% contourf(X1_test, X2_test, reshape(cov_star_vec_nm, 50, 50), 'LineStyle', 'None');
surface(X1_test, X2_test, reshape(cov_star_vec_nm, 50, 50) - max(cov_star_vec_nm), 'FaceColor','interp','EdgeColor','interp');
hold on
plot(x(y_noisefree==1,1), x(y_noisefree==1,2), '.','markersize',28,'color',[.8 0 0]); %Interior points
plot(x(y_noisefree==0,1), x(y_noisefree==0,2), '.','markersize',28,'color',[.8 .4 0]); %Border points
plot(x(y_noisefree==-1,1), x(y_noisefree==-1,2), '.','markersize',28,'color',[0 .6 0]); %Exterior points
contour(X1_test, X2_test, reshape(y_star_nm, 50, 50), [0,0], 'w');
colormap('gray');
% title('Environment modeling QN (Variance)')
xlabel('$x_1$ [m]', 'Interpreter', 'latex')
ylabel('$x_2$ [m]', 'Interpreter', 'latex')
colorbar('Ticks', [min(cov_star_vec_nm) - max(cov_star_vec_nm), 0], 'TickLabels', {'Low', 'High'})
axis('square')
hold off

param_NM = max(rmse_contour(M_NM, circleCenter, circleRadius));% / iterNM;

%%

% str_FQN = ["RMSE FQN = " + num2str(rmse_qn)];

% str = [ str_QN; str_FQN ];
% ann = annotation('textbox','String', str,'FitBoxToText','on');
% leg = legend('Real', 'QN', 'Confidence Interval QN', 'FQN', 'Confidence Interval FQN');


str_hyp = ['l1   '; 'l2   '; 'alpha'; 'sigma'];
disp("Results:");
disp("ADAM: " + str_hyp + " = " + num2str(lambda_old_adam, '%.10f') + " in " + num2str(iterADAM) + " iterations (took " + num2str(elapsedADAM) + "s). "...
    + "Gradient error = " + num2str(grad_errADAM) + " fval = " + num2str(fun_valueADAM) + " Ratio = " + num2str(param_ADAM));
disp("QN: " + str_hyp + " = " + num2str(lambda_old_qn, '%.10f') + " in " + num2str(iterQN) + " iterations (took " + num2str(elapsedQN) + "s). "...
    + "Gradient error = " + num2str(grad_errQN) + " fval = " + num2str(fun_valueSR1) + " Ratio = " + num2str(param_QN));
disp("BFGS: " + str_hyp + " = " + num2str(lambda_old_bfgs, '%.10f') + " in " + num2str(iterBFGS) + " iterations (took " + num2str(0) + "s). "...
    + "Gradient error = " + num2str(grad_errBFGS) + " fval = " + num2str(fun_valueBFGS) + " Ratio = " + num2str(param_BFGS));
disp("Nelder-Mead: " + str_hyp + " = " + num2str(lambda_old_nm, '%.10f') + " in " + num2str(iterNM) + " iterations (took " + num2str(0) + "s). "...
    + "Gradient error = N/A" + " fval = " + num2str(fun_valueNM) + " Ratio = " + num2str(param_NM));
% disp(["FQN:" + str_hyp + " = " + num2str(lambda_old_fqn, '%.10f') + " in " + num2str(iterFQN) + " iterations (took " + num2str(elapsedFQN) + "s). "...
%     + "Gradient error = " + num2str(grad_errFQN)]);

% % Additional plots for fractional
% figure
% title('Fractional');
% subplot(2, 1, 1)
% plot(grad_err_vect_FQN(1:iter-1), 'b')
% grid on;
% hold on;
% legend('Gradient');
% subplot(2, 1, 2)
% plot(radius_hist(1:iter-1), 'r')
% hold on;
% plot(s_opt_hist(1:iter-1), 'g')
% grid on;
% legend('Trust-region radius', '$s_{opt}$');

%% Functions

function [opt_sol, fval, out_struct, grad_err_vec_BFGS] = trainGP_BFGS(lambda_old_bfgs, x, y)

    options = optimoptions('fminunc','Algorithm','quasi-newton','SpecifyObjectiveGradient',true, ...
            'OptimalityTolerance', 0.5e-3, 'OutputFcn', @outFnc);

    grad_err_vec_BFGS = zeros(options.MaxIterations, 1);
    function stop = outFnc(x, optimValues,state)
        stop = false;
        grad_err_vec_BFGS(optimValues.iteration+1) = norm(optimValues.gradient, 'inf');
    end
    
    TRAIN_POINTS = size(x,1);
    [opt_sol, fval, ~, out_struct] = fminunc(@cost_fun, lambda_old_bfgs, options);
    
    function [nll, dnll] = cost_fun(lambda_old)

        K_bfgs = @(l, ampl) ampl^2 * exp(-0.5 * pdist2(x, x, 'mahalanobis', diag(l.^2)).^2) + 1e-6 * eye(size(x,1), size(x,1)); % added some jitter...
        K_tilde_bfgs= @(l, ampl, sigma) K_bfgs(l, ampl) + sigma^2 * eye(size(x,1), size(x,1)); % Kernel with noise
        dK_l1_bfgs = @(l,ampl) pdist2(x(:,1), x(:,1)).^2 / l(1)^3 .* K_bfgs(l, ampl);
        dK_l2_bfgs = @(l,ampl) pdist2(x(:,2), x(:,2)).^2 / l(2)^3 .* K_bfgs(l, ampl);
        dK_ampl_bfgs = @(l, ampl) 2 / ampl * K_bfgs(l, ampl);
        dK_sigma_bfgs = @(sigma) 2 * sigma * eye(size(x,1), size(x,1));
        NLL_bfgs = @(l, ampl, sigma) sum(log(diag(chol(K_tilde_bfgs(l, ampl, sigma))))) +...
            0.5 * y' / K_tilde_bfgs(l, ampl, sigma) * y + TRAIN_POINTS/2*log(2 * pi);


        % Transforming hyperparams to exp
        lambda_old = exp(lambda_old);
        K_tilde_tmp = K_tilde_bfgs(lambda_old(1:2), lambda_old(3), lambda_old(4));
        dK_l1_tmp = dK_l1_bfgs(lambda_old(1:2), lambda_old(3));
        dK_l2_tmp = dK_l2_bfgs(lambda_old(1:2), lambda_old(3));
        dK_ampl_tmp = dK_ampl_bfgs(lambda_old(1:2), lambda_old(3));
        dK_sigma_tmp = dK_sigma_bfgs(lambda_old(4));
        dNLL_l1 = 0.5 * trace(K_tilde_tmp \ dK_l1_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_l1_tmp) / K_tilde_tmp) * y;
        dNLL_l2 = 0.5 * trace(K_tilde_tmp \ dK_l2_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_l2_tmp) / K_tilde_tmp) * y;
        dNLL_ampl = 0.5 * trace(K_tilde_tmp \ dK_ampl_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_ampl_tmp) / K_tilde_tmp) * y;
        dNLL_sigma = 0.5 * trace(K_tilde_tmp \ dK_sigma_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_sigma_tmp) / K_tilde_tmp) * y;
        dnll = [ dNLL_l1; dNLL_l2; dNLL_ampl; dNLL_sigma ];

        nll = NLL_bfgs( lambda_old(1:2), lambda_old(3), lambda_old(4));
        
    end
    
end

function [opt_sol, fval, out_struct] = trainGP_NM(lambda_old_nm, x, y)

    fprintf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Nelder-Mead  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n");
    TRAIN_POINTS = size(x,1);
    options_NM = optimset('Display','final','MaxIter', 10000);
    [opt_sol, fval, ~, out_struct] = fminsearch(@cost_fun, lambda_old_nm, options_NM);
    disp(out_struct)
    
    function [nll, dnll] = cost_fun(lambda_old)

        K_nm = @(l, ampl) ampl^2 * exp(-0.5 * pdist2(x, x, 'mahalanobis', diag(l.^2)).^2) + 1e-6 * eye(size(x,1), size(x,1)); % added some jitter...
        K_tilde_nm= @(l, ampl, sigma) K_nm(l, ampl) + sigma^2 * eye(size(x,1), size(x,1)); % Kernel with noise
        dK_l1_nm = @(l,ampl) pdist2(x(:,1), x(:,1)).^2 / l(1)^3 .* K_nm(l, ampl);
        dK_l2_nm = @(l,ampl) pdist2(x(:,2), x(:,2)).^2 / l(2)^3 .* K_nm(l, ampl);
        dK_ampl_nm = @(l, ampl) 2 / ampl * K_nm(l, ampl);
        dK_sigma_nm = @(sigma) 2 * sigma * eye(size(x,1), size(x,1));
        NLL_nm = @(l, ampl, sigma) sum(log(diag(chol(K_tilde_nm(l, ampl, sigma))))) +...
            0.5 * y' / K_tilde_nm(l, ampl, sigma) * y + TRAIN_POINTS/2*log(2 * pi);


        % Transforming hyperparams to exp
        lambda_old = exp(lambda_old);
        K_tilde_tmp = K_tilde_nm(lambda_old(1:2), lambda_old(3), lambda_old(4));
        dK_l1_tmp = dK_l1_nm(lambda_old(1:2), lambda_old(3));
        dK_l2_tmp = dK_l2_nm(lambda_old(1:2), lambda_old(3));
        dK_ampl_tmp = dK_ampl_nm(lambda_old(1:2), lambda_old(3));
        dK_sigma_tmp = dK_sigma_nm(lambda_old(4));
        dNLL_l1 = 0.5 * trace(K_tilde_tmp \ dK_l1_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_l1_tmp) / K_tilde_tmp) * y;
        dNLL_l2 = 0.5 * trace(K_tilde_tmp \ dK_l2_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_l2_tmp) / K_tilde_tmp) * y;
        dNLL_ampl = 0.5 * trace(K_tilde_tmp \ dK_ampl_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_ampl_tmp) / K_tilde_tmp) * y;
        dNLL_sigma = 0.5 * trace(K_tilde_tmp \ dK_sigma_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_sigma_tmp) / K_tilde_tmp) * y;
        dnll = [ dNLL_l1; dNLL_l2; dNLL_ampl; dNLL_sigma ];

        nll = NLL_nm( lambda_old(1:2), lambda_old(3), lambda_old(4));
        
    end
    
end

function [opt_sol, fval, out_struct] = trainGP_SR1(lambda_old_sr1, x, y)
    
    TRAIN_POINTS = size(x,1);
    [opt_sol, fval, out_struct] = SR1(@cost_fun, lambda_old_sr1);
%     [opt_sol, fval, out_struct] = SR1Conditioning(@cost_fun, lambda_old_sr1);
    
    function [nll, dnll] = cost_fun(lambda_old)

        K_bfgs = @(l, ampl) ampl^2 * exp(-0.5 * pdist2(x, x, 'mahalanobis', diag(l.^2)).^2) + 1e-6 * eye(size(x,1), size(x,1)); % added some jitter...
        K_tilde_bfgs= @(l, ampl, sigma) K_bfgs(l, ampl) + sigma^2 * eye(size(x,1), size(x,1)); % Kernel with noise
        dK_l1_bfgs = @(l,ampl) pdist2(x(:,1), x(:,1)).^2 / l(1)^3 .* K_bfgs(l, ampl);
        dK_l2_bfgs = @(l,ampl) pdist2(x(:,2), x(:,2)).^2 / l(2)^3 .* K_bfgs(l, ampl);
        dK_ampl_bfgs = @(l, ampl) 2 / ampl * K_bfgs(l, ampl);
        dK_sigma_bfgs = @(sigma) 2 * sigma * eye(size(x,1), size(x,1));
        NLL_bfgs = @(l, ampl, sigma) sum(log(diag(chol(K_tilde_bfgs(l, ampl, sigma))))) +...
            0.5 * y' / K_tilde_bfgs(l, ampl, sigma) * y + TRAIN_POINTS/2*log(2 * pi);


        % Transforming hyperparams to exp
        lambda_old = exp(lambda_old);
        K_tilde_tmp = K_tilde_bfgs(lambda_old(1:2), lambda_old(3), lambda_old(4));
        dK_l1_tmp = dK_l1_bfgs(lambda_old(1:2), lambda_old(3));
        dK_l2_tmp = dK_l2_bfgs(lambda_old(1:2), lambda_old(3));
        dK_ampl_tmp = dK_ampl_bfgs(lambda_old(1:2), lambda_old(3));
        dK_sigma_tmp = dK_sigma_bfgs(lambda_old(4));
        dNLL_l1 = 0.5 * trace(K_tilde_tmp \ dK_l1_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_l1_tmp) / K_tilde_tmp) * y;
        dNLL_l2 = 0.5 * trace(K_tilde_tmp \ dK_l2_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_l2_tmp) / K_tilde_tmp) * y;
        dNLL_ampl = 0.5 * trace(K_tilde_tmp \ dK_ampl_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_ampl_tmp) / K_tilde_tmp) * y;
        dNLL_sigma = 0.5 * trace(K_tilde_tmp \ dK_sigma_tmp) - 0.5 * y' * ((K_tilde_tmp \ dK_sigma_tmp) / K_tilde_tmp) * y;
        dnll = [ dNLL_l1; dNLL_l2; dNLL_ampl; dNLL_sigma ];

        nll = NLL_bfgs( lambda_old(1:2), lambda_old(3), lambda_old(4));
        
    end
    
end

function out = d2LogLH_deta2 (p_coeff, eta_sol, n, m)

    out = zeros(size(eta_sol));
    for idx = 1:length(eta_sol)
        out(idx) = 0.5 * (n - m) / eta_sol(idx) * ( 2 * p_coeff(1) + 3 * p_coeff(2) / eta_sol(idx) + 4 * p_coeff(3) / eta_sol(idx)^2 ...
            + 5 * p_coeff(4) / eta_sol(idx)^3);
    end
end