%% GPNARX devastator
clear, clc, close all
set(0,'defaultTextInterpreter','latex');
set(0, 'defaultAxesTickLabelInterpreter','latex');
set(0, 'defaultLegendInterpreter','latex');

% The aim is to obtain the model identification of the Devastator Robot

% STEP 1: SISO model
% y(t) = f(x,u,t)
% with 
% u -> input  of PWM             [us]       in R^1
% y -> output of angular speed   [rad/s]    in R^1
addpath("sysId_devastator/");

% Experiments
% Case I
% load('Test_1.mat')
% load('Test_2.mat')
% load('Test_3.mat')
% load('Test_4.mat')
% load('test_5.mat')
load('test_7.mat')
% load('test_8.mat')
% load('test_9.mat')
% load('test_10.mat')

% Case II
load('test_7.mat')
omega_l = omega_r;
pwm_l = pwm_r;
pwm_l1 = pwm_l;
omega_l1 = omega_l;
t1 = t;
% load('Test_4.mat')
% pwm_l2 = pwm_l;
% omega_l2 = omega_l;
% t2 = t;
load('Test_2.mat')
omega_l = omega_r;
pwm_l = pwm_r;
pwm_l3 = pwm_l;
omega_l3 = omega_l;
t3 = t;
load('test_9.mat')
omega_l = omega_r;
pwm_l = pwm_r;
pwm_l4 = pwm_l;
omega_l4 = omega_l;
t4 = t;
load('test_8.mat')
omega_l = omega_r;
pwm_l = pwm_r;
pwm_l5 = pwm_l;
omega_l5 = omega_l;
t5 = t;
pwm_l = [pwm_l1; pwm_l3; pwm_l4; pwm_l5];
omega_l = [omega_l1; omega_l3; omega_l4; omega_l5];
t = [t1; t3+t1(end); t4+t3(end)+t1(end); t5+t4(end)+t3(end)+t1(end)];

% In every test,
% t       -> time                        [s]
% pwm_l   -> PWM values for Left Motor   [us]
% pwm_r   -> PWM values for Right Motor  [us]
% omega_l -> Angular speed for LM        [rad/s]
% omega_r -> Angular speed for RM        [rad/s]

%% DATA SELECTION

% Suppose to create a model for the LEFT side
% Output
% y = omega_l./20;
y = (omega_l - mean(omega_l))/(max(omega_l) - min(omega_l));
% Input
% u = pwm_l./20000;
u = (pwm_l - mean(pwm_l))/(max(pwm_l) - min(pwm_l));

% The selected model is a NARX model
% Input training vector
% x = [y(k-1) , y(k-2), ..., y(k-delay_y), u(k-1), u(k-2), ..., u(k-delay_u)]

% Selected delay: it is an arbitrary choise
delay_y = 3;
delay_u = 3;
delay_max = max(delay_y, delay_u);

% Creation of the input training vector 
% As many rows as the tests data (considering the delay)
% As many columns as the elements in the regressor at time k 
dim_t = size(u,1) - delay_max;
x = zeros(dim_t,delay_y+delay_u); 

% Training points for the model: 
% First iteration
% -> x(1,:) = [y(3:-1:1), u(1:-1:1)] = [y(3) y(2) y(1) u(1)]
% Second iteration
% -> x(2,:) = [y(4:-1:2), u(2:-1:2)] = [y(4) y(3) y(2) u(2)]
for ii = 1:dim_t
    x(ii,:) = [y(delay_y+ii-1:-1:ii)', u(delay_u+ii-1:-1:ii)'];
end

% Output and time
% The last output is the one that it is possible to predict considering the chosen delay 
y = y(delay_y+1:end);
t_y = t(delay_y+1:end);

% TRAINING DATA
% x -> input of the model       in R^(dim_t,delay_y+delay_u)
% y -> output of the model      in R^(dim_t,1)

% READY to FIT the model!
% But first...
% HYPOTHESIS and THEORY

% A Gaussian process regression problem can be written as
%  f(x) = GP(m(x), k(x,x'))
%  yk = f(xk) + epsilon
%  GP Gaussian process of m(x) mean and k(x,x') covariance function or kernel 
%  epsilon an indipendent noise of 0 mean and sigma^2 variance

% It is a choise to select MEAN and COVARIANCE
% Mean usually set to 0
% Covariance chosen between different kernel
% Suppose to use a squared exponential covariance function as kernel
% k(x,x') = s^2 exp( -1/2 ||x-x'||^2/l^2)
% s^2 covariance of the regressor fuctions
% l length scales of the regressor functions

% The Gaussian process regression is concerned with the following problem: 
% Given a set of observed (training) input-output data D = {(xk, yk) : k = 1, . . . , N}
% from an unknown function y = f(x), predict the values of the function at new (test) inputs
% {xk*: k = 1, . . . , M}
% Fundamental equations of Gaussian process regression
% E[f(x*)|y]   = m(x*) + k(x*)(k+sigma^2I)^(-1)(y-m(x)) 
% Cov[f(x*)|y] = k(x*,x*) - k(x)(k+sigma^2)^(-1) k'(x*)

% The mean value represents the most likely output 
% The variance can be interpreted as the measure of its confidence

% f(x) can be modeled as a basis function expansion of the Gaussian process
% In this case coefficients beta are introduced


%% TRAINING 

% GP MODEL CREATION
%fitrgp   Fit a Gaussian Process Regression (GPR) model.
%         fitrgp(x,y) accepts
%         - X as an N-by-P matrix of predictors with
%                - one row per observation   
%                - one column per predictor
%         - Y is the response vector

% The RGP Model is characterized by:
% 'FitMethod': Method used to estimate the basis function coefficients, β;
%              noise standard deviation, σ; 
%              and kernel parameters, θ.
% 'BasisFunction': Explicit basis function used in the GPR model
% 'Beta': Estimated coefficients for the explicit basis functions
% 'Sigma': Estimated noise standard deviation of the GPR model
% 'LogLikelihood': Maximized marginal log likelihood of the GPR model
% 'KernelFunction': Form of the covariance function 
% 'KernelInformation': Information about the parameters of the kernel function
% 'PredictMethod': Method that predict uses to make predictions from the GPR model
% 'Alpha': Weights used to make predictions
% 'ActiveSetVectors: Subset of training data used to make predictions from the GPR model


%     Actually used options: 
%     - 'FitMethod'      -> 'Exact' 
%     - 'BasisFunction'  -> 'Constant', ie H=1
%     - 'KernelFunction' -> 'SquaredExponential'
%     - 'PredictMethod'  -> 'Exact'
tic      
l0 = [ 1; 1; 1; 1; 1; 1 ]*3; % [ l_y(k-1) l_y(k-2) l_y(k-3) l_u(k-1) l_u(k-2) l_u(k-3) ]
% l0 = [ 1; 1; 1; 1 ]*3;
theta_init = log([ l0; 1; 1 ]);
ampl0 = 1;
sigma0 = 1;
[ ampl0, sigma0 ] = noise_est(x, y, theta_init);
est_time = toc;
fprintf("Hyp est time: %f\n", est_time);
theta0 = log([ l0; ampl0; sigma0 ]);

tic
theta_opt = trainGP(theta0, x, y);
% gpr = fitrgp(x, y);
train_time = toc;
fprintf("Training time: %f\nTotal time: %f\n", train_time, train_time+est_time);
% GP MODEL obtained!
% Let's go to see how good it is


%% VALIDATION ON TRAINING

% Predict on train data

% Define some handles. In training validation x_star = x
x_star = x;
K = @(l, ampl) ampl^2 * exp(-0.5 * pdist2(x, x, 'mahalanobis', diag(l.^2)).^2);% ...
%     + 1e-6 * eye(size(x,1), size(x,1)); % added some jitter...
K_tilde = @(l, ampl, sigma) K(l, ampl) + sigma^2 * eye(size(x,1), size(x,1)); % Kernel with noise
K_star = @(l,ampl) ampl^2 * exp(-0.5 * pdist2(x_star, x, 'mahalanobis', diag(l.^2)).^2);
K_starstar = @(l,ampl) ampl^2 * exp(-0.5 * pdist2(x_star, x_star,'mahalanobis', diag(l.^2)).^2);

tic
theta_opt_k = exp(theta_opt); % Kernel hyperparameters

fprintf("Hypparams: l = [%f %f %f %f %f %f]; ampl = %f; sigma = %f\n", ...
    theta_opt_k(1:end-2), theta_opt_k(end-1), theta_opt_k(end));
ypred_train = K_star(theta_opt_k(1:end-2), theta_opt_k(end-1)) / ...
    K_tilde(theta_opt_k(1:end-2), theta_opt_k(end-1), theta_opt_k(end)) * y;
cov_star = K_starstar(theta_opt_k(1:end-2), theta_opt_k(end-1)) - K_star(theta_opt_k(1:end-2), theta_opt_k(end-1)) / ...
    K_tilde(theta_opt_k(1:end-2), theta_opt_k(end-1), theta_opt_k(end)) * K_star(theta_opt_k(1:end-2), theta_opt_k(end-1))';
cov_star_vec_train = diag(cov_star);

% [ypred_train, ysd_train, yint_train] = predict(gpr, x);
toc
figure(2)
hold on
grid on
scatter(t_y, ypred_train,'g')                   % GPR predictions
plot(t_y, y, 'r');
plot(t_y, ypred_train, 'g');
patch([t_y;flipud(t_y)], [ypred_train + 2*sqrt(cov_star_vec_train);...
    flipud(ypred_train - 2*sqrt(cov_star_vec_train))],  'k', 'FaceAlpha', 0.1);
% patch([t_y;flipud(t_y)],[yint_train(:,1);flipud(yint_train(:,2))],'k','FaceAlpha',0.1); % Prediction intervals
legend('Prediction', 'Real data');
title('Validation on TRAINING');
xlabel('Time [s]');
ylabel('Angular speed [rad/s]')
hold off

rmse_gp  = sqrt(sum((ypred_train*20 - y*20).^2)/length(y))

fit_gp = (1-sqrt(sum((ypred_train - y).^2)/sum((ypred_train - mean(y)).^2)))*100

%%  VALIDATION ON TEST

% load('Test_1.mat')
% load('Test_2.mat')
% load('Test_3.mat')
% load('Test_4.mat')
% load('test_5.mat')
% load('test_7.mat')
load('test_8.mat')
% load('test_9.mat')
% load('test_10.mat')

omega_l = omega_r;
pwm_l = pwm_r;
omega_l_norm = (omega_l - mean(omega_l))/(max(omega_l) - min(omega_l));
pwm_l_norm = (pwm_l - mean(pwm_l))/(max(pwm_l) - min(pwm_l));

% Data preparation
delay_max = max(delay_y, delay_u);
dim_v = size(pwm_l,1) - delay_max;
xs = zeros(dim_v,delay_y+delay_u);
ys = omega_l(delay_y+1:end,1);
ts = t(delay_y+1:end);

% Test points 
for ii = 1:dim_v
    xs(ii,:) = [omega_l_norm(delay_y+ii-1:-1:ii,1)', pwm_l_norm(delay_u+ii-1:-1:ii)'];
end

% Prediction
% -ypred    ->  predicted responses for the Gaussian process regression model
% -ysd      ->  standard deviation of the process
% -yint     ->  95% prediction intervals of the response variable

% Define some handles. In test validation x_star = xs
x_star = xs;
K = @(l, ampl) ampl^2 * exp(-0.5 * pdist2(x, x, 'mahalanobis', diag(l.^2)).^2);% ...
%     + 1e-6 * eye(size(x,1), size(x,1)); % added some jitter...
K_tilde = @(l, ampl, sigma) K(l, ampl) + sigma^2 * eye(size(x,1), size(x,1)); % Kernel with noise
K_star = @(l,ampl) ampl^2 * exp(-0.5 * pdist2(x_star, x, 'mahalanobis', diag(l.^2)).^2);
K_starstar = @(l,ampl) ampl^2 * exp(-0.5 * pdist2(x_star, x_star,'mahalanobis', diag(l.^2)).^2);

tic
% theta_opt_k = exp(theta_opt); % Kernel hyperparameters

ypred = K_star(theta_opt_k(1:end-2), theta_opt_k(end-1)) / ...
    K_tilde(theta_opt_k(1:end-2), theta_opt_k(end-1), theta_opt_k(end)) * y;
cov_star = K_starstar(theta_opt_k(1:end-2), theta_opt_k(end-1)) - K_star(theta_opt_k(1:end-2), theta_opt_k(end-1)) / ...
    K_tilde(theta_opt_k(1:end-2), theta_opt_k(end-1), theta_opt_k(end)) * K_star(theta_opt_k(1:end-2), theta_opt_k(end-1))';
cov_star_vec = diag(cov_star);


% [ypred, ysd, yint] = predict(gpr, xs);
toc
% index = 125;
index = length(ys)-12;

ypred = ypred * (max(omega_l) - min(omega_l)) ...
    + mean(omega_l);
std_dev_star_vec = sqrt(cov_star_vec) * (max(omega_l) - min(omega_l)) ...
    + mean(omega_l);

figure
hold on
grid on
plot(ts(1:index), ys(1:index), 'r','Linewidth',1.5);
plot(ts(1:index), ypred(1:index),'-o','Linewidth',1,'Color',[0 0.7 0])%,'MarkerSize',5,'MarkerEdgeColor', [0 0.7 0],'MarkerFaceColor',[1, 1, 1])
% patch([ts(1:index);flipud(ts(1:index))],[yint((1:index),1);flipud(yint((1:index),2))].*20,'--k','FaceAlpha',0.1);
% patch([ts(1:index);flipud(ts(1:index))], [(ypred(1:index)*20 + 2*sqrt(cov_star_vec(1:index))*20);...
%     flipud(ypred(1:index)*20 - 2*sqrt(cov_star_vec(1:index))*20)], 'k', 'EdgeColor','None','Linestyle',':', 'FaceAlpha', 0.1);
% title('Validation');
legend('Measured output','Predicted output');
xlabel('Time [s]');
ylabel('Angular speed [rad/s]')
xlim([0 ts(index+1)])
hold off

rmse_gp  = sqrt(sum((ypred - ys).^2)/length(ys))

fit_gp = (1-sqrt(sum((ypred - ys).^2)/sum((ypred - mean(ys)).^2)))*100

%%
figure
hold on
grid on
plot(ts(1:index), ypred(1:index),'Linewidth',1, 'Color',[0, 0, 0])
% plot(ts(1:index), ys(1:index), 'r','Linewidth',1.5);
patch([ts(1:index);flipud(ts(1:index))], [(ypred(1:index) + 2*std_dev_star_vec(1:index));...
    flipud(ypred(1:index) - 2*std_dev_star_vec(1:index))], 'k', 'EdgeColor','k','Linestyle',':', 'FaceAlpha', 0.3);
% patch([ts(1:index);flipud(ts(1:index))],[yint((1:index),1);flipud(yint((1:index),2))].*20,'k','EdgeColor','k','Linestyle',':','FaceAlpha',0.3); 
% title('Validation');
% legend('Predicted output', '95% confidence interval');
xlabel('Time [s]');
ylabel('Angular speed [rad/s]')
xlim([0 ts(index+1)])
hold off
% 
figure
% title('Variance of y');
hold on, grid on
plot(ts(1:index), 2*std_dev_star_vec(1:index));
xlabel('Time [s]');
ylabel('$\sigma$ [rad/s]')
xlim([0, 51])

figure
hold on, grid on;
plot(ts(1:index), pwm_l(1:index)/20000);
xlabel('Time [s]')
ylabel('Normalized command [$\cdot$]')
xlim([0, 51])

%% Functions

% Training GPIS function, recalls the function for computing the Negative
% Log Likelihood and its derivatives
function [ theta_out, fval, exitFlag, output_struct ] = trainGP(theta0, x, y)

    options = optimoptions(@fminunc,'Display','iter-detailed');%, 'HessUpdate', {'lbfgs', 10}, 'MaxIter', 1000);
%     options = optimset;
    options.Display = 'iter-detailed';
    options.Algorithm = 'trust-region';
    options.HessianApproximation = {'lbfgs', 100};
%     options.HessianApproximation = 'bfgs';
    options.SpecifyObjectiveGradient = true;
    options.MaxIterations = 1000;
    options.UseParallel = true;

    options_nm = optimset('Display','iter','MaxFunEvals', 10000);
    
    
%     options.HessUpdate = "lbfgs";
%     [ theta_out, fval, exitFlag, output_struct ] = fminunc(@cost_fun, theta0, options);
    [ theta_out, fval, output_struct ] = SR1(@cost_fun, theta0, true);
%     [ theta_out, fval, output_struct ] = SR1Conditioning(@cost_fun, theta0, true);
%     [ theta_out, fval, exitFlag, output_struct ] = fminsearch(@cost_fun, theta0, options_nm);
%     [ optimum, ~ ] = fminadam(@cost_fun, params0);
%     [ optimum, ~, ~, ~ ] = fmin_adam(@cost_fun, params0, [], [], [], [], [], options);
%     [ theta_out, fval, exitFlag, output_struct ] = particleswarm(@cost_fun, length(theta0), ...
%         [-1000, -1000, -1000, -10000, -1, -1, -1000], [2, 2, 2, 1, 1, 1, 100]);
%     [ theta_out, fval, exitFlag, output_struct ] = ga(@cost_fun, length(theta0), [], [], [], [], ...
%         log([0.1 0.1 0.1 0.1 0.1 0.1 0.001 1e-3]), []);
    
    function [nll, dnll] = cost_fun(params)
        [nll, dnll] = NLL(params, x, y);
    end
end

% Negative Log Likelihood and its derivatives
function [nll, dnll] = NLL(theta, x, y)

    if size(theta, 2) ~= 1
        theta = theta';
    end
    TRAIN_POINTS = size(x, 1);

    theta = exp(theta);

    K = @(l, ampl) ampl^2 * exp(-0.5 * pdist2(x, x, 'mahalanobis', diag(l.^2)).^2);% ...
%         + 1e-6 * eye(size(x,1), size(x,1)); % added some jitter...
    K_tilde = @(l, ampl, sigma) K(l, ampl) + sigma^2 * eye(size(x,1), size(x,1)); % Kernel with noise
    dK_l = @(l,ampl,dim) pdist2(x(:,dim), x(:,dim)).^2 / l(dim)^3 .* K(l, ampl);
%     dK_l2 = @(l,ampl) pdist2(x, x).^2 / l(2)^3 .* K(l, ampl);
    dK_ampl = @(l, ampl) 2 / ampl * K(l, ampl);
    dK_sigma = @(sigma) 2 * sigma * eye(size(x,1), size(x,1));
    
    NLL = @(l, ampl, sigma) sum(log(diag(chol(K_tilde(l, ampl, sigma))))) +...
        0.5 * y' / K_tilde(l, ampl, sigma) * y + TRAIN_POINTS/2*log(2 * pi);

    nll = NLL( theta(1:end-2), theta(end-1), theta(end));
    K_tilde_tmp = K_tilde(theta(1:end-2), theta(end-1), theta(end));
    dK_l1 = dK_l(theta(1:end-2), theta(end-1), 1);
    dK_l2 = dK_l(theta(1:end-2), theta(end-1), 2);
    dK_l3 = dK_l(theta(1:end-2), theta(end-1), 3);
    dK_l4 = dK_l(theta(1:end-2), theta(end-1), 4);
    dK_l5 = dK_l(theta(1:end-2), theta(end-1), 5);
    dK_l6 = dK_l(theta(1:end-2), theta(end-1), 6);
    dK_ampl = dK_ampl(theta(1:end-2), theta(end-1));
    dK_sigma = dK_sigma(theta(end));
    dNLL_l1 = 0.5 * trace(K_tilde_tmp \ dK_l1) - 0.5 * (y)' * ((K_tilde_tmp \ dK_l1) / K_tilde_tmp) * (y);
    dNLL_l2 = 0.5 * trace(K_tilde_tmp \ dK_l2) - 0.5 * (y)' * ((K_tilde_tmp \ dK_l2) / K_tilde_tmp) * (y);
    dNLL_l3 = 0.5 * trace(K_tilde_tmp \ dK_l3) - 0.5 * (y)' * ((K_tilde_tmp \ dK_l3) / K_tilde_tmp) * (y);
    dNLL_l4 = 0.5 * trace(K_tilde_tmp \ dK_l4) - 0.5 * (y)' * ((K_tilde_tmp \ dK_l4) / K_tilde_tmp) * (y);
    dNLL_l5 = 0.5 * trace(K_tilde_tmp \ dK_l5) - 0.5 * (y)' * ((K_tilde_tmp \ dK_l5) / K_tilde_tmp) * (y);
    dNLL_l6 = 0.5 * trace(K_tilde_tmp \ dK_l6) - 0.5 * (y)' * ((K_tilde_tmp \ dK_l6) / K_tilde_tmp) * (y);
    dNLL_ampl = 0.5 * trace(K_tilde_tmp \ dK_ampl) - 0.5 * (y)' * ((K_tilde_tmp \ dK_ampl) / K_tilde_tmp) * (y);
    dNLL_sigma = 0.5 * trace(K_tilde_tmp \ dK_sigma) - 0.5 * (y)' * ((K_tilde_tmp \ dK_sigma) / K_tilde_tmp) * (y);
    
    dnll = [ dNLL_l1; dNLL_l2; dNLL_l3; dNLL_l4; dNLL_l5; dNLL_l6; dNLL_ampl; dNLL_sigma ];
%     dnll = [ dNLL_l1; dNLL_l2; dNLL_l3; dNLL_l4; dNLL_ampl; dNLL_sigma ];

end

function [ampl0, sigma0] = noise_est(x, y, theta0)

    % Noise estimation function
    % Input args:
    %       x       Training locations matrix (TRAIN_POINTS x dims)
    %       y       Observations vector (TRAIN_POINTS x 1)
    %       theta0  Initial hyperparameters vector
    % Output args:
    %       ampl0   Initial amplitude
    %       sigma0  Initial noise variance


    TRAIN_POINTS = size(x, 1);
    K = @(l, ampl) ampl^2 * exp(-0.5 * pdist2(x, x, 'mahalanobis', diag(l.^2)).^2);% ...
%         + 1e-6 * eye(size(x,1), size(x,1));

    % eta = sigma_noise^2 / sigma_cov^2
    
    % Asymptote of LogLH 1st derivative
    m = 7; % Number of basis functions
    X = [ ones(TRAIN_POINTS, 1), x(:,1), x(:,2), x(:,3), x(:,4), x(:,5), x(:,6) ];      % Design Matrix obtained from basis fun phi(x(1:end))
    Q = eye(TRAIN_POINTS) - X * pinv(X);
    K_tmp = K(exp(theta0(1:end-2)), exp(theta0(end-1)));
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
    ampl0 = sqrt(sigma_opt);
    sigma0 = sqrt(sigma0_opt);

end

function out = d2LogLH_deta2 (p_coeff, eta_sol, n, m)

    out = zeros(size(eta_sol));
    for idx = 1:length(eta_sol)
        out(idx) = 0.5 * (n - m) / eta_sol(idx) * ( 2 * p_coeff(1) + 3 * p_coeff(2) / eta_sol(idx) + 4 * p_coeff(3) / eta_sol(idx)^2 ...
            + 5 * p_coeff(4) / eta_sol(idx)^3);
    end
end

function K = kernelFnct(x1, x2, l, sigma)
% Different choices are available:
% NOTATION: l is the characteristic length scale
%           sigma is the signal standard deviation
% 
% -Squared Exponential Kernel
%     K = sigma^2 exp[-1/2 (xi-xj)'(xi-xj)/l^2)
% -Exponential Kernel
%     K = sigma^2 exp(- sqrt((xi-xj)'(xi-xj))/l)
% -Mater 3/2 
%     K = sigma^2 (1 + sqrt(3)/l*sqrt((xi-xj)'(xi-xj))).*exp(-sqrt(3)/l*sqrt((xi-xj)'(xi-xj)))
% -Matern 5/2
%     K = sigma^2 (1 + sqrt(5)/l*sqrt((xi-xj)'(xi-xj)) + 5/(3*l^2)*sqrt((xi-xj)'(xi-xj)).^2).*exp(-sqrt(5)/l*sqrt((xi-xj)'(xi-xj)))
% -Radial basis function kernel: parameters sigm and l
%     K = sigm*exp(-norm(x1-x2).^2/(2*l^2)); 
%     K = sigm*exp(-(x1 - x2)*l*(x1 - x2)');
 
    K = zeros(size(x1,1),size(x2,1));
    for ii = 1:size(x1,1)
        for jj = 1:size(x2,1)
            K(ii,jj) = sigma^2*exp(-1/2*(x1(ii,:) - x2(jj,:))*(x1(ii,:) - x2(jj,:))'/l^2);
        end
    end
end