%% Fast approximate Gaussian Process (FAGP)

% clear
% clc
close all
set(0,'defaultTextInterpreter','latex');
set(0, 'defaultAxesTickLabelInterpreter','latex');
set(0, 'defaultLegendInterpreter','latex');

rng('default');

% TRAIN_POINTS = N^2;% 100^2;
TRAIN_POINTS = 50^2;
sigma = 1e-3;
x = linspace(-5, 5, TRAIN_POINTS)'; 
y_real = @(x) x .* sin(2 * x);
% y_real = @(x) sin(2 * x);
% y_real = @(x) 0.8 + ( x + 0.2 ) .* ( 1 - 5 ./ (1 + exp(-2 * x)));
% y_real = @(x) sin(x.^2 / 2);
% y_real = @(x) sin(x .* sqrt(abs(x)));
y = y_real(x) + randn(TRAIN_POINTS,1) * sigma;
X = linspace(-5, 5, 200)'; %200
%normalizing
y = (y - min(y)) / (max(y) - min(y));
y_star = y_real(X);
y_star = (y_star - min(y_star)) / (max(y_star) - min(y_star));


%%

% Define the points under consideration
n = 3; % # of eigenvalues (3)

% Classic GP
l = 15;%3; % Scale factor
epsilon = 1/(sqrt(2)*l); % Parameter depending on scale factor
tic
K = exp(-epsilon^2*pdist2(x, x).^2);
Ks = exp(-epsilon^2*pdist2(X, x).^2);
Kss = exp(-epsilon^2*pdist2(X, X).^2);

ys_std = Ks*( (K + sigma^2*eye(size(x,2)))\y );
cov_std = Kss - Ks/(K + sigma^2*eye(size(x,2)))*Ks';
elapsed_classic = toc;
fprintf("Classic GP took %fs, RMSE = %.4e\n", elapsed_classic, sqrt( sum((ys_std - y_star).^2)/length(ys_std) ));
% toc

%% FAGP
% Define the phi functions
% phi accepts row vector n and column vector x
%     it returns a length(x)-by-length(n) matrix

l = 1;% 0.039; % Scale factor
alpha = 0.5;% 0.009;%sqrt(2); % Global scale factor
epsilon = 1/(sqrt(2)*l); % Parameter depending on scale factor
R = 1;
ii = 0;

for eigv = n:3:15
    ii = ii + 1;
    tic
    [ K_tilde, cov_ys, K_app, Ks_app, lambdas, indices, times(ii) ] = approximateKernel(x, X, eigv, epsilon, alpha);

    ys = K_tilde*y;
    elapsed(ii, 1) = toc;

    str_title = "FAGP with " + num2str(eigv) + " eigs";
    figure
%     title(str_title);
    hold on,grid on
    plot(X, ys)
    plot(x, y, '--')
    xlabel('x')
    ylabel('y')
    legend('GP mean', 'Real function', 'Location', 'north')
    
%     pause(2);
    
    % Comparison between the two methods
    RMSE = sqrt( sum((ys - y_star).^2)/length(ys) );
    fprintf("RMSE with %d eigenvalues is %.4e. Lambda_hat is a %dx%d matrix. Run in %fs\n", eigv, RMSE, size(indices,1), size(indices,1), elapsed(ii));
end

figure
hold on, grid on
title('Classic GP');
plot(X, ys_std)
plot(x, y, '--')
        
% figure('Position', [800, 400, 1000, 400])
% subplot(1,2,1)
% hold on
% surface(X1, X2, reshape(ys_std, 50, 50) - max(ys_std), 'FaceColor','interp','EdgeColor','interp');
% plot(x(1,y==1), x(2,y==1), '.','markersize',28,'color',[.8 0 0]); %Interior points
% plot(x(1,y==0), x(2,y==0), '.','markersize',28,'color',[.8 .4 0]); %Border points
% plot(x(1,y==-1), x(2,y==-1), '.','markersize',28,'color',[0 .6 0]); %Exterior points
% contour(X1, X2, reshape(ys_std, 50, 50), [0,0], 'linewidth',2,'color','white');
% title('Classic GP formula');
% subplot(1,2,2)
% hold on
% surface(X1, X2, reshape(diag(cov_std), 50, 50) - max(diag(cov_std)), 'FaceColor','interp','EdgeColor','interp');
% plot(x(1,y==1), x(2,y==1), '.','markersize',28,'color',[.8 0 0]); %Interior points
% plot(x(1,y==0), x(2,y==0), '.','markersize',28,'color',[.8 .4 0]); %Border points
% plot(x(1,y==-1), x(2,y==-1), '.','markersize',28,'color',[0 .6 0]); %Exterior points
% contour(X1, X2, reshape(ys_std, 50, 50), [0,0], 'linewidth',2,'color','white');

%% Approximated Gaussian Kernel

function [ K_tilde, covariance, K_approx, Ks_approx, lambda_comb, idx_comb, times ] ...
    = approximateKernel(x, xp, n, ep, alpha)

    sigma = 1e-3;
    % Combinations
    [ index1 ] = ndgrid(1:n);
    idx_comb = index1(:);
%     phi_comb = zeros(size(x,1), size(idx_comb,1));
%     phip_comb = zeros(size(xp,1), size(idx_comb,1));
%     lambda_comb = zeros(1,size(idx_comb,1));

    phi_comb = eigenFnct(x(:,1), idx_comb(:,1)', ep, alpha);
        
    phip_comb = eigenFnct(xp(:,1), idx_comb(:,1)', ep, alpha);
    lambda_comb = eigenValue(idx_comb(:,1), ep, alpha);
    
    
    inv_SigmaN = 1./sigma^2*eye(size(x,1));
    Lambda_hat = 1/sigma^2 * (phi_comb'*phi_comb) + diag(1./lambda_comb);
    
%     tmp_inv = inv(Lambda_hat);
    
    tstart_inv = tic;
    tmp = (inv_SigmaN - 1/sigma^2*phi_comb/Lambda_hat*phi_comb'* 1/sigma^2);
    times.elapsed_inv = toc(tstart_inv);
    tstart_mult = tic;
    K_tilde = phip_comb*diag(lambda_comb)*phi_comb'* tmp;
    times.elapsed_mult = toc(tstart_mult);
    covariance = phip_comb*diag(lambda_comb)*phip_comb' - K_tilde*phi_comb*diag(lambda_comb)*phip_comb';
    
    K_approx = phi_comb*diag(lambda_comb)*phi_comb';
    Ks_approx = phip_comb*diag(lambda_comb)*phi_comb';

end

function phi = eigenFnct(x_1D, n_eigv, ep, alpha)

    % Compute the eigenfunction phi corresponding to the eigenvalue n_eigv
    % using one of the dimensions of x "x_1D".
    beta = (1 + (2*ep/alpha)^2)^0.25;
    Gamma = sqrt(beta./(2.^(n_eigv-1).*gamma(n_eigv)));
    delta2 = alpha^2/2*(beta^2 - 1);
    
    phi = exp(-delta2*x_1D.^2)*Gamma.*hermiteH_user(n_eigv-1, alpha*beta*x_1D);
end

function lambda = eigenValue(n_eigv, ep, alpha)
    % Decreasing eigenvalues computation
    beta = (1 + (2*ep/alpha)^2)^0.25;
    delta2 = alpha^2/2*(beta^2 - 1);
    lambda = sqrt(alpha^2/(alpha^2 + delta2 + ep^2))*(ep^2/(alpha^2 + delta2 + ep^2)).^(n_eigv-1);
end

function [y, H0] = hermiteH_user(n, x)

    % Hermite polynomial function.
    % n is the row vector of the desired degrees of the hermite polynomial,
    % x is the column vector of the input. 
    x = sqrt(2)*x;
    y = zeros(size(x,1), length(n));
    for j = 1:length(n)
        H0 = 1;
        H1 = [1,0];
        for i = 1:n(j)
            H0_1 = i * H0;
            H0_2 = conv ( [0,0,1], H0_1 );
            H2 = conv( [1,0], H1 ) - H0_2;
            H0 = H1;
            H1 = H2;
        end
        for i = 1:n(j)+1
            y(:,j) = y(:,j) + x.^(n(j)+1-i) * H0(i);
        end
        y(:,j) = 2^(n(j)/2)*y(:,j);
    end
end
