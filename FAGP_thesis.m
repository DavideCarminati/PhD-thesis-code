%% Fast approximate Gaussian Process (FAGP)

% clear
% % 
% [x1, x2, button] = ginput(30);
% % mouse left: space; 
% %       middle: countour; 
% %       right: obstacle.
% x = [x1'; x2'];
% y = button' - 2;

rng('default');

% load('real_samples.mat');
% x = optsamples1(:, 1:2);
% y = optsamples1(:, 3);
% y_noisefree = y;
 
% load('x_GPIS2.mat');
% load('y_GPIS2.mat');
% load('y_noisefree_GPIS2.mat');
% load('artificial_cloud_1.mat');
% load('merged_cloud.mat');
% y = y_noisefree';
% x = x';

% Math function
% N = 100;
x = linspace(-pi/2, pi/2, N);
% x = linspace(-5, 5, 50);
[ X1_train, X2_train ] = ndgrid(x,x);
x = [X1_train(:), X2_train(:) ]';
y = cos(X1_train(:)') + cos(X2_train(:)');
% y = X1_train(:)' .* sin(2 * X1_train(:)') + X2_train(:)' .* sin(2 * X2_train(:)');
y = y + randn(1, length(y)) * 1e-3;
% y = (y - min(y)) / (max(y) - min(y));

% % Complex obstacle
% 
% circleRadius = 1;
% circleCenter = [ 2.5; 2.5 ];
% points = linspace(-pi,pi,200);%-pi:pi/24:(pi-1e-3);
% x_edge = circleCenter + (circleRadius + 0.1*sin(points*10) + 0.11*cos(points*20 + 12) ...
%     + 1e-5*randn(1,length(points))) .* [cos(points); sin(points)];
% y_edge = zeros(1,size(x_edge,2));
% 
% x_full = 0.01*randn(2,1) + circleCenter;
% y_full = ones(size(x_full,2));
% 
% num_free_points = 8;
% x1_free = zeros(1,num_free_points);
% x2_free = zeros(1,num_free_points);
% y_free = zeros(1,num_free_points);
% jj = 0;
% while jj < num_free_points
% %     rng(seed);
%     jj = jj + 1;
%     x1_free(jj) = 0.05 + 4.95*rand(1);
%     x2_free(jj) = 0.05 + 4.95*rand(1);
%     y_free(jj) = -1;
%     % Checking if some of the free points are inside obstacles
%     for idx = 1:size(circleCenter,2)
%         dist_from_ctr = pdist2([x1_free(jj)', x2_free(jj)'], circleCenter(:,idx)');
%         if dist_from_ctr <= circleRadius(idx) + 0.2
%             x1_free(jj) = 0;
%             x2_free(jj) = 0;
%             y_free(jj) = 0;
%             jj = jj - 1;
%             break
%         end
%     end
% end
% 
% x = [ x_full, x_edge, [ x1_free; x2_free ] ]/5;
% y = [ y_full, y_edge, y_free ];

% figure
% hold on
% plot(x(1,y==1), x(2,y==1), '.','markersize',28,'color',[.8 0 0]); %Interior points
% plot(x(1,y==0), x(2,y==0), '.','markersize',28,'color',[.8 .4 0]); %Border points
% plot(x(1,y==-1), x(2,y==-1), '.','markersize',28,'color',[0 .6 0]); %Exterior points

%%

% x = [0.2, 0.4, 0.6, 0.4, 0.6, 0.9; ...
%      0.5, 0.5, 0.5,  0.8, 0.1, 0.6];
% y = [ -1,   0,   1,   -1,  -1,  -1];
% x = [0.2, 0.4, 0.6, 0.6, 0.9, 0.55, 0.66, 0.66, 0.79, 0.08, 0.12, 0.29, 0.87, 0.91; ...
%      0.5, 0.5, 0.5, 0.1, 0.6, 0.55, 0.54, 0.33, 0.58, 0.08, 0.30, 0.16, 0.12, 0.29];
% x = [0.2, 0.4, 0.6; ...
%      0, 0, 0]; % Aligned horizontally
% x = [0.5, 0.5, 0.5; ...
%      0.1, 0.5, 0.9]; % Aligned vertically
% y = [ -1,   0,   1,  -1,  -1,    1,    1,    0,    0,   -1,   -1,   -1,   -1,   -1];
% x = [0.1, 0.5, 0.9]; % 1D case
% y = [-1, 0, 1];
% x = [0.35; 0.27];
% y = -1;

% Define the points under consideration
n = 3; % # of eigenvalues
train_size = length(y);
% with n = 30, alpha = 3
% x = [0.2, 0.4, 0.6, 0.6, 0.9, 0.55, 0.66, 0.66, 0.79, 0.08, 0.12, 0.29, 0.87, 0.91; ...
%      0.5, 0.5, 0.5, 0.1, 0.6, 0.55, 0.54, 0.33, 0.58, 0.08, 0.30, 0.16, 0.12, 0.29];
% y = [ -1,   0,   1,  -1,  -1,    1,    1,    0,    0,   -1,   -1,   -1,   -1,   -1];

% X = [linspace(-1,1,50); linspace(-1,1,50)]';
% [ X1, X2 ] = meshgrid(linspace(0,1,50), linspace(0,1,50));
% [ X1, X2 ] = meshgrid(linspace(0, 4, 50), linspace(0, 4, 50));
[ X1, X2 ] = meshgrid(linspace(-pi/2, pi/2, 50), linspace(-pi/2, pi/2, 50));
% [ X1, X2 ] = meshgrid(linspace(0, 1, 50), linspace(-0.5, 0.5, 50));
X = [ X1(:), X2(:) ];
Y = cos(X1(:)') + cos(X2(:)');

% Classic GP
l = 0.5; % Scale factor
% l = 1;
epsilon = 1/(sqrt(2)*l); % Parameter depending on scale factor
tic
K = exp(-epsilon^2*pdist2(x', x').^2);
Ks = exp(-epsilon^2*pdist2(X, x').^2);
Kss = exp(-epsilon^2*pdist2(X, X).^2);

% Thin plate cov
% K = 2*abs(pdist2(x', x').^3) - 3*R*pdist2(x', x').^2 + R^3;
% Ks = 2*abs(pdist2(X, x').^3) - 3*R*pdist2(X, x').^2 + R^3;

ys_std = Ks*( (K + 1e-3*eye(size(x,2)))\y' );
cov_std = Kss - Ks/(K + 1e-3*eye(size(x,2)))*Ks';
elapsed_classic = toc;
fprintf("Classic GP took %fs, RMSE = %.4e\n", elapsed_classic, sqrt( sum((ys_std - Y').^2)/length(ys_std) ));
% fprintf("Classic GP took %fs, RMSE = %f\n", toc, sqrt( sum((ys_std - Y').^2)/length(ys_std) ));
% toc

%% FAGP
% Define the phi functions
% phi accepts row vector n and column vector x
%     it returns a length(x)-by-length(n) matrix

l = 0.5;% 0.039; % Scale factor
% l = 1;
alpha = 0.5;% 0.009;%sqrt(2); % Global scale factor
epsilon = 1/(sqrt(2)*l); % Parameter depending on scale factor
R = 1;
ii = 0;

% select_eigv = [ 2, 5, 10, 
for eigv = n%:3:15
    ii = ii + 1;
    tic
    [ K_tilde, cov_ys, K_app, Ks_app, lambdas, indices, times(ii) ] = approximateKernel(x', X, eigv, epsilon, alpha);

    ys = K_tilde*y';
    elapsed(ii, 1) = toc;

    figure('Position', [300, 100, 1200, 400])
    subplot(1,3,1)
    hold on
    surface(X1, X2, reshape(ys, 50, 50) - max(ys), 'FaceColor','interp','EdgeColor','interp');
    % quiver(xs(1,:), xs(2,:), ys_grad(1,:), ys_grad(2,:),'color',[.2 .2 .2]);
    plot(x(1,y==1), x(2,y==1), '.','markersize',28,'color',[.8 0 0]); %Interior points
    plot(x(1,y==0), x(2,y==0), '.','markersize',28,'color',[.8 .4 0]); %Border points
    plot(x(1,y==-1), x(2,y==-1), '.','markersize',28,'color',[0 .6 0]); %Exterior points
    contour(X1, X2, reshape(ys, 50, 50), [0,0], 'linewidth',2,'color','white');
    title(['FAGP using ', num2str(eigv), ' eigenvalues'])
    axis equal
    subplot(1,3,2)
    hold on
    surface(X1, X2, reshape(diag(cov_ys), 50, 50) - 1*max(diag(cov_ys)), 'FaceColor','interp','EdgeColor','interp');
    % quiver(xs(1,:), xs(2,:), ys_grad(1,:), ys_grad(2,:),'color',[.2 .2 .2]);
    plot(x(1,y==1), x(2,y==1), '.','markersize',28,'color',[.8 0 0]); %Interior points
    plot(x(1,y==0), x(2,y==0), '.','markersize',28,'color',[.8 .4 0]); %Border points
    plot(x(1,y==-1), x(2,y==-1), '.','markersize',28,'color',[0 .6 0]); %Exterior points
    contour(X1, X2, reshape(ys, 50, 50), [0,0], 'linewidth',2,'color','white');
    title('Uncertainty')
    axis equal
    subplot(1,3,3)
    surf(reshape(indices(:,1), [eigv,eigv]), reshape(indices(:,2), [eigv,eigv]), reshape(lambdas, [eigv,eigv]))
    title(['Eigenvalues with alpha = ', num2str(alpha), ' and l = ', num2str(l)])
    view(70,20)
    drawnow
    
%     pause(2);
    
    % Comparison between the two methods
    RMSE = sqrt( sum((ys - Y').^2)/length(ys) );
    fprintf("RMSE with %d eigenvalues is %.4e. Lambda_hat is a %dx%d matrix. Run in %fs\n", eigv, RMSE, size(indices,1), size(indices,1), elapsed(ii));

end
% hermiteH_user(1,3)
% Plots

% figure
% hold on
% surface(X1, X2, reshape(ys, 50, 50) - max(ys), 'FaceColor','interp','EdgeColor','interp');
% plot(x(1,y==1), x(2,y==1), '.','markersize',28,'color',[.8 0 0]); %Interior points
% plot(x(1,y==0), x(2,y==0), '.','markersize',28,'color',[.8 .4 0]); %Border points
% plot(x(1,y==-1), x(2,y==-1), '.','markersize',28,'color',[0 .6 0]); %Exterior points
% contour(X1, X2, reshape(ys, 50, 50), [0,0], 'linewidth',2,'color',rand(1,3));
        
figure('Position', [800, 400, 1000, 400])
subplot(1,2,1)
hold on
surface(X1, X2, reshape(ys_std, 50, 50) - max(ys_std), 'FaceColor','interp','EdgeColor','interp');
plot(x(1,y==1), x(2,y==1), '.','markersize',28,'color',[.8 0 0]); %Interior points
plot(x(1,y==0), x(2,y==0), '.','markersize',28,'color',[.8 .4 0]); %Border points
plot(x(1,y==-1), x(2,y==-1), '.','markersize',28,'color',[0 .6 0]); %Exterior points
contour(X1, X2, reshape(ys_std, 50, 50), [0,0], 'linewidth',2,'color','white');
title('Classic GP formula');
subplot(1,2,2)
hold on
surface(X1, X2, reshape(diag(cov_std), 50, 50) - max(diag(cov_std)), 'FaceColor','interp','EdgeColor','interp');
plot(x(1,y==1), x(2,y==1), '.','markersize',28,'color',[.8 0 0]); %Interior points
plot(x(1,y==0), x(2,y==0), '.','markersize',28,'color',[.8 .4 0]); %Border points
plot(x(1,y==-1), x(2,y==-1), '.','markersize',28,'color',[0 .6 0]); %Exterior points
contour(X1, X2, reshape(ys_std, 50, 50), [0,0], 'linewidth',2,'color','white');

%% Approximated Gaussian Kernel

function [ K_tilde, covariance, K_approx, Ks_approx, lambda_comb, idx_comb, times ] ...
    = approximateKernel(x, xp, n, ep, alpha)

    % Combinations
    [ index1, index2 ] = ndgrid(1:n, 1:n);
    idx_comb = [ index1(:), index2(:) ];
    phi_comb = zeros(size(x,1), size(idx_comb,1));
    phip_comb = zeros(size(xp,1), size(idx_comb,1));
    lambda_comb = zeros(1,size(idx_comb,1));
%     Lambda_hat = zeros(n^2);
%     K1_tilde = zeros(size(xp,1), size(x,1));
%     K2_tilde = zeros(size(x,1));
%     K_approx = zeros(size(x,1));
%     Ks_approx = zeros(size(xp,1), size(x,1));
%     tic
%     for idx = 1:size(idx_comb,1)
        % phi_comb is phi_m(x1)*phi_p(x2), where x1, x2 the dims of x and
        % (m,p) all the grid combinations of n eigenvalues (n^dims in
        % total)
%         phi_comb(:,idx) = eigenFnct(x(:,1), idx_comb(idx,1), ep, alpha).*...
%             eigenFnct(x(:,2), idx_comb(idx,2), ep, alpha);
%         lambda_comb(idx) = eigenValue(idx_comb(idx,1), ep, alpha)*eigenValue(idx_comb(idx,2), ep, alpha);
%         
%         phip_comb(:,idx) = eigenFnct(xp(:,1), idx_comb(idx,1), ep, alpha).*...
%             eigenFnct(xp(:,2), idx_comb(idx,2), ep, alpha);
        
        % Plot idx-th eigenfunction
%         if idx == 10
%             figure
%             surf(phi_comb(:,idx)*phi_comb(:,idx)','FaceColor','interp','EdgeColor','interp')
%         end
        


% not used
%         K_approx = K_approx + lambda_comb(idx)*phi_comb(:,idx_comb(idx,1))* ...
%             phi_comb(:,idx_comb(idx,2))'; % WRONG! I ALREADY DID THE COMBINATIONS OF PHI, AND
        % THE SUMMATION IS OVER THE SAME INDEX bold n!!!
        
%         Ks_approx = Ks_approx + lambda_comb(idx)*phip_comb(:,idx_comb(idx,1))* ...
%             phi_comb(:,idx_comb(idx,2))';
        
%         inv_SigmaN = 1./1e-5*eye(size(x,1));
%         Lambda_hat = Lambda_hat + phi_comb(:,idx_comb(idx,1))'* ...
%             inv_SigmaN*phi_comb(:,idx_comb(idx,2));
%     end
%     toc

    phi_comb = eigenFnct(x(:,1), idx_comb(:,1)', ep, alpha).*...
            eigenFnct(x(:,2), idx_comb(:,2)', ep, alpha);
        
    phip_comb = eigenFnct(xp(:,1), idx_comb(:,1)', ep, alpha).*...
            eigenFnct(xp(:,2), idx_comb(:,2)', ep, alpha);
    lambda_comb = eigenValue(idx_comb(:,1), ep, alpha).*eigenValue(idx_comb(:,2), ep, alpha);
    
    noise_var = 1e-3;
    inv_SigmaN = 1./noise_var*eye(size(x,1));
    Lambda_hat = 1/noise_var * (phi_comb'*phi_comb) + diag(1./lambda_comb);
    
%     tmp_inv = inv(Lambda_hat);
    tstart_inv = tic;
    tmp = (inv_SigmaN - 1/noise_var*phi_comb/Lambda_hat*phi_comb'* 1/noise_var);
    times.elapsed_inv = toc(tstart_inv);
    tstart_mult = tic;
    K_tilde = phip_comb*diag(lambda_comb)*phi_comb'* tmp;
    times.elapsed_mult = toc(tstart_mult);
    covariance = phip_comb*diag(lambda_comb)*phip_comb' - K_tilde*phi_comb*diag(lambda_comb)*phip_comb';
    
%     for idx = 1:size(idx_comb,1)
%         K1_tilde = K1_tilde + phip_comb(:,idx_comb(idx,1))* ...
%             lambda_comb(idx)*phi_comb(:,idx_comb(idx,2))';
% %         K2_tilde = K2_tilde + phi_comb(:,idx_comb(idx,1))/ ...
% %             Lambda_hat(idx,idx)*phi_comb(:,idx_comb(idx,2))';
%     end
%     K2_tilde = K2_tilde + phi_comb(:,idx_comb(idx,1))/ ...
%     Lambda_hat(idx,idx)*phi_comb(:,idx_comb(idx,2))';
%     K_tilde = K1_tilde*(inv_SigmaN - inv_SigmaN*K2_tilde*inv_SigmaN);
    
    K_approx = phi_comb*diag(lambda_comb)*phi_comb';
    Ks_approx = phip_comb*diag(lambda_comb)*phi_comb';
%     for ii = 1:size(idx_comb,1)
%         K_approx = K_approx + lambda_comb(ii)*phi_comb(:,ii)*phi_comb(:,ii)';
%         Ks_approx = Ks_approx + lambda_comb(ii)*phip_comb(:,ii)*phi_comb(:,ii)';
%     end
end

function phi = eigenFnct(x_1D, n_eigv, ep, alpha)

    % Compute the eigenfunction phi corresponding to the eigenvalue n_eigv
    % using one of the dimensions of x "x_1D".
    beta = (1 + (2*ep/alpha)^2)^0.25;
    Gamma = sqrt(beta./(2.^(n_eigv-1).*gamma(n_eigv)));
    delta2 = alpha^2/2*(beta^2 - 1);
    
    phi = exp(-delta2*x_1D.^2)*Gamma.*hermiteH_user(n_eigv-1, alpha*beta*x_1D);
%     HH = zeros(length(x_1D), length(n_eigv));
%     for idx = 1:length(n_eigv)
%         HH(:,idx) = hermite(n_eigv(idx)-1, alpha*beta*x_1D); % Using recursive formulation for computing hermite polyn
%     end
%     phi = exp(-delta2*x_1D.^2)*Gamma.*HH;
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
