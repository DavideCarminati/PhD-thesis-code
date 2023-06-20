%% Recursive GP
set(0,'defaultTextInterpreter','latex');
set(0, 'defaultAxesTickLabelInterpreter','latex');
set(0, 'defaultLegendInterpreter','latex');
% clear
PLOTS = true;
x = linspace(-5, 7, 410);
% x = linspace(-5, 7, N_train);
y = sin(x*2).*x.^2;% + 1e-2*randn(1,length(x));
funct = @(x) sin(x*2).*x.^2;

% figure
% plot(x,y)
% grid on

% Creating indices selecting basis vector
num_points = 40; % Basis vector dimension
dist = round(size(y,2)/num_points);
% slct = 1:dist:size(y,2);
slct = round(linspace(1,size(y,2),num_points));

% Building batch vector for recursion
x_basisvector = x;
x_basisvector(slct) = []; % take out the values of the basis vector

% Create multiple batches
batch_dim = 10;
% batch_dim = 20;
X = zeros(batch_dim, length(x_basisvector)/batch_dim) - 5;
for jj = 1:length(x_basisvector)/batch_dim
    p = sort(randperm(length(x_basisvector), batch_dim));
    X(:,jj) = x_basisvector(p);
    x_basisvector(p) = [];
end
Y = sin(2*X).*X.^2 + 1e-6*randn(size(X));
basVect = x(slct)';

if PLOTS == true
    figure
    for jj = 1:size(X, 2)
        plot(X(:,jj), Y(:, jj), '.', 'Color', [0.9290 0.6940 0.1250], 'MarkerSize', 3);
        hold on;
        grid on;
    end
end

%% Classic GP

l = 1;
alpha = 3;

x_train = x;
x_train(slct) = [];
y_train = y;
y_train(slct) = [];
xs = x(slct); % Test points are the basis vector
K = kernelFnct(x_train', x_train', alpha, l) + 1e-6 * eye(length(x_train));
Ks = kernelFnct(xs', x_train', alpha, l);

tic
ypred = Ks/K*y_train';
elapsed_GP = toc;
fprintf('Elasped plain GP = %f\n', elapsed_GP);

if PLOTS == true
    figure
    title('Classic GP');
    plot(xs, ypred)
end

%%

% Initialize RGP
l = 0.25;
alpha = 3;
K = kernelFnct(basVect, basVect, alpha, l) + 1e-6 * eye(length(basVect));
mu_g_old = 5*rand(num_points,1); % Initial condition on ypred
Cg_old = 10e2*eye(num_points);
Cg_old = K;
% K_inv = inv(K);
% X = 0;
% Y = sin(X);

if PLOTS == true
    rgp_fig = figure();
    grid on;
end
% ann = annotation('textbox','String', "", 'FitBoxToText','on');
% subplot(2,1,1);
% hold on, grid on;
% % Plot real function
% for jj = 1:size(X, 2)
%     plot(X(:,jj), Y(:, jj), '.', 'Color', [0.9290 0.6940 0.1250]);
% end
tic
for ii = 1:size(X,2)
    % Inference
    Ks = kernelFnct(X(:,ii), basVect, alpha, l);
    Kss = kernelFnct(X(:,ii), X(:,ii), alpha, l);
    J = Ks/(K);
    mu_p = 0*X(:,ii) + J*(mu_g_old - 0*basVect);
    B = Kss - J*Ks';
    Cp = B + J*Cg_old*J';
%     Cp = 0.5 * (Cp + Cp');
%     Cp = Kss + J*(Cg_old - K)*J';
    % Update
    G = Cg_old*J'/(Cp + 0*ones(size(Cp))); % gain matrix, inversion of a matrix big as the batch used
    mu_g = mu_g_old + G*(Y(:,ii) - mu_p);
    Cg = Cg_old - G*J*Cg_old;
    mu_g_old = mu_g;
    Cg_old = Cg;
    
    rmse = sqrt(1/num_points * sum( (mu_g - funct(basVect)).^2 ));
    
    % Plotting results
    if PLOTS == true
    %     subplot(2,1,1)
%         for jj = 1:size(X, 2)
%             plot(X(:,jj), Y(:, jj), '.', 'Color', [0.9290 0.6940 0.1250], 'MarkerSize', 5);
%             hold on;
%         end
        figure(rgp_fig);
        plot(X(:), Y(:), '.', 'Color', [0.9290 0.6940 0.1250], 'MarkerSize', 5);
        hold on
        plot(basVect, mu_g, 'LineWidth', 3, 'Color', [0 0.4470 0.7410]);
    %     hold on
        scatter(basVect, mu_g, 30, [0.8500 0.3250 0.0980]);
    %     stdev_tmp = real(sqrt(diag(Cg)));
    %     stdev_nonzero = stdev>0;
        stdev(:,1) = mu_g + max(real(sqrt(diag(Cg))), 0)*2;
        stdev(:,2) = mu_g - max(real(sqrt(diag(Cg))), 0)*2;
        patch([basVect; flipud(basVect)], [stdev(:,1); flipud(stdev(:,2))], 'k', 'FaceAlpha', 0.15);
    %     plot(x, funct(x), '--')
    %     title('Mean', 'FontSize', 14, 'FontWeight', 'bold')

        ylim([-60, 60]);
        str = [". RMSE = " + num2str(rmse) ];
    %     ann.String = str;
        subtitle([ "Batches provided: " + num2str(ii) + str]);
        xlabel("x")
        ylabel("y")
        legend('Train points', 'RGP mean', 'Basis vectors', 'Confidence interval $95\%$', 'Location', 'NorthWest');
        grid on
        hold off

    %     subplot(2,1,2)
    %     plot(basVect, 2*max(real(sqrt(diag(Cg))), 0));
    %     title('Standard deviation', 'FontSize', 14, 'FontWeight', 'bold')
    %     ylim([-0.1, 1.5])
    %     xlabel("x")
    %     ylabel("$\sigma$")
    %     grid on
        drawnow
        pause();
    end
end
% subplot(2,1,1)
% hold on
% patch([basVect; flipud(basVect)], [stdev(:,1); flipud(stdev(:,2))], 'g');
elapsed_RGP = toc;
fprintf('Elapsed RGP = %f\n', elapsed_RGP);

if PLOTS == true
    % True function
    figure
    % subplot(2,1,1)
    hold on
    plot(x,y, '--k');
    % hold on
    % scatter(basVect, y(slct), 'g');
    % title('True fnct and basis vector')
    legend('Estimated', 'Basis vector', 'Real');
    hold off
end

%%
function K = kernelFnct(x1, x2, alpha, l)

    K = alpha*exp(-pdist2(x1, x2).^2/(2*l^2)); % Basis function
    
end