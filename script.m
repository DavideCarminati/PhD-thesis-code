clear
clc
close all
set(0,'defaultTextInterpreter','latex');
set(0, 'defaultAxesTickLabelInterpreter','latex');
set(0, 'defaultLegendInterpreter','latex');

TRAIN_POINTS = 100;
x = linspace(-5, 5, TRAIN_POINTS)'; 
y_real = @(x) x .* sin(2 * x);
% y_real = @(x) sin(2 * x);
% y_real = @(x) 0.8 + ( x + 0.2 ) .* ( 1 - 5 ./ (1 + exp(-2 * x)));
% y_real = @(x) sin(x.^2 / 2);
% y_real = @(x) sin(x .* sqrt(abs(x)));
y = y_real(x) + randn(TRAIN_POINTS,1) * 1e-1;
x_star = linspace(-5, 5, 200)';
%normalizing
% y = (y - min(y)) / (max(y) - min(y));
y_star = y_real(x_star);
% y_star = (y_star - min(y_star)) / (max(y_star) - min(y_star));

figure
plot(x_star, y_star)
hold on, grid on;
plot(x, y, 'o')
legend('True function', 'Training points', 'Location', 'best')
xlabel('x')
ylabel('y')

for l0 = 2
    hyp0 = [ l0; 1; 1 ];
%     GP_NE(x, y, x_star, y_star, hyp0, false, false);
    GP_NE_thesis(x, y, x_star, y_star, hyp0, true, false);
%     GP_NE_old(x, y, x_star, y_star, hyp0, true, false);
%     GP_NE_QN(x, y, x_star, y_star, hyp0, true, true);
end

hyp0 = [ 4; 1; 1e-3 ];
% GP_NE(x, y, x_star, y_star, hyp0, false, false);