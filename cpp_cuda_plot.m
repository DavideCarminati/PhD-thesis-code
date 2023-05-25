%% FAGP with cuda
% The script reads and plots the result of FAGP saved in csv format.

load("cpp_cuda_results/X1_train.csv");
load("cpp_cuda_results/X2_train.csv");
load("cpp_cuda_results/y_train.csv");

load("cpp_cuda_results/X1_test.csv");
load("cpp_cuda_results/X2_test.csv");
load("cpp_cuda_results/y_test.csv");

figure
subplot(1, 2, 1)
surf(X1_test, X2_test, reshape(y_test, size(X1_test, 1), size(X1_test, 1)));
grid on;
subplot(1, 2, 2)
surf(X1_train, X2_train, reshape(y_train, size(X1_train, 1), size(X1_train, 1)));
grid on;