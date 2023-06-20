%% Script for measuring execution times of Recursive GP in 1D
clc
clear

N = [100, 500, 1000, 1700, 2500, 5000, 7500, 10000];

for count = 1:length(N)
    N_train = N(count);
    RGP_1D;
    runtime_GP(count) = elapsed_GP;
    runtime_RGP(count) = elapsed_RGP;
end

figure
grid on, hold on;
plot(N, runtime_GP);
plot(N, runtime_RGP);
legend('GP', 'RGP');
xlabel('N')
ylabel('Execution time [s]');