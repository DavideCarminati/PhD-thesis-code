% Monte-Carlo runs to assess mean execution time
close all
clear
clc
for jj = 1%:20

    FAGP_1D_thesis
    elapsed_cl(jj) = elapsed_classic;
    elapsed_fa(:, jj) = elapsed;
    close all
end

mean_time_classic = mean(elapsed_cl);
mean_time_fa = mean(elapsed_fa, 2);
fprintf("Mean execution time for classic GP with %d training samples: %.4e\n", TRAIN_POINTS, mean_time_classic);
fprintf("Mean execution time for FAGP for each set of eigenvalues:\n");
fprintf("%.4e\n", mean_time_fa);

for kk = 1:length(times)
    elapsed_inv_1D(kk) = times(kk).elapsed_inv;
    elapsed_mult_1D(kk) = times(kk).elapsed_mult;
end

%%
% close all
% clear
% clc
for jj = 1%:20

    FAGP_thesis
    elapsed_cl(jj) = elapsed_classic;
    elapsed_fa(:, jj) = elapsed;
    close all
end

mean_time_classic = mean(elapsed_cl);
mean_time_fa = mean(elapsed_fa, 2);
fprintf("Mean execution time for classic GP with %d training samples: %.4e\n", length(y), mean_time_classic);
fprintf("Mean execution time for FAGP for each set of eigenvalues:\n");
fprintf("%.4e\n", mean_time_fa);
%%
for kk = 1:length(times)
    elapsed_inv(kk) = times(kk).elapsed_inv;
    elapsed_mult(kk) = times(kk).elapsed_mult;
end

num_eig = 3:3:15;
plot(num_eig, elapsed_inv);
hold on
grid on
plot(num_eig, elapsed_inv_1D)
% plot(num_eig, elapsed_mult);
% plot(num_eig, elapsed_inv + elapsed_mult);
legend('Inversion time 2D', 'Inversion time 1D')

figure
hold on, grid on
plot(num_eig, elapsed_mult, 'b')
plot(num_eig, elapsed_mult_1D, 'r')
legend('Multiplication time 2D', 'Multiplication time 1D')
plot(num_eig, elapsed_mult, 'bo')
plot(num_eig, elapsed_mult_1D, 'ro')

%% Runs at different N

close all
clear
clc
N_vec = [10, 25, 50, 75, 100, 125, 150];
for jj = 1:7%:20

    N = N_vec(jj);
    FAGP_1D_thesis
    elapsed_cl(jj) = elapsed_classic;
    elapsed_fa(:, jj) = elapsed;
    elapsed_inv_1D(jj) = times.elapsed_inv;
    elapsed_mult_1D(jj) = times.elapsed_mult;
    close all
end

mean_time_classic = mean(elapsed_cl);
mean_time_fa = mean(elapsed_fa, 2);
fprintf("Mean execution time for classic GP with %d training samples: %.4e\n", TRAIN_POINTS, mean_time_classic);
fprintf("Mean execution time for FAGP for each set of eigenvalues:\n");
fprintf("%.4e\n", mean_time_fa);

%%
% close all
% clear
% clc
N_vec = [10, 25, 50, 75, 100, 125, 150];
for jj = 1:7%:20

    N = N_vec(jj);
    FAGP_thesis
    elapsed_cl(jj) = elapsed_classic;
    elapsed_fa(:, jj) = elapsed;
    elapsed_inv(jj) = times.elapsed_inv;
    elapsed_mult(jj) = times.elapsed_mult;
    close all
end

mean_time_classic = mean(elapsed_cl);
mean_time_fa = mean(elapsed_fa, 2);
fprintf("Mean execution time for classic GP with %d training samples: %.4e\n", length(y), mean_time_classic);
fprintf("Mean execution time for FAGP for each set of eigenvalues:\n");
fprintf("%.4e\n", mean_time_fa);
%%
% for kk = 1:length(times)
%     elapsed_inv(kk) = times(kk).elapsed_inv;
%     elapsed_mult(kk) = times(kk).elapsed_mult;
% end

num_eig = N_vec.^2;
plot(num_eig, elapsed_inv);
hold on
grid on
plot(num_eig, elapsed_inv_1D)
% plot(num_eig, elapsed_mult);
% plot(num_eig, elapsed_inv + elapsed_mult);
xlabel('$N$')
ylabel('Time [$s$]')
legend('Inversion time 2D', 'Inversion time 1D', 'Location', 'Best')

figure

plot(num_eig, elapsed_mult + elapsed_inv)
hold on, grid on
plot(num_eig, elapsed_mult_1D + elapsed_inv_1D)

plot(num_eig, elapsed_mult + elapsed_inv, 'o', 'Color', [0 0.4470 0.7410], 'HandleVisibility','off')
plot(num_eig, elapsed_mult_1D + elapsed_inv_1D, 'o', 'Color', [0.8500 0.3250 0.0980], 'HandleVisibility','off')
xlabel('$N$')
ylabel('Time [$s$]')
legend('Multiplication time 2D', 'Multiplication time 1D', 'Location', 'Best')
