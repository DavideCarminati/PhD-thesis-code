%% Script for computing the mean of the execution times of Recursive GPIS
%% OLD SCRIPT!!!
set(0,'defaultTextInterpreter','latex');
set(0, 'defaultAxesTickLabelInterpreter','latex');
set(0, 'defaultLegendInterpreter','latex');

exec_times = [  0.2702;
                0.9892;
                0.7322;
                1.0473;
                0.7294;
                0.0982; % 1st
                0.6165;
                0.0968;
                0.0801;
                1.1338;
                1.0138;
                0.1840;
                0.1540; % 1st
                0.1243;
                0.2802;
                0.2456;
                0.6969;
                0.1113;
                0.6833;
                0.9083;
                0.4168;
                0.3065;
                0.1072; % 1st
                0.6191;
                0.1631;
                0.5724;
                0.1680;
                0.1459;
                0.9258; % opt started
                0.6848; % opt stopped
                0.0970; % 1st 
                0.3319;
                0.0857;
                0.4294;
                0.0726;
                0.0801;
                0.2130;
                0.1869;
                0.6611;
                0.6094;
                0.1466;
                0.1198; % 1st
                0.1418;
                0.4160;
                0.1301;
                0.1794;
                0.0847;
                0.1419;
                0.5708;
                0.6903;
                0.5510;
                0.1051;
                0.3269;
                0.0748;
                0.0795
                ];
num_iter = 1:length(exec_times);
mean_exec = mean(exec_times);

figure
grid on, hold on;
% plot(num_iter, exec_times, '-o');
stem(num_iter, exec_times);
plot(num_iter, mean_exec * ones(1, length(exec_times)), '--g');
% str = {'Mean execution', 'time'};
% annotation("textarrow", [0.8 0.8], [ 0.8 mean_exec ], 'String', str);
xlim([1, 55]);
legend('RGPIS execution time', 'Mean execution time')
xlabel('Iteration')
ylabel('Time [s]')
