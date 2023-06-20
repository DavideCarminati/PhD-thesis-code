%% This script creates obstacles for GPIS
close all
clc
clear
num_circles = 3;
num_full_points = 3; % 3 red points per circle
circleRadius = [ 0.6; 0.85; 0.5 ] * 4 / 5; 
circleCenter = [    1, 4, 2.25;
                    3, 3.5, 4.5 ]  * 4 / 5;
points = linspace(-pi,pi,25);%-pi:pi/24:(pi-1e-3);
% x_edge = circleCenter + (circleRadius + 0.1*sin(points*10) + 0.11*cos(points*20 + 12) ...
%     + 1e-5*randn(1,length(points))) .* [cos(points); sin(points)];

x_edge = [];
y_edge = [];
x_full = [];
y_full = [];
for idx = 1:num_circles
    x_edge = [ x_edge, circleCenter(:,idx) + circleRadius(idx) * [cos(points); sin(points)] ];
    y_edge = [ y_edge, zeros(1, length(points)) ];
    x_full = [ x_full, 0.1*circleRadius(idx)*rand(2, num_full_points) + circleCenter(:, idx) ];
    y_full = [ y_full, ones(1, num_full_points) ];
end

num_free_points = 40;
x1_free = zeros(1,num_free_points);
x2_free = zeros(1,num_free_points);
y_free = zeros(1,num_free_points);
jj = 0;
while jj < num_free_points
%     rng(seed);
    jj = jj + 1;
    x1_free(jj) = 0.05 + 3.95*rand(1);
    x2_free(jj) = 0.05 + 3.95*rand(1);
    y_free(jj) = -1;
    % Checking if some of the free points are inside obstacles
    for idx = 1:size(circleCenter,2)
        dist_from_ctr = pdist2([x1_free(jj)', x2_free(jj)'], circleCenter(:,idx)');
        if dist_from_ctr <= circleRadius(idx) + 0.25
            x1_free(jj) = 0;
            x2_free(jj) = 0;
            y_free(jj) = 0;
            jj = jj - 1;
            break
        end
    end
end

x = [ x_full, x_edge, [ x1_free; x2_free ] ]';
y_noisefree = [ y_full, y_edge, y_free ]';
y = y_noisefree + randn(length(y_noisefree), 1) * 1e-3;

save("artificial_cloud_1.mat", "x", "y", "y_noisefree");

figure
hold on, grid on;
plot(x(y_noisefree==1,1), x(y_noisefree==1,2), '.','markersize',28,'color',[.8 0 0]); %Interior points
plot(x(y_noisefree==0,1), x(y_noisefree==0,2), '.','markersize',28,'color',[.8 .4 0]); %Border points
plot(x(y_noisefree==-1,1), x(y_noisefree==-1,2), '.','markersize',28,'color',[0 .6 0]); %Exterior points
