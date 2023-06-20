%% Compute the RMSE of a contour given GPIS edge points
function rmse = rmse_contour(contour_data, centers, radii)
    % Inputs:
    %       - contour_data is the 2xM matrix returned by Matlab contour
    %       function.
    %       - centers is a 2xD matrix containing the coordinates of the
    %       center of the real circles as [ x1; x2 ].
    %       - radii is a Dx1 vector containing the radius of the given
    %       circles.
    % Output:
    %       - rmse is the Root Mean Square Error for each obstacle.
    
    num_points_contour = 0;
    num_obstacles = 0;
    while ~isempty(contour_data)
        num_obstacles = num_obstacles + 1;
        num_points_contour = uint64(contour_data(2, 1));
        points_contour{num_obstacles} = contour_data(:, 2:num_points_contour+1);
        circle_points{num_obstacles} = size(points_contour{num_obstacles}, 2);
        contour_data(:, 1:num_points_contour+1) = [];
    end

    all_rmse = zeros(num_obstacles, length(radii));
    evalLineFcn = ...   % distance evaluation function
        @(model, points) sum((points(:, 2) - polyval(model, points(:,1))).^2,2);
    sampleSize = 2; % number of points to sample per trial
    maxDistance = 1; % max allowable distance for inliers

    for idx = 1:num_obstacles
        % Trying to fit the data to a circle and finding its center and
        % radius...
        c = [points_contour{idx}' ones(circle_points{idx},1)]\-(sum(points_contour{idx}.^2, 1))'; %least squares fit
        xhat = -c(1)/2;
        yhat = -c(2)/2;
%         rhat = sqrt((xhat^2+yhat^2) - c(3));
%         c = CircleFitByTaubin(points_contour{idx}');
%         c = fitellipse(points_contour{idx}(1,:)', points_contour{idx}(2,:)');
%         xhat = c(1);
%         yhat = c(2);
%         rhat = sqrt(c(3));
%         out = ellipseDataFilter_RANSAC(points_contour{idx}');
%         area_contour = polyarea(points_contour{idx}(1,:)', points_contour{idx}(2,:)');
        % Find mean radius for the current contour
        radius_hat = mean(pdist2([xhat, yhat], points_contour{idx}'));
        area_contour = pi * radius_hat^2;
        
        % Creating the circle from data for comparison with the real one
        points = linspace(-pi,pi,circle_points{idx});
%         data_circle = [ xhat; yhat ] + rhat * [cos(points); sin(points)];
        for ii = 1:length(radii)
%             inlierIdx = 0;
%             circle = @(x) (x(1,:) - centers(1,ii)).^2 + (x(2,:) - centers(2,ii)).^2 - radii(ii)^2;
            real_circle{ii} = centers(:,ii) + radii(ii) * [cos(points); sin(points)];
            area_real = pi * radii(ii)^2;
            r_max = max(pdist2(points_contour{idx}', centers(:,ii)'), [], 'all');
            r_min = min(pdist2(points_contour{idx}', centers(:,ii)'), [], 'all');
%             all_rmse(idx, ii) = sqrt(immse(real_circle{ii}, data_circle));
%             try
%                 [~, inlierIdx] = ransac(points_contour{idx},circle,evalLineFcn, ...
%                     sampleSize,maxDistance);
%             catch
%                 warning("No inliers found for contour #%d with circle #%d.", idx, ii);
%             figure
%             plot(points_contour{idx}(1,:)', points_contour{idx}(2,:)', 'o')
%             hold on
%             plot(centers(1,ii), centers(2,ii), 'o');
%             plot(corr(1,:));
%             plot(corr(2,:));
%             plot(corr(3,:));
%             [ ~, idx_max ] = max(corr(2,:));
%             data_circle_tmp = circshift(points_contour{idx}, -idx_max, 2);
%             corr2 = xcorr2(data_circle_tmp, real_circle{ii});
%             all_rmse(idx, ii) = sqrt(immse(real_circle{ii}, data_circle));
%             all_rmse(idx, ii) = area_contour / area_real / (1 + abs(r_max - r_min)/radii(ii));
            all_rmse(idx, ii) = radius_hat / radii(ii) / (1 + abs(r_max - r_min)/radii(ii));
%             all_rmse(idx, ii) = max(corr(2,:));
%             all_rmse(idx, ii) = sum(inlierIdx);
%             for jj = 1:length(points)
%                 data_circle_tmp = circshift(points_contour{idx}, jj-1, 2);
%                 all_rmse(idx, ii) = min(all_rmse(idx, ii), sqrt(immse(real_circle{ii}, data_circle_tmp)));
%             end
        end
%         figure
%         plot(real_circle{idx}(1,:), real_circle{idx}(2,:), '--');
%         hold on, grid on;
%         plot(data_circle(1,:), data_circle(2,:));
    end
    
%     rmse = max(all_rmse, [], 1);
    rmse = diag(all_rmse);

end

        