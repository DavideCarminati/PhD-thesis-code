function [ sol, feval, output_struct ] = SR1Conditioning(f_handle, x0, VERBOSE)
    % Symmetric Rank-1 Trust-region solver

    iter = 0;
    grad_err = 1;
    N = 200;
    sol_change = 1;
    B = eye(length(x0)) * 1e1; % Initial approx Hessian
    max_radius = 100; % Max trust region radius
    radius = 1; % Initial trust region radius
    grad_err_tol = 1e-3;
    sol = x0;
%     VERBOSE = false;

    
    disp('Quasi-Newton optimizer');

    exit_msg = 'Solution reached. First order optimality lower than tolerance';
    
    grad_err_vect_QN = zeros(N, 1);
    tic
    while grad_err > grad_err_tol && iter < N && sol_change > 1e-60

        [ feval, dfeval ] = f_handle(sol);

%         if iter == 0
%             % Approximate initial Hessian as Matlab does
%             B = eye(length(x0)) * max(dfeval) / max(x0); 
%         end
        D = imchol(B);
        B = D' \ B / D;
        
        eps_tol = min(0.5, sqrt(norm(D' \ dfeval))) * norm(D' \ dfeval);
        z = zeros(length(x0),1);
        r = D' \ dfeval;
        d = -r;
        subproblem_iter = 0;
        
        while subproblem_iter < 5
            min_fun = inf;
            if d'*B*d <= 0
                fun = @(tau) dfeval' * (z + tau * d) + 0.5 * (z + tau * d)' * B * (z + tau * d);
                polyn = [ d' * d, 2 * d' * z, z' * z - radius^2 ];
%                 tau_sol = real(roots(polyn));
                tau_sol = real( (-polyn(2) + sqrt(polyn(2)^2 - 4 * polyn(1) * polyn(3))) / (2 * polyn(1)) );
                for ii = 1:length(tau_sol)
                    if fun(tau_sol(ii)) < min_fun
                        tau_min = tau_sol(ii);
                        min_fun = fun(tau_sol(ii));
                    end
                end
                s_opt = z + tau_min * d;
                subp_status = ['Negative curvature {', num2str(subproblem_iter), '}'];
                break;
            end
            alpha = r' * r / (d' * B * d);
            z = z + alpha * d;
            if norm(z) > radius
    %             fun = @(tau) dfeval' * (z + tau * d) + 0.5 * (z + tau * d)' * B * (z + tau * d);
                polyn = [ d' * d, 2 * d' * z, z' * z - radius^2 ];
%                 tau_sol = real(roots(polyn));
                tau_sol = real( (-polyn(2) + sqrt(polyn(2)^2 - 4 * polyn(1) * polyn(3))) / (2 * polyn(1)) );
%                 for ii = 1:length(tau_sol)
%                     if 1% tau_sol(ii) >= 0
                        s_opt = z + max(tau_sol) * d;
%                         break;
%                     end
%                 end
                subp_status = ['z beyond boundary {', num2str(subproblem_iter), '}'];
                break;
            end
            r_new = r + alpha * B * d;
            
%             % Preconditioning
%             y_cond = ConditioningCalculation(M,r_new);

            if norm(r_new) < eps_tol
                s_opt = z;
                subp_status = ['Degenerated in Newton method {', num2str(subproblem_iter), '}'];
                break;
            end
            beta = r_new' * r_new / (r' * r);
            d = -r_new + beta * d;
            r = r_new;
            subproblem_iter = subproblem_iter + 1;
        end
        
        sol_tmp = sol + s_opt; % Candidate next solution

        [ feval_tmp, dfeval_tmp ] = f_handle(sol_tmp);
        
        yk = dfeval_tmp - dfeval;
        
        actual_red = feval - feval_tmp;
        predicted_red = -(dfeval' * s_opt + 0.5 * s_opt' * B * s_opt);
        
        rho = actual_red / predicted_red;
        update = false;
        if rho > 1e-6
            sol = sol_tmp;
            grad_err = norm(dfeval_tmp, 'inf');
            feval = feval_tmp;
            update = true;
        else
    %         continue; % Do not update lambda since the actual cost fun reduction is smaller than thought
    %         if subproblem_iter == 5
    %             lambda_old_qn = lambda_old_qn - 1e-6 * dfeval;
    %         end
        end
        
        if rho < 0.15
            radius = 0.5 * radius;
        else
            if rho > 3/4 && (norm(s_opt) - radius) <= 1e-6
                radius = min(1.5 * radius, max_radius);
            end
        end
        
        if radius < 1e-6 && update == false
            exit_msg = 'Trust region radius lower than tolerance. Stopped.';
            break;
        end
        
        % Updating the approximate Hessian
        cond_on_B = abs(s_opt' * (yk - B * s_opt)) >= 1e-8 * norm(s_opt) * norm(yk - B * s_opt);
        if cond_on_B == true
            B = B + (yk - B * s_opt) * (yk - B * s_opt)' / ( (yk - B * s_opt)' * s_opt);
    %     else
    %         continue; % do not update B
        end
        
        grad_err_vect_QN(iter+1) = grad_err ;
%         fun_value = NLL( lambda_old_qn(1:2), lambda_old_qn(2), lambda_old_qn(3));
%         sol_new = sol;
    %     sol_change = norm(sol_new - sol_old);

        iter = iter + 1;
        
        if VERBOSE == true
            disp("[" + num2str(iter) + "] argmin = " + num2str(sol, '%.10f') + " (updated: " + update + ") with err: " + num2str(grad_err)+ " (fun_value = " + ...
                num2str(feval) + ") -- status: " + subp_status);
        end
    end

    if iter == N
        exit_msg = 'Maximum number of iterations reached. Stopped.';
    end
    output_struct.gradients = grad_err_vect_QN;
    output_struct.iterations = iter;
    disp(exit_msg);

end

function y = ConditioningCalculation(M,r)
    y = M\r;
end

function D = imchol(B)
    % Computes preconditioning matrix D using Inexact Modified Cholesky
    I = eye(size(B,1));
    T = diag(vecnorm(B * I));
%     for i = 1:size(B,1)
%         T_bis(i,i) = norm(B * I(:,i));
%     end
    B_hat = sparse(T^-0.5 * B * T^-0.5);
    beta = norm(B);
    if min(B, [], 'all') > 0
        alpha = 0;
    else
        alpha = beta / 2;
    end
    k = 0;
    while k < 50
        try
            L = ichol(B_hat + sparse(alpha * I));
            D = L';
            return;
        catch
            alpha = max(2* alpha * k , beta/2);
            k = k + 1;
        end
    end
    D = eye(size(B,1));
    
end




