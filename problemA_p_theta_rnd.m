function p = problemA_p_theta_rnd(lb, ub, N)
% Sampling from the prior PDF for the example in problemA.m
%
% USAGE:
% p = problemA_p_theta_rnd(lb, ub, N)
%
% INPUTS:
% lb, ub = lower and upper bounds of the uniform PDF   1 x dim_theta
% N      = number of samples to generate
%
% OUTPUTS:
% p      = samples                                     N x dim_theta
%
% EXAMPLE:
%{
p = problemA_p_theta_rnd([1 2 3 4 5], [2 3 4 5 6], 10)
%}

%--------------------------------------------------------------------------
% who                    when         observations
%--------------------------------------------------------------------------
% Diego Andres Alvarez   Jul-24-2013  First algorithm
%--------------------------------------------------------------------------
% Diego Andres Alvarez - daalvarez@unal.edu.co

% Here an uniform non informative prior is employed

dim_theta = length(lb);

p = zeros(N, dim_theta);
for i = 1:dim_theta
   p(:,i) = unifrnd(lb(i), ub(i), N, 1);
end

return;
