function p = problemA_p_theta_pdf(theta, lb, ub)
% Definition of the prior PDF for the example in problemA.m
%
% USAGE:
% p = problemA_p_theta_pdf(theta, lb, ub)
%
% INPUTS:
% theta  = samples                                     N x dim_theta
% lb, ub = lower and upper bounds of the uniform PDF   1 x dim_theta
%
% OUTPUTS:
% p      = p_theta(theta)                              N x 1
%
% EXAMPLE:
%{
rnd = randn(100,3);
p = problemA_p_theta_pdf(rnd, [-1 -1 -1], [1 1 1]);
%}

%--------------------------------------------------------------------------
% who                    when         observations
%--------------------------------------------------------------------------
% Diego Andres Alvarez   Jul-24-2013  First algorithm
%--------------------------------------------------------------------------
% Diego Andres Alvarez - daalvarez@unal.edu.co

% Here an uniform non informative prior is employed

[n, dim_theta] = size(theta);

marginal_PDF = zeros(n, dim_theta);
for i = 1:dim_theta
   marginal_PDF(:,i) = unifpdf(theta(:,i), lb(i), ub(i));
end

p = prod(marginal_PDF,2);
   
return;
