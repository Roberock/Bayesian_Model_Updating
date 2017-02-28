function logL = Likelihood_Ferson_Challenge(D, theta)
% Calculation of the log_likelihood for the example in problemA.m
%
% USAGE:
% logL = problemA_log_p_D_theta(D, theta)
%
% INPUTS:
% D     = experimental observations   nobs x dim_x
% theta = epistemic parameters        npar x dim_theta
%
% OUTPUTS:
% logL(i)  = loglikelihood for the set of parameters theta(i,:) and the
%            data D, i = 1, 2, ...npar.        logL = npar x 1

%--------------------------------------------------------------------------
% who                    when         observations
%--------------------------------------------------------------------------
% Diego Andres Alvarez   Jul-24-2013  First algorithm
% Rocchetta Roberto      Gen-12-2016  Modified for crack detection
%--------------------------------------------------------------------------
% Diego Andres Alvarez - daalvarez@unal.edu.co

%%
npar = size(theta,1);  % number of thetas to evaluate
logL = zeros(npar,1);
for i = 1:npar
    logL(i) = sum(log(p_x_theta_pdf(D, theta(i,:))));
    %logL(i) = sum((p_x_theta_pdf(D, theta(i,:),net)));
    if isinf(logL(i))
        logL(i) = -1e10;
    end
end

return;

%%
function p = p_x_theta_pdf(x, theta_i)


Mu=theta_i(1);
Sig=theta_i(2);
Vu=theta_i(3);
Omega=theta_i(4);
A=theta_i(5);
B=theta_i(6);
% check the normal distribution parameter,if lbound greater than ubound, fix the likelihoood to infinity and evaluate next theta
if A>B 
    p=Inf;
    return
end

Ns=2000; %MC samples for the probabilistic model to be tested
X = normrnd(Mu,Sig,[Ns,1]);
Y = betarnd(Vu,Omega,[Ns,1]);
Z = unifrnd(A,B,[Ns,1]);
W_model=X.*Y./Z;
%% Estimate the PDF p_x_theta_pdf(x | theta)
%Type 1)  f = ksdensity(x,xi) specifies the vector xi of values, where the density 
% estimate of the data in x is to be evaluated
  p = ksdensity(W_model, x);  % p(i) = p_x_theta_pdf(x(i,:) | theta)

%% Type 1) compute 2-sided Kolmogorov-Smirnov test values for each of the samples
% binEdges    =  [-inf ; sort([x;W_model]) ; inf];
% binCounts1  =  histc (x , binEdges, 1);
% binCounts2  =  histc (W_model , binEdges, 1);
% sampleCDF1  =  cumsum(binCounts1)./sum(binCounts1);
% sampleCDF2  =  cumsum(binCounts2)./sum(binCounts2);
% 
% ks = abs(sampleCDF1(ismember(binEdges,x))-sampleCDF2(ismember(binEdges,x)))';
% assuming that the ks value are zero mean normally distributed, evaluate
% the value of the normal pdf for these ks.
% p = normpdf(ks);    
 
return;
