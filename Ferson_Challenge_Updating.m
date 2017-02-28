clc
clear variables
close all
%% Ferson Challenge
 % Given n samples from 3 ranodm W=(X.*Y)./Z and knowing that 
 % X~Gaussian(\Mu,\Sig); \Sig>=0
 % Y~Beta(\nu,\Omega);\nu>=0 \Omega>=0
 % Z~Uniform(\A,\B); 0<\A<=\B
 % Estimate  the parameters of the probabilistic model [ Mu  Sig  Vu  Omega  A  B ].
 % what is the uncertianty about the estimator?
 
%% 1) extract a random taget for the 'exact' real probabilistic modelparameters
    Mu=unifrnd(-50,50);
    Sig=unifrnd(1,50);
    Vu=unifrnd(1,50);
    Omega=unifrnd(1,50);
    A=unifrnd(1,30);
    B=unifrnd(10,50); 
    
    Targets=[Mu;Sig;Vu;Omega;A;B];
    
    Ns=10;
    X = normrnd(Mu,Sig,[1 Ns]);
    Y = betarnd(Vu,Omega,[1 Ns]);
    Z = unifrnd(A,B,[1 Ns]);
    W_10=X.*Y./Z;
    
    Ns=100; 
    X = normrnd(Mu,Sig,[1 Ns]);
    Y = betarnd(Vu,Omega,[1 Ns]);
    Z = unifrnd(A,B,[1 Ns]);
    W_100=X.*Y./Z; 
    
    Ns=250;
    X = normrnd(Mu,Sig,[1 Ns]);
    Y = betarnd(Vu,Omega,[1 Ns]);
    Z = unifrnd(A,B,[1 Ns]);
    W_250=X.*Y./Z;  
% Prealocate Memory
Ncases=3; % 3 different n=number of samples available
ThetA=cell(1,Ncases);
[MU_posterior,VAR_Theta,P90Theta,P10Theta]=deal(zeros(Ncases,6));
Computationa_Time=zeros(1,Ncases);
for k=1:Ncases
    if k==1 % update for n=10
        Data=W_10';
    elseif k==2 % update for n=100
        Data=W_100';
    elseif k==3 % update for n=250
        Data=W_250';
    end
    %display target 
    display([' The target parameter vector is  [Mu  Sig  Vu  Omega  A  B]  =  [' num2str(Targets') ']']) 
    display(['Number of samples n  =  ' num2str(length(Data))]) 
    %% TRANSITIONAL MCMC AND BAYESIAN UPDATING
    tic
    %% VARIABLE NAME  LOWER_BOUND  UPPER_BOUND
    variables = { ...
        '\mu'       -50        50          % interval information
        '\sigma'     1         50          % interval information
        '\nu'        1         50          % interval information
        '\omega'     1         50          % interval information
        'a'          1         30          % interval information
        'b'          10        50          % interval information
        };
    % Defining the prior PDF p(theta)
    lb = cell2mat(variables(:,2))';
    ub = cell2mat(variables(:,3))';
    p_theta    = @(x) problemA_p_theta_pdf(x, lb, ub);
    p_thetarnd = @(N) problemA_p_theta_rnd(lb, ub, N);
    % The loglikelihood of D given theta
    log_p_D_theta = @(theta) Likelihood_Ferson_Challenge(Data, theta);
    %% Bayesian estimation of theta: bayesian model updating using TMCMC
    Nsamples =30; % number of samples from prior;
    fprintf('Nsamples TMCMC = %d\n', Nsamples);
    [samples_ftheta_D] = problemA_tmcmc(log_p_D_theta, p_theta, p_thetarnd, Nsamples);
    Computationa_Time(k)=toc;
    display(['CPU Time for the Detection:    ' num2str(Computationa_Time(k)) ' seconds'])
    %% Save results and some statistic data
    ThetA{k}=samples_ftheta_D;
    MU_posterior(k,:)=mean(samples_ftheta_D);
    VAR_Theta(k,:)=var(samples_ftheta_D);
    P90Theta(k,:)=prctile(samples_ftheta_D,90);
    P10Theta(k,:)=prctile(samples_ftheta_D,10);
    %% Plot
    figure(k)
    for i=1:6
        subplot(3,2,i)
        hold on
       %  hist(samples_ftheta_D(:,i), ceil(sqrt(Nsamples)));
        %         line([ Target_updating(i) Target_updating(i)],[0 Ymax],'LineWidth',3)
        %         line([ mean(samples_ftheta_D(:,i)) mean(samples_ftheta_D(:,i))],[0 Ymax],'LineWidth',1)
        % KDensity plots
         ksdensity(samples_ftheta_D(:,i), 'support', [lb(i) ub(i)]);
         Ymax=max(ksdensity(samples_ftheta_D(:,i), 'support', [lb(i) ub(i)]));
        line([ Targets(i) Targets(i)],[0 Ymax],'LineWidth',5,'Color',[0.5 0.5 0.5],'LineStyle',':')
        line([ mean(samples_ftheta_D(:,i)) mean(samples_ftheta_D(:,i))],[0 Ymax],'LineWidth',3,'Color',[1 0 0])
        line([P90Theta(k,i) P90Theta(k,i)],[0 Ymax],'LineWidth',3,'Color',[0 1 0])
        line([P10Theta(k,i) P10Theta(k,i)],[0 Ymax],'LineWidth',3,'Color',[0 1 0]) 
        ylabel('PDF')
        xlabel(variables{i})
        box on
        grid on;
    end
    title(['Number of samples n  =  ' num2str(length(Data))])
    pause(0.01)
end