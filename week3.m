% Statistical Parameter Estimation week 3
close all
clc
clearvars

% Teht. 1
lambda = 3;
sigma = 0.5;
x0 = sigma*randn();
h = 0.1;
t = 0:h:20;
X = zeros(1,length(t));

% one realisation of OU
for i = 2:length(t)
    X(i) = X(i-1) - lambda*X(i-1)*h + sigma*sqrt(h)*randn();
end

figure()
plot(t,X)
title("OU with \sigma=0.5 and \lambda=3")
xlabel("t")

%% MCMC for estimation


n = length(t);
lambda_h = @(lambda) 1 - lambda*h;
L = @(lambda) diag(ones(1,n)) - diag( ones(1,n-1)*lambda_h(lambda), -1 );
log_post = @(lambda) -n/2*log(2*pi) - n/2*log( sigma^2*h ) - 1/2*X*1/(sigma^2*h)*L(lambda)'*L(lambda)*X';

fmin_log_post = @(lambda) -log_post(lambda);
init_val = fminsearch(fmin_log_post, 0)

% DRAM
Nsim = 50000;

thetas = zeros(1, Nsim);
thetas(1) = init_val;
c = 1;
for i = 1:Nsim
    if(i>10)
        c = cov(thetas(1:i)) + 1e-06;
    end
    % proposal distribution N(theta(i-1),c)
    theta_new = sqrt(c)*randn() + thetas(i);
    alpha = exp( log_post(theta_new) - log_post(thetas(i)) );
    u = rand();
    if(u <= alpha)
        thetas(i+1) = theta_new;
    else
        % Delayed rejection part
        c_2 = c/2;
        theta_new2 = sqrt(c_2)*randn() + thetas(i);
        alpha = exp( log_post(theta_new2) - log_post(thetas(i)) );
        u = rand();
        if(u <= alpha)
            thetas(i+1) = theta_new2;
        else
            thetas(i+1) = thetas(:,i);
        end
    end   
end

figure()
plot(thetas)
yline(mean(thetas), "r")
title("Chain for \lambda")
ylabel("\lambda")

figure()
hist(thetas, 100)
xline(mean(thetas), "r", LineWidth=3)
xlabel("\lambda")
title("Histogram of \lambda chain")

figure()
autocorr(thetas, "NumLags", 50)


mean(thetas)



%% MCMC with priors
close all

% Gaussian prior N(0,sigma_prior^2)

n = length(t);
prior = @(lambda) pdf("Normal", lambda, 0, 1);
lambda_h = @(lambda) 1 - lambda*h;
L = @(lambda) diag(ones(1,n)) - diag( ones(1,n-1)*lambda_h(lambda), -1 );
log_post = @(lambda) log(prior(lambda)) - n/2*log(2*pi) - n/2*log( sigma^2*h ) - 1/2*X*1/(sigma^2*h)*L(lambda)'*L(lambda)*X';

fmin_log_post = @(lambda) -log_post(lambda);
init_val = fminsearch(fmin_log_post, 0)

% DRAM
Nsim = 50000;

thetas = zeros(1, Nsim);
thetas(1) = init_val;
c = 1;
for i = 1:Nsim
    if(i>10)
        c = cov(thetas(1:i)) + 1e-06;
    end
    % proposal distribution N(theta(i-1),c)
    theta_new = sqrt(c)*randn() + thetas(i);
    alpha = exp( log_post(theta_new) - log_post(thetas(i)) );
    u = rand();
    if(u <= alpha)
        thetas(i+1) = theta_new;
    else
        % Delayed rejection part
        c_2 = c/2;
        theta_new2 = sqrt(c_2)*randn() + thetas(i);
        alpha = exp( log_post(theta_new2) - log_post(thetas(i)) );
        u = rand();
        if(u <= alpha)
            thetas(i+1) = theta_new2;
        else
            thetas(i+1) = thetas(:,i);
        end
    end   
end

figure()
plot(thetas)
yline(mean(thetas), "r")
title("Chain for \lambda")
ylabel("\lambda")

figure()
hist(thetas, 100)
xline(mean(thetas), "r", LineWidth=3)
xlabel("\lambda")
title("Histogram of \lambda chain")

figure()
autocorr(thetas, "NumLags", 50)


mean(thetas)


%% Logarithmic transformation prior


n = length(t);
prior = @(lambda) pdf("Normal", log(lambda), 0, 1);
lambda_h = @(lambda) 1 - lambda*h;
L = @(lambda) diag(ones(1,n)) - diag( ones(1,n-1)*lambda_h(lambda), -1 );
log_post = @(lambda) log(prior(lambda)) - n/2*log(2*pi) - n/2*log( sigma^2*h ) - 1/2*X*1/(sigma^2*h)*L(lambda)'*L(lambda)*X';

fmin_log_post = @(lambda) -log_post(lambda);
init_val = fminsearch(fmin_log_post, 0)

% DRAM
Nsim = 50000;

thetas = zeros(1, Nsim);
thetas(1) = init_val;
c = 1;
for i = 1:Nsim
    if(i>10)
        c = cov(thetas(1:i)) + 1e-06;
    end
    % proposal distribution N(theta(i-1),c)
    theta_new = sqrt(c)*randn() + thetas(i);
    alpha = exp( log_post(theta_new) - log_post(thetas(i)) );
    u = rand();
    if(u <= alpha)
        thetas(i+1) = theta_new;
    else
        % Delayed rejection part
        c_2 = c/2;
        theta_new2 = sqrt(c_2)*randn() + thetas(i);
        alpha = exp( log_post(theta_new2) - log_post(thetas(i)) );
        u = rand();
        if(u <= alpha)
            thetas(i+1) = theta_new2;
        else
            thetas(i+1) = thetas(:,i);
        end
    end   
end

figure()
plot(thetas)
yline(mean(thetas), "r")
title("Chain for \lambda")
ylabel("\lambda")

figure()
hist(thetas, 100)
xline(mean(thetas), "r", LineWidth=3)
xlabel("\lambda")
title("Histogram of \lambda chain")

figure()
autocorr(thetas, "NumLags", 50)


mean(thetas)


%% Uniform prior


n = length(t);
prior = @(lambda) 1/10*(lambda >= -5)*(lambda <= 5);
lambda_h = @(lambda) 1 - lambda*h;
L = @(lambda) diag(ones(1,n)) - diag( ones(1,n-1)*lambda_h(lambda), -1 );
log_post = @(lambda) log(prior(lambda)) - n/2*log(2*pi) - n/2*log( sigma^2*h ) - 1/2*X*1/(sigma^2*h)*L(lambda)'*L(lambda)*X';

fmin_log_post = @(lambda) -log_post(lambda);
init_val = fminsearch(fmin_log_post, 0)

% DRAM
Nsim = 50000;

thetas = zeros(1, Nsim);
thetas(1) = init_val;
c = 1;
for i = 1:Nsim
    if(i>10)
        c = cov(thetas(1:i)) + 1e-06;
    end
    % proposal distribution N(theta(i-1),c)
    theta_new = sqrt(c)*randn() + thetas(i);
    alpha = exp( log_post(theta_new) - log_post(thetas(i)) );
    u = rand();
    if(u <= alpha)
        thetas(i+1) = theta_new;
    else
        % Delayed rejection part
        c_2 = c/2;
        theta_new2 = sqrt(c_2)*randn() + thetas(i);
        alpha = exp( log_post(theta_new2) - log_post(thetas(i)) );
        u = rand();
        if(u <= alpha)
            thetas(i+1) = theta_new2;
        else
            thetas(i+1) = thetas(:,i);
        end
    end   
end

figure()
plot(thetas)
yline(mean(thetas), "r")
title("Chain for \lambda")
ylabel("\lambda")

figure()
hist(thetas, 100)
xline(mean(thetas), "r", LineWidth=3)
xlabel("\lambda")
title("Histogram of \lambda chain")

figure()
autocorr(thetas, "NumLags", 50)


mean(thetas)


%% Teht. 3
close all

% using Gaussian prior
lambda = 3;
n = length(t);
prior = @(sigma) pdf("Normal", sigma, 0, 1);
lambda_h = 1 - lambda*h;
L = diag(ones(1,n)) - diag( ones(1,n-1)*lambda_h, -1 );
log_post = @(sigma) log(prior(sigma)) - n/2*log(2*pi) - n/2*log( sigma^2*h ) - 1/2*X*1/(sigma^2*h)*L'*L*X';

fmin_log_post = @(sigma) -log_post(sigma);
init_val = fminsearch(fmin_log_post, 0)

% DRAM
Nsim = 50000;

thetas = zeros(1, Nsim);
thetas(1) = init_val;
c = 1;
for i = 1:Nsim
    if(i>10)
        c = cov(thetas(1:i)) + 1e-06;
    end
    % proposal distribution N(theta(i-1),c)
    theta_new = sqrt(c)*randn() + thetas(i);
    alpha = exp( log_post(theta_new) - log_post(thetas(i)) );
    u = rand();
    if(u <= alpha)
        thetas(i+1) = theta_new;
    else
        % Delayed rejection part
        c_2 = c/2;
        theta_new2 = sqrt(c_2)*randn() + thetas(i);
        alpha = exp( log_post(theta_new2) - log_post(thetas(i)) );
        u = rand();
        if(u <= alpha)
            thetas(i+1) = theta_new2;
        else
            thetas(i+1) = thetas(:,i);
        end
    end   
end

figure()
plot(thetas)
yline(mean(thetas), "r")
title("Chain for \sigma")
ylabel("\sigma")

figure()
hist(thetas, 100)
xline(mean(thetas), "r", LineWidth=3)
xlabel("\sigma")
title("Histogram of \sigma chain")

figure()
autocorr(thetas, "NumLags", 50)


mean(thetas)

%% estimating lambda and sigma
close all

% using Gaussian prior, theta = (lambda,sigma)
n = length(t);
prior = @(theta) mvnpdf(theta', [0 0], [5 1]);
lambda_h = @(lambda) 1 - lambda*h;
L = @(lambda) diag(ones(1,n)) - diag( ones(1,n-1)*lambda_h(lambda), -1 );
log_post = @(theta) log(prior(theta)) - n/2*log(2*pi) - n/2*log( theta(2)^2*h ) - 1/2*X*1/(theta(2)^2*h)*L(theta(1))'*L(theta(1))*X';

fmin_log_post = @(theta) -log_post(theta);
init_val = fminsearch(fmin_log_post, [0 0]')

% DRAM
Nsim = 70000;

thetas = zeros(2, Nsim);
thetas(:,1) = init_val';
R = eye(2);
for i = 1:Nsim
    if(i>10)
        c = cov(thetas(1:i)) + 1e-06*eye(2);
        R = chol(c);
    end
    % proposal distribution N(theta(i-1),c)
    theta_new = R*randn(2,1) + thetas(:,i);
    alpha = exp( log_post(theta_new) - log_post(thetas(:,i)) );
    u = rand();
    if(u <= alpha)
        thetas(:,i+1) = theta_new;
    else
        % Delayed rejection part
        c_2 = c./2 + 1e-06*eye(2);
        R2 = chol(c_2);
        theta_new2 = R2*randn(2,1) + thetas(:,i);
        alpha = exp( log_post(theta_new2) - log_post(thetas(:,i)) );
        u = rand();
        if(u <= alpha)
            thetas(:,i+1) = theta_new2;
        else
            thetas(:,i+1) = thetas(:,i);
        end
    end   
end

figure()
subplot(1,2,1)
plot(thetas(1,:))
yline(mean(thetas(1,:)), "r")
title("Chain for \lambda")
ylabel("\lambda")

subplot(1,2,2)
plot(thetas(2,:))
yline(mean(thetas(2,:)), "r")
title("Chain for \sigma")
ylabel("\sigma")

figure()
subplot(1,2,1)
hist(thetas(1,:), 100)
xline(mean(thetas(1,:)), "r", LineWidth=3)
xlabel("\lambda")
title("Histogram of \lambda chain")

subplot(1,2,2)
hist(thetas(2,:), 100)
xline(mean(thetas(2,:)), "r", LineWidth=3)
xlabel("\sigma")
title("Histogram of \sigma chain")

mean(thetas, 2)

% joint density
posterior = @(theta) 1/sqrt( (2*pi)^n*(theta(2)^2*h)^n )*exp(-1/2*X*1/(theta(2)^2*h)*L(theta(1))'*L(theta(1))*X')
[lambdas, sigmas] = meshgrid(0:.1:5, 0.1:.1:1);
sz_lambdas = size(lambdas);
densities = zeros(sz_lambdas);
for i = 1:sz_lambdas(1)
    for j = 1:sz_lambdas(2)
        densities(i,j) = posterior([lambdas(i,j) sigmas(i,j)]);
    end
end

figure()
mesh(lambdas, sigmas, densities)
xlabel("\lambda")
ylabel("\sigma")
zlabel("density")



