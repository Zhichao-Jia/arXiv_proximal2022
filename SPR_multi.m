global m n p rho rhohat epsilon flb glb domain A b xc


% initialize, m*n for dimension, p for SCAD RHS, domain [-x,x]

m = 120;
n = 120;
p = 320;
domain = 10;
epsilon = 0.01;
flb = 0;
glb = -p;


% set parameters

rho = 3;
rhohat = rho * 2;
tau = (rhohat - rho) * epsilon^2 / (4 * rhohat * (2 * rhohat - rho));


% initialize numbers of iterations

K = 1000
T = 10000


% set the lists

GRADFJ_list1 = [];
GRADKKT_list1 = [];
GRADFJ_list2 = [];
GRADKKT_list2 = [];
GRADFJ_list3 = [];
GRADKKT_list3 = [];


% run the algorithm

for trails = 1:50

    % generate data

    xstar = [zeros(n*3/4, 1); (5 * rand(n/4, 1) + 5) .* (2 * binornd(1, 0.5, n/4, 1) - 1)];
    xstar = xstar(randperm(n));
    A = normrnd(0, 1, m, n);
    noise = randn(m, 1);
    b = (A * xstar).^2 + noise;
    x0 = normrnd(0, 0.1, n, 1);
    while g(x0) > 0
        x0 = normrnd(0, 0.1, n, 1);
    end


    x_list = [x0];
    gradFJ_list = [];
    gradKKT_list = [];
    x = x0;

    for k = 1:K

        % algorithmic initialization for each outer loop

        xc = x;
        z = xc;
        ztotal = zeros(n, 1);
        s = 0;
        alphaf = 0;
        alphag = 0;


        % inner loops start

        for t = 1:T
            alpha = 2 / ((rhohat - rho) * (t + 1) + 36 * rhohat^2 / ((rhohat - rho) * t));
            if G(z) > tau
                z = proj(z - alpha * subgradG(z));
                alphag = alphag + alpha;
            else
                s = s + t;
                ztotal = ztotal + z * t;
                z = proj(z - alpha * subgradF(z));
                alphaf = alphaf + alpha;
            end
        end
        zfinal = ztotal / s;
        x = zfinal;


        % save the results to the list

        gradFJ_list(end+1) = rhohat * norm(x - x_list(:,end));
        gradKKT_list(end+1) = (1 + alphag / alphaf) * rhohat * norm(x - x_list(:,end));


        % save the results, and stop when reaching the max number of iterations

        if k == K / 100
            GRADFJ_list1(end+1) = gradFJ_list(end);
            GRADKKT_list1(end+1) = gradKKT_list(end);
        end

        if k == K / 10
            GRADFJ_list2(end+1) = gradFJ_list(end);
            GRADKKT_list2(end+1) = gradKKT_list(end);
        end

        if k == K
            break;
        end


        x_list(:,end+1) = x;

        % show some data during iterations

        if mod(k,K/10) == 0
            k / (K / 10)
        end
    end


    GRADFJ_list3(end+1) = gradFJ_list(end);
    GRADKKT_list3(end+1) = gradKKT_list(end);

    % show some data during iterations

    trails
end

% save('m1k1t1','GRAD_list');


% compute mean and variance

disp(['FJ Results:']);
disp(['K=' num2str(K/100) ', T=' num2str(T), ': median is ' num2str(median(GRADFJ_list1)) ', mean is ' num2str(mean(GRADFJ_list1)) ', variance is ' num2str(var(GRADFJ_list1))]);
disp(['K=' num2str(K/10) ', T=' num2str(T), ': median is ' num2str(median(GRADFJ_list2)) ', mean is ' num2str(mean(GRADFJ_list2)) ', variance is ' num2str(var(GRADFJ_list2))]);
disp(['K=' num2str(K) ', T=' num2str(T), ': median is ' num2str(median(GRADFJ_list3)) ', mean is ' num2str(mean(GRADFJ_list3)) ', variance is ' num2str(var(GRADFJ_list3))]);
disp(['KKT Results:']);
disp(['K=' num2str(K/100) ', T=' num2str(T), ': median is ' num2str(median(GRADKKT_list1)) ', mean is ' num2str(mean(GRADKKT_list1)) ', variance is ' num2str(var(GRADKKT_list1))]);
disp(['K=' num2str(K/10) ', T=' num2str(T), ': median is ' num2str(median(GRADKKT_list2)) ', mean is ' num2str(mean(GRADKKT_list2)) ', variance is ' num2str(var(GRADKKT_list2))]);
disp(['K=' num2str(K) ', T=' num2str(T), ': median is ' num2str(median(GRADKKT_list3)) ', mean is ' num2str(mean(GRADKKT_list3)) ', variance is ' num2str(var(GRADKKT_list3))]);

% save('result','GRADFJ_list1','GRADFJ_list2','GRADFJ_list3','GRADKKT_list1','GRADKKT_list2','GRADKKT_list3');


% define the functions

function v = f(x)
global A b m
v = norm((A * x).^2 - b, 1) / m;
end

function v = F(x)
global rhohat xc
v = f(x) + rhohat / 2 * norm(x - xc)^2;
end

function v = scad(x)
t = abs(x);
if t >= 0 && t <= 1
    v = 2 * t;
elseif t > 1 && t <= 2
    v = -t^2 + 4 * t - 1;
else
    v = 3;
end
end

function v = g(x)
global n p
v = 0;
for i = 1:n
    v = v + scad(x(i));
end
v = v - p;
end

function v = G(x)
global rhohat xc
v = g(x) + rhohat / 2 * norm(x - xc)^2;
end

function v = subgradf(x)
global m n A b
v = zeros(n, 1);
a = A * x;
s = sign((A * x).^2 - b);
for i = 1:n
    v(i) = sum(2 * a .* A(:,i) .* s);
end
v = v / m;
end

function v = subgradF(x)
global rhohat xc
v = subgradf(x) + rhohat * (x - xc);
end

function v = subgradscad(x)
if x == 0
    v = 4 * rand() - 2;
else
    t = abs(x);
    if t > 0 && t <= 1
        v = 2;
    elseif t > 1 && t <= 2
        v = -2 * t + 4;
    else
        v = 0;
    end
    if x < 0
        v = -v;
    end
end
end

function v = subgradg(x)
global n
v = zeros(n, 1);
for i = 1:n
    v(i) = subgradscad(x(i));
end
end

function v = subgradG(x)
global rhohat xc
v = subgradg(x) + rhohat * (x - xc);
end

function v =  proj(x)
global n domain
v = zeros(n, 1);
for i = 1:n
    if x(i) < -domain
        v(i) = -domain;
    elseif x(i) > domain
        v(i) = domain;
    else
        v(i) = x(i);
    end
end
end

function [c,ceq] = cons(x)
c = g(x);
ceq = [];
end