global m n p rho rhohat M epsilon flb glb domain A b xc


% initialize, m*n for dimension, p for SCAD RHS, domain [-x,x]

m = 120;
n = 120;
p = 320;
domain = 10;
epsilon = 0.01;
flb = 0;
glb = -p;


% generate data

% load('data_inactive','xstar','A','noise','b');
xstar = [zeros(n*3/4, 1); 5 * rand(n/4, 1) + 5];
xstar = xstar(randperm(n));
A = normrnd(0, 1, m, n);
noise = randn(m, 1);
b = (A * xstar).^2 + noise;

x0 = normrnd(0, 0.1, n, 1);
while g(x0) > 0
    x0 = normrnd(0, 0.1, n, 1);
end
% load('data_inactive_','xstar','A','noise','b','x0');


% set parameters

rho = 3;
rhohat = rho * 2;
D = sqrt(-8 * glb / (rhohat - rho));
tau = (rhohat - rho) * epsilon^2 / (4 * rhohat * (2 * rhohat - rho));
M = 2 * max(max(abs(A)))^2 * domain * n * sqrt(n);
L0 = 9 * M^2 - 6 * rhohat * glb;
L1 = 6 * rhohat;


% compute and initialize numbers of iterations

Ktotal = ceil(4 * rhohat^2 * (f(x0) - flb) / ((rhohat - rho) * epsilon^2))
Ttotal = ceil(max(8 * L0^2 / ((rhohat - rho) * tau), sqrt(2 * L1^2 * D^2 / ((rhohat - rho) * tau))))
K = 1000
T = 10000


% set lists

x_list = [x0];
f_list = [f(x0)];
g_list = [g(x0)];
gradFJ_list = [];
gamma_list = [];
gradKKT_list = [];
lambda_list = [];


% run the algorithm

x = x0;
stop = 0;

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

    gamma_list(:,end+1) = alphag / (alphaf + alphag);
    lambda_list(:,end+1) = alphag / alphaf;
    gradFJ_list(end+1) = rhohat * norm(x - x_list(:,end));
    gradKKT_list(end+1) = (1 + alphag / alphaf) * rhohat * norm(x - x_list(:,end));


    % stopping criteria

    if stop == 0
        if g(x) >= 0 || f(x) >= f(x_list(:,end))
            stop = k;
        end
    end


    % stop when reaching the max number of iterations

    if k == K
        break;
    end


    % if not stopping, continue

    f_list(end+1) = f(x);
    g_list(end+1) = g(x);
    x_list(:,end+1) = x;


    % show some data during iterations

    gradFJ_list(end)
    lambda_list(end)
    k
end

stop - 1


% compute the local minimum

options = optimoptions('fmincon','Display','off','Algorithm','interior-point');
options.MaxIter = 1e9;
options.TolFun = 1e-9;
options.MaxFunEvals = 1e9;
options.TolCon = 1e-9;
optimal_x = fmincon(@f,x_list(:,end),[],[],[],[],-domain*ones(n,1),domain*ones(n,1),@cons,options);
opt_value = f(optimal_x);


% plot the figures

figure;
semilogy((0:size(x_list, 2)-1)*T, f_list - opt_value, 'k', LineWidth=2);
title('Objective value');
xlabel('Subgradient evaluations');
ylabel('f(x_k)-f(x_{lo})');
if stop > 0
    xline((stop-1)*T, '--k', LineWidth=2);
end
set(gca, 'FontSize', 20);

figure;
semilogy((0:size(x_list, 2)-1)*T, g_list, 'k', LineWidth=2);
title('Feasibility');
xlabel('Subgradient evaluations');
ylabel('g(x_k)');
if stop > 0
    xline((stop-1)*T, '--k', LineWidth=2);
end
set(gca, 'FontSize', 20);

figure;
semilogy((0:size(x_list, 2)-1)*T, gradFJ_list, 'k', LineWidth=2);
title('FJ stationarity');
xlabel('Subgradient evaluations');
ylabel('||\gamma_{k0}\zeta_{fk}+\gamma_k\zeta_{gk}||');
if stop > 0
    xline((stop-1)*T, '--k', LineWidth=2);
end
set(gca, 'FontSize', 20);

figure;
plot((0:size(x_list, 2)-1)*T, 1-gamma_list, ':k', LineWidth=2);
hold on
plot((0:size(x_list, 2)-1)*T, gamma_list, 'k', LineWidth=2);
title('\gamma_{k0} and \gamma_k');
xlabel('Subgradient evaluations');
if stop > 0
    xline((stop-1)*T, '--k', LineWidth=2);
end
legend('\gamma_{k0}', '\gamma_k');
set(gca, 'FontSize', 20);

figure;
semilogy((0:size(x_list, 2)-1)*T, gradKKT_list, 'k', LineWidth=2);
title('KKT stationarity');
xlabel('Subgradient evaluations');
ylabel('||\zeta_{fk}+\lambda_k\zeta_{gk}||');
if stop > 0
    xline((stop-1)*T, '--k', LineWidth=2);
end
set(gca, 'FontSize', 20);

figure;
plot((0:size(x_list, 2)-1)*T, lambda_list, 'k', LineWidth=2);
title('Lagrange multipliers \lambda_k');
xlabel('Subgradient evaluations');
ylabel('\lambda_k');
if stop > 0
    xline((stop-1)*T, '--k', LineWidth=2);
end
set(gca, 'FontSize', 20);


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