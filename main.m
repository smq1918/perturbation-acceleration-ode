set(0, 'DefaultAxesFontSize', 20);           % set axes font
set(0, 'DefaultLineLineWidth', 2);
figure(1);
set(gcf, 'Position', [100, 100, 909, 600]);
%%%text of quadratic functions

mu = 1;
L = 100;
n = 100;
%a = [mu,L];
a = zeros(n,1);
for i = 1:n
    a(i) = exp(log(mu) + (i-1)/(n-1)*(log(L)-log(mu)));
    %a(i) = mu+(i-1)/(n-1)*(L-mu);
end
%seed = 42;
%rng(seed);
[Q,~] = qr(randn(n));
A=diag(a);
A = Q*A*Q';


%case: delta1 = 0,delta2 = 0
x0 = ones(n,1);
x0 = Q*x0;
s = 1/L;
delta1 = 0;
delta2 = 0;
g0 = A*x0;
%y0 = x0 - 1/L*g0;
x1 = x0 - (1+delta1)/(1+2*sqrt(mu*s))*s*g0;
g1 = A*x1;
%y1 = x1 - 1/L*g1;

f0 = 1/2*x0'*A*x0;
f1 = 1/2*x1'*A*x1;
%v = [norm(x1-x0)];

%{
f0 = 1/2*y0'*A*y0;
f1 = 1/2*y1'*A*y1;
v = [norm(y1-y0)];
%}
f = [f0,f1];
eps = 1e-6;
%while abs(f1-f0) >= 1e-6 && norm(g1) >= 1e-8
while norm(g1) >= eps
    temp = x1 + 1/(1+2*sqrt(mu*s))*(x1-x0-(1+delta1)*s*g1); 
    f0 = f1;
    x0 = x1;
    g0 = g1;
    %y0 = y1;
    x1 = temp;
    g1 = A*x1;
    %y1 = x1-1/L*g1;
    
    f1 = 1/2*x1'*A*x1;
    %v = [v,norm(x1-x0)];
    
    %{
    f1 = 1/2*y1'*A*y1;
    v = [v,norm(y1-y0)];
    %}
    f = [f,f1];
end
figure(1);
semilogy(f,"red")
hold on
%{
figure(2);
semilogy(v,"red")
hold on
%}


%case: delta1 = 0,delta2 > 0
x0 = ones(n,1);
x0 = Q*x0;
s = 1/L;
delta1 = 0;
delta2 = sqrt(s);
%delta2 = 2/3*sqrt(s);
g0 = A*x0;
%y0 = x0 - 1/L*g0;
x1 = x0 - (1+delta1)/(1+2*sqrt(mu*s))*s*g0;
g1 = A*x1;
%y1 = x1 - 1/L*g1;

f0 = 1/2*x0'*A*x0;
f1 = 1/2*x1'*A*x1;
%v = [norm(x1-x0)];

%{
f0 = 1/2*y0'*A*y0;
f1 = 1/2*y1'*A*y1;
v = [norm(y1-y0)];
%}
f = [f0,f1];
%while abs(f1-f0) >= 1e-6 && norm(g1) >= 1e-8
while norm(g1) >= eps
    temp = x1 + 1/(1+2*sqrt(mu*s))*(x1-x0-(1+delta1)*s*g1-delta2*sqrt(s)*(g1-g0));
    f0 = f1;
    x0 = x1;
    g0 = g1;
    %y0 = y1;
    x1 = temp;
    g1 = A*x1;
    %y1 = x1 - 1/L*g1;
    
    f1 = 1/2*x1'*A*x1;
    %v = [v,norm(x1-x0)];
    
    %{
    f1 = 1/2*y1'*A*y1;
    v = [v,norm(y1-y0)];
    %}
    f = [f,f1];
end
figure(1);
semilogy(f,"green")
hold on
%{
figure(2);
semilogy(v,"green")
hold on
%}


%case: delta1 > 0,delta2 = 0
x0 = ones(n,1);
x0 = Q*x0;
s = 1/L;
%delta1 = sqrt(mu*s);
delta1 = 1;
delta2 = 0;
g0 = A*x0;
%y0 = x0 - 1/L*g0;
x1 = x0 - (1+delta1)/(1+2*sqrt(mu*s))*s*g0;
g1 = A*x1;
%y1 = x1 - 1/L*g1;

f0 = 1/2*x0'*A*x0;
f1 = 1/2*x1'*A*x1;
%v = [norm(x1-x0)];

%{
f0 = 1/2*y0'*A*y0;
f1 = 1/2*y1'*A*y1;
v = [norm(y1-y0)];
%}
f = [f0,f1];
%while abs(f1-f0) >= 1e-6 && norm(g1) >= 1e-8
while norm(g1) >= eps
    temp = x1 + 1/(1+2*sqrt(mu*s))*(x1-x0-(1+delta1)*s*g1); 
    f0 = f1;
    x0 = x1;
    g0 = g1;
    %y0 = y1;
    x1 = temp;
    g1 = A*x1;
    %y1 = x1 - 1/L*g1;
    
    f1 = 1/2*x1'*A*x1;
    %v = [v,norm(x1-x0)];
    
    %{
    f1 = 1/2*y1'*A*y1;
    v = [v,norm(y1-y0)];
    %}
    f = [f,f1];
end
figure(1);
semilogy(f,"blue")
hold on
%{
figure(2);
semilogy(v,"blue")
hold on
%}


%case: delta1 > 0, delta2 > 0
x0 = ones(n,1);
x0 = Q*x0;
s = 1/L;
%delta1 = sqrt(mu*s);
delta1 = 1;
delta2 = sqrt(s);
%delta2 = 2/3*sqrt(s);
g0 = A*x0;
%y0 = x0 - 1/L*g0;
x1 = x0 - (1+delta1)/(1+2*sqrt(mu*s))*s*g0;
g1 = A*x1;
%y1 = x1 - 1/L*g1;

f0 = 1/2*x0'*A*x0;
f1 = 1/2*x1'*A*x1;
%v = [norm(x1-x0)];

%{
f0 = 1/2*y0'*A*y0;
f1 = 1/2*y1'*A*y1;
v = [norm(y1-y0)];
%}
f = [f0,f1];
%while abs(f1-f0) >= 1e-6 && norm(g1) >= 1e-8
while norm(g1) >= eps
    temp = x1 + 1/(1+2*sqrt(mu*s))*(x1-x0-(1+delta1)*s*g1-delta2*sqrt(s)*(g1-g0));
    f0 = f1;
    x0 = x1;
    g0 = g1;
    %y0 = y1;
    x1 = temp;
    g1 = A*x1;
    %y1 = x1 - s*g1;
    
    f1 = 1/2*x1'*A*x1;
    %v = [v,norm(x1-x0)];
    
    %{
    f1 = 1/2*y1'*A*y1;
    v = [v,norm(y1-y0)];
    %}
    f = [f,f1];
end
figure(1);
semilogy(f,"Color",[1 0.5 0])
hold on
%{
figure(2);
semilogy(v,"Color",[1 0.5 0])
hold on
%}

%NAG-SC

x0 = ones(n,1);
x0 = Q*x0;
s = 1/L;
y1 = x0;
g0 = A*y1;
x1 = y1 - s*g0;
f0 = 0;
%f1 = 1/2*x1'*A*x1;
f1 = 1/2*y1'*A*y1;
h = f1;
%v = [];
%while abs(f1-f0) >= 1e-6 && norm(g0) >= 1e-8
while norm(g0) >= eps
    y0 = y1;
    y1 = x1+(1-sqrt(mu*s))/(1+sqrt(mu*s))*(x1-x0);
    x0 = x1;
    f0 = f1;
    g0 = A*y1;
    x1 = y1 - s*g0;
    
    f1 = 1/2*y1'*A*y1;
    %v = [v,norm(y1-y0)];
    
    %{
    f1 = 1/2*x1'*A*x1;
    v = [v,norm(x1-x0)];
    %}
    h = [h,f1];  
end



%\hat{Delta}_1 = \sqrt{\mu s}, \hat{Delta}_2 = \sqrt{s}
%{
figure(1);
semilogy(h,"black");
legend('$\Delta_1=0,\Delta_2=0$','$\Delta_1=0,\Delta_2=\sqrt{s}$', ...
    '$\Delta_1=\sqrt{\mu s},\Delta_2=0$','$\Delta_1=\sqrt{\mu s},\Delta_2=\sqrt{s}$', ...
    'NAG-SC', 'Interpreter','latex','Position', [0.55, 0.55, 0.33, 0.33]);
xlabel('iteration ($k$)', 'Interpreter','latex');
ylabel('$\log_{10}(f(x_{k})-f(x^*))$','Interpreter','latex');

hold off
%}

%\hat{Delta}_1 = 1, \hat{Delta}_2 = \sqrt{s}

figure(1);
semilogy(h,"black");
legend('$\Delta_1=0,\Delta_2=0$','$\Delta_1=0,\Delta_2=\sqrt{s}$', ...
    '$\Delta_1=1,\Delta_2=0$','$\Delta_1=1,\Delta_2=\sqrt{s}$', ...
    'NAG-SC', 'Interpreter','latex','Position', [0.55, 0.55, 0.33, 0.33]);
xlabel('iteration ($k$)', 'Interpreter','latex');
ylabel('$\log_{10}(f(x_{k})-f(x^*))$','Interpreter','latex');

hold off


%\hat{Delta}_1 = \sqrt{\mu s}, \hat{Delta}_2 = \frac{2}{3}\sqrt{s}
%{
figure(1);
semilogy(h,"black");
legend('$\Delta_1=0,\Delta_2=0$','$\Delta_1=0,\Delta_2=\frac{2}{3}\sqrt{s}$', ...
    '$\Delta_1=\sqrt{\mu s},\Delta_2=0$','$\Delta_1=\sqrt{\mu s},\Delta_2=\frac{2}{3}\sqrt{s}$', ...
    'NAG-SC', 'Interpreter','latex','Position', [0.55, 0.55, 0.33, 0.33]);
xlabel('iteration ($k$)', 'Interpreter','latex');
ylabel('$\log_{10}(f(x_{k})-f(x^*))$','Interpreter','latex');

hold off
%}



%%%text of logistic regression
%{

%{
dataset = 'a9a.test';
[b,A] = libsvmread(dataset);
%}
%{
dataset = 'CINA.test';
[b,A] = libsvmread(dataset);
%}

dataset = 'ijcnn1.test';
[b,A] = libsvmread(dataset);


[m,n] = size(A);
mu = 1e-2;
L = 1/m/4*norm(A,"fro")^2 + mu;

%The function value obtained by NAG-SC under stricter stopping criterion  
% is taken as the approximate optimal solution.
x0 = zeros(n,1);
s = 1/L;
y1 = x0;
g0 = zeros(n,1);
for i = 1 : m
    g0 = g0 - 1/(1+exp(b(i)*A(i,:)*y1))*b(i)*A(i,:)';
end
g0 = 1/m*g0 + mu*y1;
x1 = y1 - s*g0;

f1 = 0;
for i = 1 : m
    f1 = f1 + log(1+exp(-b(i)*A(i,:)*y1));
end
f1 = 1/m*f1 + 1/2*mu*(y1')*y1;

while norm(g0) >= 1e-8
    y0 = y1;
    y1 = x1+(1-sqrt(mu*s))/(1+sqrt(mu*s))*(x1-x0);
    x0 = x1;
    g0 = zeros(n,1);
    for i = 1 : m
        g0 = g0 - 1/(1+exp(b(i)*A(i,:)*y1))*b(i)*A(i,:)';
    end
    g0 = 1/m*g0 + mu*y1;
    x1 = y1 - s*g0;
    f1 = 0;
    for i = 1 : m
        f1 = f1 + log(1+exp(-b(i)*A(i,:)*y1));
    end
    f1 = 1/m*f1 + 1/2*mu*(y1')*y1;
end
f_opt = f1;




%case: delta1 = 0, delta2 = 0
x0 = zeros(n,1);
s = 1/L;
delta1 = 0;
delta2 = 0;
g0 = zeros(n,1);
for i = 1 : m
    g0 = g0 - 1/(1+exp(b(i)*A(i,:)*x0))*b(i)*A(i,:)';
end
g0 = 1/m*g0 + mu*x0;
%y0 = x0 - 1/L*g0;
x1 = x0 - (1+delta1)/(1+2*sqrt(mu*s))*s*g0;
g1 = zeros(n,1);
for i = 1 : m
    g1 = g1 - 1/(1+exp(b(i)*A(i,:)*x1))*b(i)*A(i,:)';
end
g1 = 1/m*g1 + mu*x1;
%y1 = x1 - 1/L*g1;

f0 = 0;
f1 = 0;
for i = 1 : m
    f0 = f0 + log(1+exp(-b(i)*A(i,:)*x0));
    f1 = f1 + log(1+exp(-b(i)*A(i,:)*x1));
end
f0 = 1/m*f0 + 1/2*mu*(x0')*x0;
f1 = 1/m*f1 + 1/2*mu*(x1')*x1;
f = [f0,f1];
%v = [norm(x1-x0)];

%{
f0 = 0;
f1 = 0;
for i = 1 : m
    f0 = f0 + log(1+exp(-b(i)*A(i,:)*y0));
    f1 = f1 + log(1+exp(-b(i)*A(i,:)*y1));
end
f0 = 1/m*f0 + mu*(y0')*y0;
f1 = 1/m*f1 + mu*(y1')*y1;
f = [f0,f1];
v = [norm(y1-y0)];
%}

%while abs(f1-f0) >= 1e-6 && norm(g1) >= 1e-8
while norm(g1) >= 1e-6
    temp = x1 + 1/(1+2*sqrt(mu*s))*(x1-x0-(1+delta1)*s*g1-delta2*sqrt(s)*(g1-g0)); 
    f0 = f1;
    x0 = x1;
    g0 = g1;
    %y0 = y1;
    x1 = temp;
    g1 = zeros(n,1);
    for i = 1 : m
        g1 = g1 - 1/(1+exp(b(i)*A(i,:)*x1))*b(i)*A(i,:)';
    end
    g1 = 1/m*g1 + mu*x1;
    %y1 = x1 - 1/L*g1;
    
    f1 = 0;
    for i = 1 : m
        f1 = f1 + log(1+exp(-b(i)*A(i,:)*x1));
    end
    f1 = 1/m*f1 + 1/2*mu*(x1')*x1;
    %v = [v,norm(x1-x0)];
    
    %{
    f1 = 0;
    for i = 1 : m
        f1 = f1 + log(1+exp(-b(i)*A(i,:)*y1));
    end
    f1 = 1/m*f1 + mu*(y1')*y1;
    v = [v,norm(y1-y0)];
    %}
    f = [f,f1];
end
figure(1);
semilogy(f-f_opt,"red")
hold on
%{
figure(2);
semilogy(v,"red")
hold on
%}


%case: delta1 = 0, delta2 > 0
x0 = zeros(n,1);
s = 1/L;
delta1 = 0;
delta2 = sqrt(s);
%delta2 = 2/3*sqrt(s);
g0 = zeros(n,1);
for i = 1 : m
    g0 = g0 - 1/(1+exp(b(i)*A(i,:)*x0))*b(i)*A(i,:)';
end
g0 = 1/m*g0 + mu*x0;
%y0 = x0 - 1/L*g0;
x1 = x0 - (1+delta1)/(1+2*sqrt(mu*s))*s*g0;
g1 = zeros(n,1);
for i = 1 : m
    g1 = g1 - 1/(1+exp(b(i)*A(i,:)*x1))*b(i)*A(i,:)';
end
g1 = 1/m*g1 + mu*x1;
%y1 = x1 - 1/L*g1;

f0 = 0;
f1 = 0;
for i = 1 : m
    f0 = f0 + log(1+exp(-b(i)*A(i,:)*x0));
    f1 = f1 + log(1+exp(-b(i)*A(i,:)*x1));
end
f0 = 1/m*f0 + 1/2*mu*(x0')*x0;
f1 = 1/m*f1 + 1/2*mu*(x1')*x1;
f = [f0,f1];
%v = [norm(x1-x0)];

%{
f0 = 0;
f1 = 0;
for i = 1 : m
    f0 = f0 + log(1+exp(-b(i)*A(i,:)*y0));
    f1 = f1 + log(1+exp(-b(i)*A(i,:)*y1));
end
f0 = 1/m*f0 + mu*(y0')*y0;
f1 = 1/m*f1 + mu*(y1')*y1;
f = [f0,f1];
v = [norm(y1-y0)];
%}

%while abs(f1-f0) >= 1e-6 && norm(g1) >= 1e-8
while norm(g1) >= 1e-6
    temp = x1 + 1/(1+2*sqrt(mu*s))*(x1-x0-(1+delta1)*s*g1-delta2*sqrt(s)*(g1-g0)); 
    f0 = f1;
    x0 = x1;
    g0 = g1;
    %y0 = y1;
    x1 = temp;
    g1 = zeros(n,1);
    for i = 1 : m
        g1 = g1 - 1/(1+exp(b(i)*A(i,:)*x1))*b(i)*A(i,:)';
    end
    g1 = 1/m*g1 + mu*x1;
    %y1 = x1 - 1/L*g1;
    
    f1 = 0;
    for i = 1 : m
        f1 = f1 + log(1+exp(-b(i)*A(i,:)*x1));
    end
    f1 = 1/m*f1 + 1/2*mu*(x1')*x1;
    %v = [v,norm(x1-x0)];
    
    %{
    f1 = 0;
    for i = 1 : m
        f1 = f1 + log(1+exp(-b(i)*A(i,:)*y1));
    end
    f1 = 1/m*f1 + mu*(y1')*y1;
    v = [v,norm(y1-y0)];
    %}
    f = [f,f1];
end
figure(1);
semilogy(f-f_opt,"green")
hold on
%{
figure(2);
semilogy(v,"green")
hold on
%}


%case: delta1 > 0, delta2 = 0
x0 = zeros(n,1);
s = 1/L;
delta1 = 1;
%delta1 = sqrt(mu*s);
delta2 = 0;
g0 = zeros(n,1);
for i = 1 : m
    g0 = g0 - 1/(1+exp(b(i)*A(i,:)*x0))*b(i)*A(i,:)';
end
g0 = 1/m*g0 + mu*x0;
%y0 = x0 - 1/L*g0;
x1 = x0 - (1+delta1)/(1+2*sqrt(mu*s))*s*g0;
g1 = zeros(n,1);
for i = 1 : m
    g1 = g1 - 1/(1+exp(b(i)*A(i,:)*x1))*b(i)*A(i,:)';
end
g1 = 1/m*g1 + mu*x1;
%y1 = x1 - 1/L*g1;

f0 = 0;
f1 = 0;
for i = 1 : m
    f0 = f0 + log(1+exp(-b(i)*A(i,:)*x0));
    f1 = f1 + log(1+exp(-b(i)*A(i,:)*x1));
end
f0 = 1/m*f0 + 1/2*mu*(x0')*x0;
f1 = 1/m*f1 + 1/2*mu*(x1')*x1;
f = [f0,f1];
%v = [norm(x1-x0)];

%{
f0 = 0;
f1 = 0;
for i = 1 : m
    f0 = f0 + log(1+exp(-b(i)*A(i,:)*y0));
    f1 = f1 + log(1+exp(-b(i)*A(i,:)*y1));
end
f0 = 1/m*f0 + mu*(y0')*y0;
f1 = 1/m*f1 + mu*(y1')*y1;
f = [f0,f1];
v = [norm(y1-y0)];
%}

%while abs(f1-f0) >= 1e-6 && norm(g1) >= 1e-8
while norm(g1) >= 1e-6
    temp = x1 + 1/(1+2*sqrt(mu*s))*(x1-x0-(1+delta1)*s*g1-delta2*sqrt(s)*(g1-g0)); 
    f0 = f1;
    x0 = x1;
    g0 = g1;
    %y0 = y1;
    x1 = temp;
    g1 = zeros(n,1);
    for i = 1 : m
        g1 = g1 - 1/(1+exp(b(i)*A(i,:)*x1))*b(i)*A(i,:)';
    end
    g1 = 1/m*g1 + mu*x1;
    %y1 = x1 - 1/L*g1;
    
    f1 = 0;
    for i = 1 : m
        f1 = f1 + log(1+exp(-b(i)*A(i,:)*x1));
    end
    f1 = 1/m*f1 + 1/2*mu*(x1')*x1;
    %v = [v,norm(x1-x0)];
    
    %{
    f1 = 0;
    for i = 1 : m
        f1 = f1 + log(1+exp(-b(i)*A(i,:)*y1));
    end
    f1 = 1/m*f1 + mu*(y1')*y1;
    v = [v,norm(y1-y0)];
    %}
    f = [f,f1];
end
figure(1)
semilogy(f-f_opt,"blue")
hold on
%{
figure(2)
semilogy(v,"blue")
hold on
%}


%case: delta1 > 0, delta2 > 0
x0 = zeros(n,1);
s = 1/L;
delta1 = 1;
%delta1 = sqrt(mu*s);
delta2 = sqrt(s);
%delta2 = 2/3*sqrt(s);
g0 = zeros(n,1);
for i = 1 : m
    g0 = g0 - 1/(1+exp(b(i)*A(i,:)*x0))*b(i)*A(i,:)';
end
g0 = 1/m*g0 + mu*x0;
%y0 = x0 - 1/L*g0;
x1 = x0 - (1+delta1)/(1+2*sqrt(mu*s))*s*g0;
g1 = zeros(n,1);
for i = 1 : m
    g1 = g1 - 1/(1+exp(b(i)*A(i,:)*x1))*b(i)*A(i,:)';
end
g1 = 1/m*g1 + mu*x1;
%y1 = x1 - 1/L*g1;

f0 = 0;
f1 = 0;
for i = 1 : m
    f0 = f0 + log(1+exp(-b(i)*A(i,:)*x0));
    f1 = f1 + log(1+exp(-b(i)*A(i,:)*x1));
end
f0 = 1/m*f0 + 1/2*mu*(x0')*x0;
f1 = 1/m*f1 + 1/2*mu*(x1')*x1;
f = [f0,f1];
%v = [norm(x1-x0)];

%{
f0 = 0;
f1 = 0;
for i = 1 : m
    f0 = f0 + log(1+exp(-b(i)*A(i,:)*y0));
    f1 = f1 + log(1+exp(-b(i)*A(i,:)*y1));
end
f0 = 1/m*f0 + mu*(y0')*y0;
f1 = 1/m*f1 + mu*(y1')*y1;
f = [f0,f1];
v = [norm(y1-y0)];
%}
%while abs(f1-f0) >= 1e-6 && norm(g1) >= 1e-8
while norm(g1) >= 1e-6
    temp = x1 + 1/(1+2*sqrt(mu*s))*(x1-x0-(1+delta1)*s*g1-delta2*sqrt(s)*(g1-g0)); 
    f0 = f1;
    x0 = x1;
    g0 = g1;
    %y0 = y1;
    x1 = temp;
    g1 = zeros(n,1);
    for i = 1 : m
        g1 = g1 - 1/(1+exp(b(i)*A(i,:)*x1))*b(i)*A(i,:)';
    end
    g1 = 1/m*g1 + mu*x1;
    %y1 = x1 - 1/L*g1;
    
    f1 = 0;
    for i = 1 : m
        f1 = f1 + log(1+exp(-b(i)*A(i,:)*x1));
    end
    f1 = 1/m*f1 + 1/2*mu*(x1')*x1;
    %v = [v,norm(x1-x0)];
    
    %{
    f1 = 0;
    for i = 1 : m
        f1 = f1 + log(1+exp(-b(i)*A(i,:)*y1));
    end
    f1 = 1/m*f1 + mu*(y1')*y1;
    v = [v,norm(y1-y0)];
    %}
    f = [f,f1];
end
figure(1);
semilogy(f-f_opt,"Color",[1 0.5 0])
hold on
%{
figure(2);
semilogy(v,"Color",[1 0.5 0])
hold on
%}


%NAG-SC

x0 = zeros(n,1);
s = 1/L;
y1 = x0;
g0 = zeros(n,1);
for i = 1 : m
    g0 = g0 - 1/(1+exp(b(i)*A(i,:)*y1))*b(i)*A(i,:)';
end
g0 = 1/m*g0 + mu*y1;
x1 = y1 - s*g0;
%{
f1 = 0;
for i = 1 : m
    f1 = f1 + log(1+exp(-b(i)*A(i,:)*x1));
end
f1 = 1/m*f1 + mu*(x1')*x1;
f0 = f1 + 1;
%}

f1 = 0;
for i = 1 : m
    f1 = f1 + log(1+exp(-b(i)*A(i,:)*y1));
end
f1 = 1/m*f1 + 1/2*mu*(y1')*y1;
f0 = f1 + 1;

h = f1;
%v = [];
%while abs(f1-f0) >= 1e-6 && norm(g0) >= 1e-8
while norm(g0) >= 1e-6
    y0 = y1;
    y1 = x1+(1-sqrt(mu*s))/(1+sqrt(mu*s))*(x1-x0);
    x0 = x1;
    g0 = zeros(n,1);
    for i = 1 : m
        g0 = g0 - 1/(1+exp(b(i)*A(i,:)*y1))*b(i)*A(i,:)';
    end
    g0 = 1/m*g0 + mu*y1;
    x1 = y1 - s*g0;
    f0 = f1;
    f1 = 0;
    %{
    for i = 1 : m
        f1 = f1 + log(1+exp(-b(i)*A(i,:)*x1));
    end
    f1 = 1/m*f1 + mu*(x1')*x1;
    v = [v,norm(x1-x0)];
    %}
    
    for i = 1 : m
        f1 = f1 + log(1+exp(-b(i)*A(i,:)*y1));
    end
    f1 = 1/m*f1 + 1/2*mu*(y1')*y1;
    %v = [v,norm(y1-y0)];
    
    h = [h,f1];
end
figure(1);
semilogy(h-f_opt,"Color","black")
hold on


%\hat{Delta}_1 = \sqrt{\mu s}, \hat{Delta}_2 = \sqrt{s}
%{
legend('$\Delta_1=0,\Delta_2=0$','$\Delta_1=0,\Delta_2=\sqrt{s}$', ...
    '$\Delta_1=\sqrt{\mu s},\Delta_2=0$','$\Delta_1=\sqrt{\mu s},\Delta_2=\sqrt{s}$', ...
    'NAG-SC', 'Interpreter','latex','Position', [0.55, 0.55, 0.33, 0.33]);
xlabel('iteration ($k$)', 'Interpreter','latex');
ylabel('$\log_{10}(f(x_{k})-f(x^*))$','Interpreter','latex');

hold off
%}

%\hat{Delta}_1 = 1, \hat{Delta}_2 = \sqrt{s}

legend('$\Delta_1=0,\Delta_2=0$','$\Delta_1=0,\Delta_2=\sqrt{s}$', ...
    '$\Delta_1=1,\Delta_2=0$','$\Delta_1=1,\Delta_2=\sqrt{s}$', ...
    'NAG-SC', 'Interpreter','latex','Position', [0.55, 0.55, 0.33, 0.33]);
xlabel('iteration ($k$)', 'Interpreter','latex');
ylabel('$\log_{10}(f(x_{k})-f(x^*))$','Interpreter','latex');

hold off


%\hat{Delta}_1 = \sqrt{\mu s}, \hat{Delta}_2 = \frac{2}{3}\sqrt{s}
%{
legend('$\Delta_1=0,\Delta_2=0$','$\Delta_1=0,\Delta_2=\frac{2}{3}\sqrt{s}$', ...
    '$\Delta_1=\sqrt{\mu s},\Delta_2=0$','$\Delta_1=\sqrt{\mu s},\Delta_2=\frac{2}{3}\sqrt{s}$', ...
    'NAG-SC', 'Interpreter','latex','Position', [0.55, 0.55, 0.33, 0.33]);
xlabel('iteration ($k$)', 'Interpreter','latex');
ylabel('$\log_{10}(f(x_{k})-f(x^*))$','Interpreter','latex');

hold off
%}
%}


