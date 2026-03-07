function [FC,CV,Cvth,A]=hopf_int(gC,f_diff,sigma)
N=size(gC,1);

a=-0.02;
%gamma_sig = 5;
%gamma=0.01;
%a = -0.02 + gamma * pmap_sub;


wo = f_diff'*(2*pi);

%Cvth = zeros(2*N);

% Jacobian:

s = sum(gC,2);
B = diag(s);

Axx = a*eye(N) - B + gC;
%Axx = diag(a) - B + gC;
Ayy = Axx;
Axy = -diag(wo);
Ayx = diag(wo);

A = [Axx Axy; Ayx Ayy];

Qn = (sigma^2)*eye(2*N);
%sigma_per=sigma^2*(1+gamma_sig*pmap_sub);
%Qn = diag(repmat(sigma_per, 1, 2));


Cvth=sylvester(A,A',-Qn);
FCth=corrcov(Cvth);
FC=FCth(1:N,1:N);
CV=Cvth(1:N,1:N);
