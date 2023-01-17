function [A,B,C,D,iter]=KLM4cRE(T,R,mez,m,numit,A,B,C,D,muAll)
%
% Krylov-Levenberg-Marquardt method for CP decomposition of order-4
% tensors. Input:
%
%    T... tensor to be decomposed
%    R ... tensor rank
%    mez ... constant to limit tr((A^TA)*(B^TB)*(C^TC)+(C^TC)*(B^TB)*(D^TD)+(A^TA)*(C^TC)*(D^TD)+(A^TA)*(B^TB)*(D^TD))
%    m ... parameter controlling complexity of one iteration of KLM
%    numit .... required number of iterations
%    A,B,C,D .... initial estimate of factor matrices (if available)
%
% Programmed by Petr Tichavsky, October 2018
%
[Ia,Ib,Ic,Id]=size(T); 
Iab=Ia+Ib;
Iabc=Iab+Ic;
Iabcd=Iabc+Id;
if nargin<9
    A=randn(Ia,R);
    B=randn(Ib,R);
    C=randn(Ic,R);
    D=randn(Id,R);
end  
if nargin<10
   muAll=logspace(-2,2,30);
end
imuAll=length(muAll);
fmu=sqrt(muAll(imuAll)/muAll(1));
a1=sum(A.^2); b1=sum(B.^2); c1=sum(C.^2); d1=sum(D.^2);
p1=sum(a1.*b1.*(c1+d1)+c1.*d1.*(a1+b1));
nr=(mez/p1)^(1/6);
A=A*nr; B=B*nr; C=C*nr; D=D*nr;   
Y1=reshape(T,Ia,Ib*Ic*Id);
Y2=reshape(permute(T,[2,3,4,1]),Ib,Ia*Ic*Id);
Y3=reshape(permute(T,[3,4,1,2]),Ic,Ia*Ib*Id);
Y4=reshape(permute(T,[4,1,2,3]),Id,Ia*Ib*Ic);
iter=zeros(1,numit); 
na=sum(A.^2); nb=sum(B.^2); nc=sum(C.^2);  nd=sum(D.^2);
mm=[nb.*nc.*nd na.*nc.*nd nd.*nb.*na na.*nb.*nc];
muAll=muAll*max(mm);  % initial parameter mu
% err=chyba(Y1,A,B,C,D);              %%% computes the error of the current approximation
for it=1:numit
    AA=A'*A;
    BB=B'*B;
    CC=C'*C;
    DD=D'*D;
    gA=reshape(T,Ia,Ib*Ic*Id)*krb(D,krb(C,B))-A*(BB.*CC.*DD);   % computes error gradient (mttkrp)
    gB=Y2*krb(A,krb(D,C))-B*(AA.*CC.*DD);
    gC=Y3*krb(B,krb(A,D))-C*(BB.*AA.*DD);
    gD=Y4*krb(C,krb(B,A))-D*(BB.*AA.*CC);
    g=[gA(:); gB(:); gC(:); gD(:)];
    Y=zeros(R*Iabcd,m+2); %  Krylov subspace
    Z=zeros(R*Iabcd,m+1);
    ng=norm(g);
    a1=sum(A.^2); b1=sum(B.^2); c1=sum(C.^2); d1=sum(D.^2);
    u1=A.*repmat(b1.*c1+c1.*d1+b1.*d1,Ia,1); u2=B.*repmat(a1.*c1+c1.*d1+a1.*d1,Ib,1); 
    u3=C.*repmat(b1.*a1+b1.*d1+a1.*d1,Ic,1); u4=D.*repmat(b1.*a1+b1.*c1+a1.*c1,Id,1);
    u=[u1(:); u2(:); u3(:); u4(:)];
    u=u/norm(u);
    Y(:,1)=g/ng;
    gA=gA/ng; gB=gB/ng; gC=gC/ng; gD=gD/ng;
    W=zeros(m+1,m+1); 
    for i=1:m+1
        i1=max([1,i-1]);
        yA=gA*(CC.*BB.*DD)+A*((gB'*B).*CC.*DD+((gC'*C).*DD+(gD'*D).*CC).*BB);
        yB=gB*(AA.*CC.*DD)+B*(((gC'*C).*AA+(gA'*A).*CC).*DD+(gD'*D).*AA.*CC);
        yC=gC*(AA.*BB.*DD)+C*(((gB'*B).*AA+(gA'*A).*BB).*DD+(gD'*D).*BB.*AA);
        yD=gD*(AA.*BB.*CC)+D*(((gB'*B).*AA+(gA'*A).*BB).*CC+(gC'*C).*BB.*AA);
        Z(:,i)=[yA(:); yB(:); yC(:); yD(:)];
        W(i1:i,i)=Y(:,i1:i)'*Z(:,i);
        Y(:,i+1)=Z(:,i)-Y(:,i1:i)*W(i1:i,i);
        Y(:,i+1)=Y(:,i+1)/norm(Y(:,i+1));
        gA=reshape(Y(1:R*Ia,i+1),Ia,R); 
        gB=reshape(Y(R*Ia+1:R*Iab,i+1),Ib,R); 
        gC=reshape(Y(R*Iab+1:R*Iabc,i+1),Ic,R);
        gD=reshape(Y(R*Iabc+1:R*Iabcd,i+1),Id,R);
    end
    W=W+W'-diag(diag(W)); iW=inv(W);
    errmin=1e40;
    A1=A; B1=B; C1=C; D1=D; chyby=muAll;
for j=1:imuAll
    mu=muAll(j);
   % a2=[u,g]/mu-Y(:,1:m+1)/(iW+eye(m+1)/mu)*(Y(:,1:m+1)'*[u,g])/mu^2;
    a2=[u,g]/mu-Y(:,1:m+1)*((iW+eye(m+1)/mu)\([u,g]'*Y(:,1:m+1))')/mu^2;
    a=a2(:,1); c=a2(:,2);
    d=c-a*(a'*g)/(u'*a);
    A=A1+reshape(d(1:Ia*R),Ia,R); 
    B=B1+reshape(d(1+Ia*R:(Ia+Ib)*R),Ib,R); 
    C=C1+reshape(d(1+(Ia+Ib)*R:Iabc*R),Ic,R); 
    D=D1+reshape(d(1+Iabc*R:Iabcd*R),Id,R); 
    a1=sum(A.^2); b1=sum(B.^2); c1=sum(C.^2); d1=sum(D.^2);
    p1=sum(a1.*b1.*(c1+d1)+c1.*d1.*(a1+b1));
    nr=(mez/p1)^(1/6);
    A=A*nr; B=B*nr; C=C*nr; D=D*nr;   
    err2=chyba(Y1,A,B,C,D);
    chyby(j)=err2;
    if err2<errmin
      errmin=err2;  
      Amin=A; Bmin=B; Cmin=C; Dmin=D;       
      jmin=j;
    end
end  
iter(it)=errmin;
A=Amin; B=Bmin; C=Cmin; D=Dmin;
semilogx(muAll,chyby,'-',muAll(jmin),chyby(jmin),'*')
drawnow
if jmin==1
     muAll=muAll/fmu;
end
if jmin==imuAll
    muAll=muAll*fmu;
end  
    if rem(it,10)==0
        [it errmin]
    end    
end
end
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [err,Y]=chyba(X,A,B,C,D)
%
% computes an error of approximation of X by sum
% of outer products of columns of A,B and C
%
Y=X-A*krb(D,krb(C,B))';
err=sum(Y(:).^2);
end
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function AB = krb(A,B)
%KRB Khatri-Rao product
%
% The columnwise Khatri-Rao-Bro product (Harshman, J.Chemom., 2002, 198-205)
% For two matrices with similar column dimension the khatri-Rao-Bro product
% is krb(A,B) = [kron(A(:,1),B(:,1)) .... kron(A(:,F),B(:,F))]
% 
% I/O AB = krb(A,B);
%

% Copyright (C) 1995-2006  Rasmus Bro & Claus Andersson
% Copenhagen University, DK-1958 Frederiksberg, Denmark, rb@life.ku.dk
%
% This program is free software; you can redistribute it and/or modify it under 
% the terms of the GNU General Public License as published by the Free Software 
% Foundation; either version 2 of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but WITHOUT 
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
% FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
% You should have received a copy of the GNU General Public License along with 
% this program; if not, write to the Free Software Foundation, Inc., 51 Franklin 
% Street, Fifth Floor, Boston, MA  02110-1301, USA.

% $ Version 1.02 $ Date 28. July 1998 $ Not compiled $
% $ Version 2.00 $ May 2001 $ Changed to array notation $ RB $ Not compiled $
% $ Version 2.01 $ May 2001 $ Error in helpfile - A and B reversed $ RB $ Not compiled $

[I,F]=size(A);
[J,F1]=size(B);

if F~=F1
   error(' Error in krb.m - The matrices must have the same number of columns')
end

AB=zeros(I*J,F);
for f=1:F
   ab=B(:,f)*A(:,f).';
   AB(:,f)=ab(:);
end
end
