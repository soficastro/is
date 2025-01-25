%Algoritmo de detecção de estrutura Forward Regressive Orthogonal
%Least Squares (FROLS) BILLINGS (pag 74).
%Com o critério de informação de Akaike e Bayes
function [Qs, As, gs, teta_CGSs, ERR, indice] = alg_FROLS_AIK(fi, Y)
%fi = matriz de regressores
%Y = vetor coluna de saída
y = Y;

[N,n_teta]  = size(fi);
%n_l = número de amostras (número de linhas)
%n_teta = número de parâmetros (número de colunas)

%Variáveis auxiliares utilizadas para eliminar as colunas já escolhidas
index = 1:n_teta;
indice_final = zeros(1,n_teta);

%Passo 1 - Fazer a matriz ortogonal igual a matriz candidata
Q = fi;
Phi = fi;
As = eye(n_teta, n_teta);

for i = 1:n_teta
    g(i) = (Q(:,i)'*Y)/(Q(:,i)'*Q(:,i));
    err(i) = g(i).^2 .* ((Q(:,i)'*Q(:,i)) / (Y'*Y));
end

[maxERR(1), indice(1)]=max(err);

%A primeira coluna da matriz reduzida recebe a coluna de Phi com maior ERR
%As(1,1) = 1;
Qs(:,1) = Q(:,indice(1));
gs(1) = g(indice(1));
ERR(1) = err(indice(1));

indice_final(1) = index(indice(1));
index(indice(1)) = [];

% %Limiar para seleção de estrutura. (AGUIRRE, 2004) pag. 387
% %(BILLINGS, 2014) pag. 75 (enquanto for maior que 10^-2)
% rho = 0.01;
% ESR = 1;

k=1;
aux_res = zeros(length(y), 1);
for j  = 1:k
    aux_res = aux_res + gs(j)*Qs(:,j);
end
residuo = y - aux_res;

%1.14) Calcular a variância dos resíduos (C)
C(k) = (1/N)*(residuo'*residuo);

%1.15) Calcular AIC4(k) e BIC(k)
AIC4(k) = N*log(C(k)) + 4*(k);
BIC(k) = N*log(C(k)) + (k)*log(N);

Phi(:,indice(1)) = []; %Retirar melhor regressor já escolhido

% Passo 2
k = 2; %Variável responsável pela contagem das iterações
aux = 1; %Variável que diminui o valor da dimensão a cada iteração

while(k <= n_teta)
    clear g;
    clear Q;
    clear err;
    
    for i = 1:n_teta - aux;
        soma = 0;
        for j = 1:k-1
            soma = soma + (Phi(:,i)'*Qs(:,j))/(Qs(:,j)'*Qs(:,j)) * Qs(:,j);
        end
        Q(:,i) = Phi(:,i) - soma;
        g(i) = (Y'*Q(:,i))/(Q(:,i)'*Q(:,i));
        err(i) = g(i).^2 .* ((Q(:,i)'*Q(:,i)) / (Y'*Y));
    end
    
    [maxERR(k), indice(k)]=max(err);
    Qs(:,k) = Q(:,indice(k));
    gs(k) = g(indice(k));
    
    for j = 1:k-1
        As(j,k) = (Qs(:,j)'*Phi(:,indice(k)))/(Qs(:,j)'*Qs(:,j));
    end
    
    ERR(k) = err(indice(k));
    Phi(:,indice(k)) = [];
    
    indice_final(k) = index(indice(k));
    index(indice(k)) = [];
    
    aux_res = zeros(length(y), 1);
    for j  = 1:k
        aux_res = aux_res + gs(j)*Qs(:,j);
    end
    residuo = y - aux_res;
    
    C(k) = (1/N)*(residuo'*residuo);
    
    AIC4(k) = N*log(C(k)) + 4*(k);
    BIC(k) = N*log(C(k)) + (k)*log(N);
    
    if ((AIC4(k) >= AIC4(k-1)) || (BIC(k) >= BIC(k-1)))
        break;
    end
    
    k = k + 1;
    aux = aux +1;
    
end

indice = indice_final(1:k-1);
disp(indice);
disp(maxERR(1:k-1));
A = As; 
As = A(1:k-1, 1:k-1);

teta_CGSs = inv(As)*gs(1:k-1)';
disp(teta_CGSs)