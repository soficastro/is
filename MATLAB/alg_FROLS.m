%Algoritmo de detec��o de estrutura Forward Regressive Orthogonal
%Least Squares (FROLS)
%Livro do BILLINGS (pag 74).
%Autor: �caro Bezerra Queiroz de Ara�jo
%Universidade Federal do Rio Grande do Norte
%Data da �ltima modifica��o: 16/07/2018
%-------------------------------------------------------------------------%
%Dados de sa�da
%Qs -> Matriz dos regressores ortogonalizada
%As -> Matriz triangular superior auxiliar
%gs -> Vetor dos par�metros ortogonalizados
%theta_frols -> Vetor de par�metros identificados
%ERR -> Taxa de redu��o de erro
%indice -> Indice dos regressores selecionados do modelos
%erro_val -> Erro de valida��o calculado para verificar a converg�ncia do
%m�todo
%-------------------------------------------------------------------------%
%Dados de Entrada
%fi -> Matriz de regressores utilizada para estima��o
%Y -> Vetor contendo os dados de sa�da
%fi_val -> Matriz de regressores usados para valida��o para verificar a
%converg�ncia do mpetodo
%Y_val -> Vetor de dados de sa�da utilizados para verificar a converg�ncia
%do m�todo
%-------------------------------------------------------------------------%

function [Qs, As, gs, theta_frols, ERR, indice] = alg_FROLS(fi, Y, rho)

[~,n_teta]  = size(fi);
%n_teta = n�mero de par�metros (n�mero de colunas)

%Vari�veis auxiliares utilizadas para eliminar as colunas j� escolhidas
index = 1:n_teta;
indice_final = zeros(1,n_teta);

%Passo 1 - Fazer a matriz ortogonal igual a matriz candidata
Q = fi;
Phi = fi;
As = eye(n_teta, n_teta); %a(1,1) = 1

for m = 1:n_teta
    g(m) = (Y'*Q(:,m))/(Q(:,m)'*Q(:,m));
    err(m) = g(m).^2 .* ((Q(:,m)'*Q(:,m)) / (Y'*Y));
end

[maxERR(1), indice(1)]=max(err);

%A primeira coluna da matriz reduzida recebe a coluna de Phi com maior ERR
Qs(:,1) = Q(:,indice(1));
gs(1) = g(indice(1));
ERR(1) = err(indice(1));

indice_final(1) = index(indice(1));
index(indice(1)) = [];

%Limiar para sele��o de estrutura. (AGUIRRE, 2004) pag. 387
%(BILLINGS, 2014) pag. 75
%rho = 0.025;% Para o caso 1 -> rho = 0.002 (sem ruido) -> 0.025 (ruido gaussiano)

Phi(:,indice(1)) = []; %Retirar melhor regressor j� escolhido

% Passo 2
s = 2; %Vari�vel respons�vel pela contagem das itera��es
aux = 1; %Vari�vel que diminui o valor da dimens�o a cada itera��o

while(s <= n_teta)
    
    clear g;
    clear Q;
    clear A;
    clear err;
    
    for m = 1:n_teta - aux
        soma = 0;
        for r = 1:s-1
            soma = soma + (Phi(:,m)'*Qs(:,r))/(Qs(:,r)'*Qs(:,r)) * Qs(:,r);
        end
        Q(:,m) = Phi(:,m) - soma;
        g(m) = (Y'*Q(:,m))/(Q(:,m)'*Q(:,m));
        err(m) = g(m).^2 .* ((Q(:,m)'*Q(:,m)) / (Y'*Y));
    end
    
    [maxERR(s), indice(s)]=max(err);
    Qs(:,s) = Q(:,indice(s));
    gs(s) = g(indice(s));
    
    for r = 1:s-1
        As(r,s) = (Qs(:,r)'*Phi(:,indice(s)))/(Qs(:,r)'*Qs(:,r));
    end
    
    ERR(s) = err(indice(s));
    Phi(:,indice(s)) = [];
    
    indice_final(s) = index(indice(s));
    index(indice(s)) = [];
    
    ESR = 1 - sum(ERR);
    
    s = s + 1;
    aux = aux +1;
    
    if (ESR <= rho)
        break;
    end
    
end
clear A;
indice = indice_final(1:s-1);
% disp(indice);
% disp(maxERR(1:k-1));
A = As;
As = A(1:s-1, 1:s-1);

theta_frols = As\gs';
%disp(teta_CGSs);