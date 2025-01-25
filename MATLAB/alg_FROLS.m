%Algoritmo de detecção de estrutura Forward Regressive Orthogonal
%Least Squares (FROLS)
%Livro do BILLINGS (pag 74).
%Autor: Ícaro Bezerra Queiroz de Araújo
%Universidade Federal do Rio Grande do Norte
%Data da última modificação: 16/07/2018
%-------------------------------------------------------------------------%
%Dados de saída
%Qs -> Matriz dos regressores ortogonalizada
%As -> Matriz triangular superior auxiliar
%gs -> Vetor dos parâmetros ortogonalizados
%theta_frols -> Vetor de parâmetros identificados
%ERR -> Taxa de redução de erro
%indice -> Indice dos regressores selecionados do modelos
%erro_val -> Erro de validação calculado para verificar a convergência do
%método
%-------------------------------------------------------------------------%
%Dados de Entrada
%fi -> Matriz de regressores utilizada para estimação
%Y -> Vetor contendo os dados de saída
%fi_val -> Matriz de regressores usados para validação para verificar a
%convergência do mpetodo
%Y_val -> Vetor de dados de saída utilizados para verificar a convergência
%do método
%-------------------------------------------------------------------------%

function [Qs, As, gs, theta_frols, ERR, indice] = alg_FROLS(fi, Y, rho)

[~,n_teta]  = size(fi);
%n_teta = número de parâmetros (número de colunas)

%Variáveis auxiliares utilizadas para eliminar as colunas já escolhidas
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

%Limiar para seleção de estrutura. (AGUIRRE, 2004) pag. 387
%(BILLINGS, 2014) pag. 75
%rho = 0.025;% Para o caso 1 -> rho = 0.002 (sem ruido) -> 0.025 (ruido gaussiano)

Phi(:,indice(1)) = []; %Retirar melhor regressor já escolhido

% Passo 2
s = 2; %Variável responsável pela contagem das iterações
aux = 1; %Variável que diminui o valor da dimensão a cada iteração

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