% Algoritmo SEMP para detecção de estrutura e estimação de parâmetros de
% modelos NARMAX de sistemas dinâmicos não-lineares
% (baseado em Piroddi (2008))
% Autor: Ícaro Bezerra Queiroz de Araújo
% Universidade Federal do Rio Grande do Norte
% Programa de Pós-Graduação em Engenharia Elétrica e da Computação
% Data: 26/01/2018
% Alterado: 15/12/2018

function [parametro, indice_final, SRR_final] = SEMP(fi, y, u, limiar, ny, nu, ne, L, qs, qe)
%Iniciliazação das variáveis

% N = Número de observações
% n_teta = número de parâmetros totais
[N,n_teta]  = size(fi);
J_old = 100; % Valor de custo inicial para Pi_in = []
Pi_in = []; % conjunto de regressores incluídos no modelo final;
Pi_out = fi; % conjunto de regressores excluidos do modelo final;
indice_final = []; %Indices finais escolhidos
y_var = (y'*y)/N; % Variância do sinal de saída
indice = 1:n_teta;
% indice_final = zeros(1,n_teta);
aux_poda = 0;

for i = 1:n_teta
    ii = i - aux_poda;
    %-------------------------Forward Regression--------------------------%
    [~, n_teta_atual] = size(Pi_out);
    parfor regressor = 1:n_teta_atual
        Pi = [Pi_in Pi_out(:,regressor)];
        thetaRegressor(:,regressor) = (Pi'*Pi\Pi'*y);
        %thetaRegressor(:,regressor) = LMS_icaro(y, Pi, tamFR, 0.5) ;
        %[custo(regressor), y_est(:,regressor)] = MSPE(y, Pi, thetaRegressor(:, regressor));
        %[fi_novo, ~] = montaRegressores(y_est(:,regressor)', u, ny, nu, ne, L, N, qs, qe);
        %y_sim = fi_novo(:, [indice_final indice(regressor)])*thetaRegressor(:,regressor);
        y_sim = montaModelo2(y', u, [indice_final indice(regressor)], thetaRegressor(:,regressor), ny, nu, L, qs, qe, ny+10);
        custo_sim(regressor) = sum((y-y_sim').^2)/N;
        %custo_sim(regressor) = custo(regressor);
        SRR(regressor) = (J_old - custo_sim(regressor))/y_var;
    end
    
    [~,I] = max(SRR);
    J = custo_sim(I);
    
    if (J < J_old && abs(J_old - J) > limiar)
        Pi_in(:,ii) = Pi_out(:,I);
        Pi_out(:,I) = [];
        indice_final(ii) = indice(I);
        SRR_final(ii) = SRR(I);
        indice(I) = [];
        parametro = thetaRegressor(:,I);
        
        J_old = J;
        clear custo
        clear custo_sim
        clear SRR
        clear thetaRegressor
        clear y_est
    else
        break;
    end
    %-----------------------------Pruning---------------------------------%
    
    if length(indice_final) > 1
        clear thetaRegressorPoda;
        clear custoPoda;
        for j = 1:length(indice_final)
            if (j > length(indice_final))
                break;
            end
            indPruning = 1:length(indice_final);
            indPruning(j) = [];
            thetaRegressorPoda = Pi_in(:,indPruning)'*Pi_in(:,indPruning)\Pi_in(:,indPruning)'*y;
            [~, y_est_poda(:,j)] = MSPE(y, Pi_in(:,indPruning), thetaRegressorPoda);
            %[fi_poda, ~] = montaRegressores(y_est_poda(:,j)', u, ny, nu, ne, L, N, qs, qe);
            %y_sim_poda = fi_poda(:, indPruning)*thetaRegressorPoda;
            y_sim_poda = montaModelo2(y', u, indPruning, thetaRegressorPoda, ny, nu, L, qs, qe, ny+10);
            custo_sim_poda = sum((y-y_sim_poda).^2)/N;
            SRR = (J_old - custo_sim_poda)/y_var;
            
            if custo_sim_poda < J_old
                Pi_in(:,j) = [];
                indice_final(j) = [];
                parametro = thetaRegressorPoda;
                aux_poda = aux_poda+1;
            end
        end
    end
end