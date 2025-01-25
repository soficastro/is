function D = delay_matrix(u, y, nu, ny)
    % Função para construir uma matriz apenas com os sinais atrasados
    % Sem incluir os sinais originais (não atrasados).
    %
    % Parâmetros:
    %   u  - Sinal de entrada (vetor coluna)
    %   y  - Sinal de saída (vetor coluna)
    %   nu - Máximo atraso do sinal de entrada
    %   ny - Máximo atraso do sinal de saída
    %
    % Retorno:
    %   D  - Matriz com os sinais atrasados
    
    % Verificar se os sinais são vetores coluna
    if ~iscolumn(u)
        u = u(:); % Transformar em vetor coluna, se necessário
    end
    if ~iscolumn(y)
        y = y(:); % Transformar em vetor coluna, se necessário
    end
    
    % Número de amostras nos sinais
    N = length(u);
    
    % Determinar o número total de linhas úteis após considerar os atrasos
    maxDelay = max(nu, ny); % Maior atraso entre entrada e saída
    validLength = N - maxDelay; % Número de linhas válidas após os atrasos
    
    % Inicializar a matriz de atrasos
    D = [];
    
    % Construir as colunas para os atrasos de y (começando de atraso 1)
    for i = 1:ny
        D = [D, y(maxDelay + 1 - i:N - i)];
    end
    
    % Construir as colunas para os atrasos de u (começando de atraso 1)
    for i = 1:nu
        D = [D, u(maxDelay + 1 - i:N - i)];
    end
end
