% main.m
% Este script visita os arquivos na pasta 'data' pelo nome e carrega seus dados.

% Caminho da pasta de dados
dataPath = 'data';

% Obter lista de arquivos na pasta
fileList = dir(fullfile(dataPath, '*.*')); % Obtém todos os arquivos
fileList = fileList(~[fileList.isdir]);   % Remove diretórios da lista

% Lista de arquivos esperados (por nome)
expectedFiles = {'ballbeam.dat', 'dataBenchmark.mat', 'exchanger.dat', ...
                 'robot_arm.dat', 'SNLS80mV.mat'};

% Iterar sobre os arquivos esperados
for i = 1:1 % length(expectedFiles)
    fileName = expectedFiles{i};
    filePath = fullfile(dataPath, fileName);

    if isfile(filePath)
        fprintf('Carregando arquivo: %s\n', fileName);

        % Obter extensão do arquivo
        [~, ~, ext] = fileparts(fileName);

        switch ext
            case '.dat'
                % Arquivos .dat - carregar como tabela ou matriz
                data = importdata(filePath);
                disp('Conteúdo do arquivo .dat:');
                disp(data);

            case '.mat'
                % Arquivos .mat - carregar variáveis
                data = load(filePath);
                disp('Variáveis no arquivo .mat:');
                disp(data);

            otherwise
                fprintf('Extensão não reconhecida (%s). Arquivo ignorado.\n', ext);
        end
    else
        fprintf('Arquivo não encontrado: %s\n', fileName);
    end

    switch fileName
        case expectedFiles{1}
            u = data(:, 1);
            y = data(:, 2);
    end

    %% main

    Phi = delay_matrix(u, y, 2, 2);
    candidatos = poly_matrix(Phi, 3);

    [~, ~, ~, theta_frols, ~, indice] = alg_FROLS(candidatos, y(3:end), 1e-3)

    X = [];
    for i = indice
        X = [X candidatos(:, i)];
    end
    y_hat = X*theta_frols;

    %% Free Sim
    N = size(y, 1);
    y_free = zeros(1, N);
    y_free(1:3) = y(1:3);
    for i = 3:N
        Phi = delay_matrix(u(i-2:i), y_free(i-2:i), 2, 2);
        Phi = poly_matrix(Phi, 3);

        X = [];
        for j = indice
            X = [X Phi(:, j)];
        end

        disp(X*theta_frols);
        y_free(i) = X*theta_frols;
    end

    plot(y, 'b-', 'LineWidth', 1.5); hold on;
    plot(y_free, 'r--', 'LineWidth', 1.5);
    legend('y (Real)', 'y\_hat (Estimated)');
    hold off;

end

fprintf('Todos os arquivos esperados foram processados.\n');
