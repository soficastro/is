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

    Phi = delay_matrix([1;2;3;4;5], [6;7;8;9;10], 1, 2);

    disp(Phi);

end

fprintf('Todos os arquivos esperados foram processados.\n');
