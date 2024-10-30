import itertools
import numpy as np

def melhor_ordem(candidatos, M, input, output, grau, n_theta):
    """
    Implements the FROLS algorithm for system identification.
    
    Entrada:
        candidatos - matriz com regressores candidatos
        M - número de regressores candidatos
        input - vetor de entrada do sistema
        output - vetor de saída do sistema
        grau - max(nu, ny)
        n_theta - quantidade desejada de regressores
        
    Saída:
        h - índice que indica a ordem encontrada dos regressores
        ERR_total - ERR de cada regressor escolhido
    """
    n = len(input)
    q_s = np.zeros((n - grau, M))
    g_s_hat = np.zeros(M)
    ERR_total = np.zeros(n_theta)
    h = []
    Q = []
    
    ### Ordenar os regressores do melhor ao pior
    for k in range(n_theta):
        ERR = []

        ## A primeira iteração difere das iterações seguintes
        if k == 0:

            # Initial iteration: Find the first regressor
            for i in range(M):

                #Tome o i-ésimo regressor original para compor o 1º regressor ortogonal
                q_s[:, i] = candidatos[:, i]

                #Estime por mínimos quadrados o respectivo coeficiente
                g_s_hat[i] = np.dot(q_s[:, i], output[:-grau]) / np.dot(q_s[:, i], q_s[:, i])

                #Determine o ERR de cada possível regressor, candidato ao 1º regressor ortogonal
                ERR.append(g_s_hat[i] ** 2 * np.dot(q_s[:, i], q_s[:, i]) / np.dot(output[:-grau], output[:-grau]))

            #Escolha para ser o 1º regressor ortogonal, aquele com maior ERR
            best_index = np.argmax(ERR)
            h.append(best_index)
            Q.append(candidatos[:, h[0]])
            ERR_total[k] = ERR[best_index]


        ## Para k = 2, ..., n_theta
        else:
            # For subsequent iterations, select the next best regressor
            ERR = []

            ## Para i = 1, ..., M, i =/= h_1, ..., i =/= h_k-1
            ## Ou seja, até completar o número de termos desejado no modelo, para todos os regressores que ainda não foram escolhidos
            for i in range(M):
                if i not in h:
                    # Orthogonalize the candidate regressor against selected regressors in Q
                    q_k = candidatos[:, i]
                    for j in range(k):

                        #Estime por mínimos quadrados os coeficientes dos regressores ortogonais escolhidos até a presente iteração k
                        alpha = np.dot(Q[j], candidatos[:, i]) / np.dot(Q[j], Q[j])

                        #Determine o próximo regressor ortogonal (candidato) eliminando de um regressor original o efeito dos k-1 regressores
                        #ortogonais escolhidos até o presente
                        q_k -= alpha * Q[j]
                    
                    #Calcule o coeficiente do regressor ortogonal determinado no passo anterior
                    g_k_hat = np.dot(q_k, output[:-grau]) / np.dot(q_k, q_k)
                
                    #Determine o valor do respectivo ERR 
                    ERR.append(g_k_hat ** 2 * np.dot(q_k, q_k) / np.dot(output[:-grau], output[:-grau]))
                else:
                    ERR.append(0)  #Preciso zerar o que eu escolher para não escolher de novo

            #Escolha para ser o k-ésimo regressor ortogonal aquele regressor, entre os restantes, com maior ERR
            best_index = np.argmax(ERR)
            h.append(best_index)
            #Esse é o k-ésimo regressor ortogonal regressor ortogonal, que é a k-ésima coluna de Q
            Q.append(candidatos[:, best_index] - sum((np.dot(Q[j], candidatos[:, best_index]) / np.dot(Q[j], Q[j])) * Q[j] for j in range(k))) #Isso é alpha
            ERR_total[k] = ERR[best_index]

    return h, ERR_total

def matriz_candidatos(input, output, nu, ny, l):
    n = len(input)
    grau = max(nu, ny)
    linear_psi = np.zeros((n - grau, nu + ny))

    # Create candidate regressors using input and output delays
    for i in range(ny):
        linear_psi[:, ny - 1 - i] = output[i + 1:n - grau + i + 1]
    for i in range(nu):
        linear_psi[:, ny + nu - 1 - i] = input[i + 1:n - grau + i + 1]

    # Generate combinations of terms up to degree l
    array = np.arange(nu + ny)
    combinations = find_combinations(array, l)
    M = len(combinations)

    # Create the candidate matrix
    candidatos = np.array([np.prod(linear_psi[:, comb], axis=1) for comb in combinations]).T

    return candidatos, M, combinations


def find_combinations(arr, l):
    combinations = []
    for r in range(1, l + 1):
        combinations.extend(list(itertools.combinations_with_replacement(arr, r)))
    return combinations
