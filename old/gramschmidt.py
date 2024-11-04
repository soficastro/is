import itertools
import numpy as np

####


def melhor_ordem(candidatos, M, input, output, grau, n_theta):

    #########################################################
    ## Entrada:
    ##     candidatos - matriz com regressores candidatos
    ##     M - número de regressores candidatos
    ##     input - vetor de entrada do sistema
    ##     output - vetor de saída do sistema
    ##     grau - max(nu, ny)
    ##     n_theta - quantidade desejada de regressores
    ##
    ## Saída:
    ##     h - índice que indica a ordem encontrada dos regressores
    ##     ERR_total - ERR de cada regressor escolhido

    n = len(input)

    #################################################################
    ##                                                             ##
    ## Gram-Schmidt conforme descrito no artigo do Aguirre de 1998 ##
    ##                                                             ##
    #################################################################
    ## Uma matriz P é decomposta é duas submatrizes: P = QA        ##
    #################################################################

    q_s = np.zeros((n-grau,M))
    g_s_hat = np.zeros(M)
    ERR = []
    ERR_total = np.zeros(n_theta)
    h = []
    Q = []

    ### Ordenar os regressores do melhor ao pior
    for k in range(n_theta):

        ## A primeira iteração difere das iterações seguintes
        if k == 0:
            for i in range(M):

                #Tome o i-ésimo regressor original para compor o 1º regressor ortogonal
                q_s[:,i]=candidatos[:, i]

                #Estime por mínimos quadrados o respectivo coeficiente
                g_s_hat[i] = q_s[:, i].dot(output[:-grau]) / q_s[:, i].dot(q_s[:, i])

                #Determine o ERR de cada possível regressor, candidato ao 1º regressor ortogonal
                ERR.append(g_s_hat[i]**2*q_s[:,i].dot(q_s[:,i])/output[:-grau].dot(output[: -grau]))

            #Escolha para ser o 1º regressor ortogonal, aquele com maior ERR
            h.append(np.argmax(ERR))
            #Este é o 1º regressor ortogonal, que é a primeira coluna de Q
            Q.append(candidatos[:, h[0]])

            ERR_total[k] = ERR[h[0]]

        ## Para k = 2, ..., n_theta
        else:
            ERR = [] #lista
            
            ## Para i = 1, ..., M, i =/= h_1, ..., i =/= h_k-1
            ## Ou seja, até completar o número de termos desejado no modelo, para todos os regressores que ainda não foram escolhidos
            for i in range(M):
                
                if i not in h:       
                    alpha = np.zeros(k)
                    q_k = np.zeros(k)
                    somatorio = 0
                    for j in range(k):

                        #Estime por mínimos quadrados os coeficientes dos regressores ortogonais escolhidos até a presente iteração k
                        alpha[j] = Q[j].dot(candidatos[:,i])/Q[j].dot(Q[j])

                        somatorio = somatorio + alpha[j]*Q[j]
                    
                    #Determine o próximo regressor ortogonal (candidato) eliminando de um regressor original o efeito dos k-1 regressores
                    #ortogonais escolhidos até o presente
                    q_k = candidatos[:,i] - somatorio

                    #Calcule o coeficiente do regressor ortogonal determinado no passo anterior
                    g_k_hat = q_k.dot(output[:-grau])/q_k.dot(q_k)

                    #Determine o valor do respectivo ERR 
                    ERR.append(g_k_hat**2*q_k.dot(q_k)/output[:-grau].dot(output[:-grau]))          
                                  
                else:
                    ERR.append(0) #Preciso zerar o que eu escolher para não escolher de novo
                

            somatorio2 = 0
            for j in range(k):
                somatorio2 = somatorio2 + Q[j].dot(candidatos[:, k]) / Q[j].dot(Q[j]) #Isso é alpha


            #Escolha para ser o k-ésimo regressor ortogonal aquele regressor, entre os restantes, com maior ERR
            h.append(np.argmax(ERR))
   

            #Esse é o k-ésimo regressor ortogonal regressor ortogonal, que é a k-ésima coluna de Q
            Q.append(candidatos[:, h[k]] - somatorio2)

            ERR_total[k] = ERR[h[k]]

    #h indica os regressores, ou seja, quais as colunas de P (candidatos) que devem ser incluídas nos modelos.
    return h, ERR_total


def matriz_candidatos(input, output, nu, ny, l):
    n = len(input)
    grau = max(nu, ny)
    linear_psi = np.zeros((n-grau, nu+ny))

    for i in range(ny):
        linear_psi[:, ny-1-i] = output[i+1:n-grau+i+1]

    for i in range(nu):
        linear_psi[:, ny+nu-1-i] = input[i+1:n-grau+i+1]


    array = np.zeros(nu+ny)
    for i in range(nu+ny):
        array[i]=i
    array = array.astype(int)    

    
    combinations = find_combinations(array, l)
    
    M = len(combinations)

    candidatos1 = []
    i = 0
    for comb in combinations:
        m = linear_psi[:, comb]
        multiplica = np.prod(m, axis=1)
        candidatos1.append(multiplica)
        i = i+1

    candidatos1 = np.array(candidatos1)
    candidatos = candidatos1.T

    return candidatos, M, combinations

def matriz_candidatos_narmax(input, output, nu, ny, ne, l):
    n = len(input)
    grau = max(nu, ny, ne)
    linear_psi = np.zeros((n-grau, nu+ny+ne))
    erro = np.random.default_rng().random(n)

    for i in range(ny):
        linear_psi[:, ny-1-i] = output[i+1:n-grau+i+1]

    for i in range(nu):
        linear_psi[:, ny+nu-1-i] = input[i+1:n-grau+i+1]

    for i in range(ne):
        linear_psi[:, ny+nu+ne-1-i] = erro[i+1:n-grau+i+1]


    array = np.zeros(nu+ny+ne)
    for i in range(nu+ny+ne):
        array[i]=i
    array = array.astype(int)    

    
    combinations = find_combinations(array, l)
    
    M = len(combinations)

    candidatos1 = []
    i = 0
    for comb in combinations:
        m = linear_psi[:, comb]
        multiplica = np.prod(m, axis=1)
        candidatos1.append(multiplica)
        i = i+1

    candidatos1 = np.array(candidatos1)
    candidatos = candidatos1.T

    return candidatos, M, combinations

def find_combinations(arr, l):
    combinations = []
    n = len(arr)

    for r in range(1, l + 1):
        product = itertools.combinations_with_replacement(arr, r)
        combinations.extend(list(product))

    return combinations





