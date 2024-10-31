import numpy as np

def frols(candidatos, M, output, grau):
    """
    Implementação do algoritmo FROLS.
    
    Entrada:
        candidatos - matriz (D) com regressores candidatos (p)
        M - número de regressores candidatos
        output - vetor y(t) de saída do sistema
        grau - max(nu, ny)
        
    Saída:
        h - índice que indica a ordem encontrada dos regressores
        ERR_total - ERR de cada regressor escolhido
    """
    N = len(output)
    M_0 = M
    
    sigma = np.dot(output[:-grau], output[:-grau])

    g_s = np.zeros(M)
    g_k = np.zeros(M)
    ERR_total = np.zeros(M)
    h = []
    Q = []
    g_final = []
    rho = 0.00001
    a = np.zeros((M, M))


    


    for k in range(M):
        
        ## A primeira iteração difere das iterações seguintes (s)
        if k == 0:

            q_s = np.zeros((N - grau, M)) # Matriz de regressores ortogonais 
            ERR_s = []                    # Taxa de Redução de Erro

            for i in range(M):

                #Tome o i-ésimo regressor original para compor o 1º regressor ortogonal
                q_s[:, i] = candidatos[:, i]

                #Estime por mínimos quadrados o respectivo coeficiente
                g_s[i] = np.dot(q_s[:, i], output[:-grau]) / np.dot(q_s[:, i], q_s[:, i])

                #Determine o ERR de cada possível regressor, candidato ao 1º regressor ortogonal
                ERR_s.append(g_s[i] ** 2 * np.dot(q_s[:, i], q_s[:, i]) / sigma)

            #Escolha para ser o 1º regressor ortogonal, aquele com maior ERR
            h.append(np.argmax(ERR_s))
            #Este é o 1º regressor ortogonal, que é a primeira coluna de Q
            Q.append(candidatos[:, h[0]])
            #Guarde o g do 1º regressor
            g_final.append(g_s[h[0]])
            #Guarde o ERR do 1º regressor
            ERR_total[0] = ERR_s[h[0]]
            #Construa a matriz A
            a[0,0] = 1



        ## Para k = 2, ..., M0 (k)
        else:

            ERR_k = []
            ## Para i = 1, ..., M, i =/= h_1, ..., i =/= h_k-1
            ## Ou seja, até chegar no critério de parada, para todos os regressores que ainda não foram escolhidos
            for i in range(M):

                if i not in h:
                    
                    #Estime por mínimos quadrados os coeficientes (que são os elementos da matriz A) dos regressores
                    #ortogonais escolhidos até a presente iteração (k)
                    alpha_sum = sum((np.dot(Q[j], candidatos[:, i]) / np.dot(Q[j], Q[j])) * Q[j] for j in range(k))

                    #Determine o próximo regressor ortogonal (candidato) eliminando de um regressor original o efeito dos k-1 regressores
                    #ortogonais escolhidos até o presente
                    q_k = candidatos[:,i] - alpha_sum
                    
                    #Calcule o coeficiente do regressor ortogonal determinado no passo anterior
                    g_k[i] = np.dot(q_k, output[:-grau]) / np.dot(q_k, q_k)
                    
                    #Determine o valor do respectivo ERR 
                    ERR_k.append(g_k[i] ** 2 * np.dot(q_k, q_k) / sigma)
                else:
                    ERR_k.append(0)  #Preciso zerar o que eu escolher para não escolher de novo



            #Escolha para ser o k-ésimo regressor ortogonal aquele regressor, entre os restantes, com maior ERR
            best_index = np.argmax(ERR_k)
            h.append(best_index)
            #Esse é o k-ésimo regressor ortogonal regressor ortogonal, que é a k-ésima coluna de Q
            Q.append(candidatos[:, best_index] - sum((np.dot(Q[j], candidatos[:, best_index]) / np.dot(Q[j], Q[j])) * Q[j] for j in range(k))) #Isso é alpha
            #Guarde o g do k-ésimo regressor
            g_final.append(g_s[h[0]])
            #Guarde o ERR do k-ésimo regressor
            ERR_total[k] = ERR_k[best_index]

            #Construa a matriz A
            for j in range(k):
                a[j,k] = np.dot(Q[j], candidatos[:, best_index]) / np.dot(Q[j], Q[j])
            a[k,k] = 1
                


            esr = 1 - sum(ERR_total)
            print(esr, sum(ERR_total))
            if esr < rho:
                M_0 = k
                break
    
    a = a[:M_0+1,:M_0+1]
    
    theta = g_final @ np.linalg.inv(a)

    return h, ERR_total, theta


