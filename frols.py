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

    g_s = np.zeros(M)
    q_s = np.zeros((N - grau, M)) # Matriz de regressores ortogonais 
    ERR_s = []                    # Taxa de Redução de Erro
    ERR_total = np.zeros(M)
    h = []
    Q = []
    g_final = []
    rho = 0.00001
    a = np.zeros((M, M))

    for k in range(M):
        
        ## A primeira iteração difere das iterações seguintes (s)
        if k == 0:
            for i in range(M):

                #Tome o i-ésimo regressor original para compor o 1º regressor ortogonal
                q_s[:, i] = candidatos[:, i]

                #Estime por mínimos quadrados o respectivo coeficiente
                g_s[i] = q_s[:, i].dot(output[:-grau]) / q_s[:, i].dot(q_s[:, i])

                #Determine o ERR de cada possível regressor, candidato ao 1º regressor ortogonal
                ERR_s.append(g_s[i] ** 2 * q_s[:, i].dot(q_s[:, i]) / output[: -grau].dot(output[: -grau]))

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
            g_k = []
            q_k = []
            ## Para i = 1, ..., M, i =/= h_1, ..., i =/= h_k-1
            ## Ou seja, até chegar no critério de parada, para todos os regressores que ainda não foram escolhidos
            for i in range(M):

                if i not in h:
                    alpha = np.zeros(k)
                    
                    somatorio = 0
                    for j in range(k):

                        #Estime por mínimos quadrados os coeficientes dos regressores ortogonais escolhidos até a presente iteração k
                        alpha[j] = Q[j].dot(candidatos[:, i]) / Q[j].dot(Q[j])

                        somatorio = somatorio + alpha[j] * Q[j]

                    #Determine o próximo regressor ortogonal (candidato) eliminando de um regressor original o efeito dos k-1 regressores
                    #ortogonais escolhidos até o presente
                    q_k.append(candidatos[:, i] - somatorio)
                    
                    #Calcule o coeficiente do regressor ortogonal determinado no passo anterior
                    g_k.append(np.dot(q_k[i], output[: -grau]) / np.dot(q_k[i], q_k[i]))
                    
                    #Determine o valor do respectivo ERR 
                    ERR_k.append(g_k[i] ** 2 * q_k[i].dot(q_k[i]) / output[: -grau].dot(output[: -grau]))
                else:
                    ERR_k.append(0)  #Preciso zerar o que eu escolher para completar o número k
                    q_k.append(0)
                    g_k.append(0)


            #Escolha para ser o k-ésimo regressor ortogonal aquele regressor, entre os restantes, com maior ERR
            h.append(np.argmax(ERR_k))

            #Esse é o k-ésimo regressor ortogonal regressor ortogonal, que é a k-ésima coluna de Q
            Q.append(q_k[h[k]])
            #Q.append(q_k[k]) # Antigo 
            
            #Guarde o g do k-ésimo regressor
            g_final.append(g_k[h[k]])

            #Guarde o ERR do k-ésimo regressor
            ERR_total[k] = ERR_k[h[k]]

            #Construa a matriz A
            for j in range(k):
                a[j,k] = np.dot(Q[j], candidatos[:, h[k]]) / np.dot(Q[j], Q[j])
            a[k,k] = 1
                

            esr = 1 - sum(ERR_total)
            if esr < rho:
                M_0 = k + 1
                break
    
    a = a[:M_0,:M_0]
    
    theta = g_final @ np.linalg.inv(a)


    return h, ERR_total, theta


