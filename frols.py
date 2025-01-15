import numpy as np
import scipy

def frols(candidatos, M, output, grau):
    """  
    Entrada:
        candidatos - matriz (D) com regressores candidatos (p)
        M - número de regressores candidatos
        output - vetor y(t) de saída do sistema
        grau - max(nu, ny)
        
    Saída:
        h - índice que indica a ordem encontrada dos regressores
        ERR_total - ERR de cada regressor escolhido
        theta - vetor com thetas escolhidos
    """
    N = len(output)               # Tamanho do vetor y(t)
    M_0 = M                       # Número de regressores selecionados
    rho = 0.0001                  

    l = []                        # Lista de índices (colunas de P) que foram selecionados
    ERR_total = np.zeros(M)       # Taxa de Redução de Erro de cada passo
    Q = []                        # Matriz de regressores ortogonais
    g_final = []                  # Parâmetros dos regressores ortogonais
    a = np.zeros((M, M))          
              
    for m in range(M):

        ## A primeira iteração difere das iterações seguintes (s)
        if m == 0:

            g_s = np.zeros(M)             
            q_s = np.zeros((N - grau, M)) 
            ERR_s = []

            for i in range(M):

                #Tome o i-ésimo regressor original para compor o 1º regressor ortogonal
                q_s[:, i] = candidatos[:, i]

                #Estime por mínimos quadrados o respectivo coeficiente
                g_s[i] = q_s[:, i].dot(output[grau:]) / q_s[:, i].dot(q_s[:, i])

                #Determine o ERR de cada possível regressor, candidato ao 1º regressor ortogonal
                ERR_s.append(g_s[i] ** 2 * q_s[:, i].dot(q_s[:, i]) / output[grau:].dot(output[grau:]))

            #Escolha para ser o 1º regressor ortogonal, aquele com maior ERR
            l.append(np.argmax(ERR_s))
            #Este é o 1º regressor ortogonal, que é a primeira coluna de Q
            Q.append(candidatos[:, l[0]])
            #Guarde o g do 1º regressor
            g_final.append(g_s[l[0]])
            #Guarde o ERR do 1º regressor
            ERR_total[0] = ERR_s[l[0]]
            #Construa a matriz A
            a[0,0] = 1

        ## Para m = 2, ..., M0 (k)
        else:
            ERR_k = []
            g_k = []
            q_k = []
            ## Para i = 1, ..., M, i =/= l_1, ..., i =/= l_m-1
            ## Ou seja, até chegar no critério de parada, para todos os regressores que ainda não foram escolhidos
            for i in range(M):

                if i not in l:
                    alpha = np.zeros(m)
                    
                    somatorio = 0
                    for j in range(m):

                        #Estime por mínimos quadrados os coeficientes dos regressores ortogonais escolhidos até a presente iteração m
                        alpha[j] = Q[:,j].dot(candidatos[:, i]) / Q[:,j].dot(Q[:,j])

                        somatorio = somatorio + alpha[j] * Q[:,j]

                    #Determine o próximo regressor ortogonal (candidato) eliminando de um regressor original o efeito dos m-1 regressores
                    #ortogonais escolhidos até o presente
                    q_k.append(candidatos[:, i] - somatorio)
                    
                    #Calcule o coeficiente do regressor ortogonal determinado no passo anterior
                    g_k.append(np.dot(q_k[i], output[grau:]) / np.dot(q_k[i], q_k[i]))
                    
                    #Determine o valor do respectivo ERR 
                    ERR_k.append(g_k[i] ** 2 * q_k[-1].dot(q_k[i]) / output[grau:].dot(output[grau:]))
                else:
                    ERR_k.append(0)  #Preciso zerar o que eu escolher para completar o número m
                    q_k.append(candidatos[:, i] * 0)
                    g_k.append(0)


            #Escolha para ser o m-ésimo regressor ortogonal aquele regressor, entre os restantes, com maior ERR
            l.append(np.argmax(ERR_k))

            #Esse é o m-ésimo regressor ortogonal regressor ortogonal, que é a m-ésima coluna de Q
            Q.append(q_k[l[m]])
            
            #Guarde o g do m-ésimo regressor
            g_final.append(g_k[l[m]])

            #Guarde o ERR do m-ésimo regressor
            ERR_total[m] = ERR_k[l[m]]

            #Construa a matriz A
            for r in range(m):
                a[r,m] = np.dot(Q[:,r], candidatos[:, l[m]]) / np.dot(Q[:,r], Q[:,r])
            a[m,m] = 1
                

            esr = 1 - sum(ERR_total)
            if esr < rho:
                M_0 = m + 1
                break
    
  
    a = a[:M_0,:M_0]
    
    theta = scipy.linalg.solve_triangular(a,g_final,unit_diagonal=True)

    return l, ERR_total, theta


