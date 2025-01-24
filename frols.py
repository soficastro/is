import numpy as np
import scipy

def frols(candidatos, y, max_delay, rho):
    """  
    Entrada:
        candidatos - matriz (D) com regressores candidatos (p)
        y - vetor y(t) de saída do sistema
        max_delay - max(nu, ny)
        
    Saída:
        l - índice que indica a ordem encontrada dos regressores
        theta - vetor com thetas escolhidos
        rho - limiar
    """

    N = len(y)                    # Tamanho do vetor y(t)
    M = candidatos.shape[1]       # Número de regressores candidatos
    M_0 = M                       # Inicializando número de regressores selecionados          

    l = []                        # Índices (colunas de P) que foram selecionados
    ERR_total = np.zeros(M)       # Taxa de Redução de Erro dos regressores
    g = []                        # Parâmetros dos regressores ortogonais
    A = np.zeros((M, M))          # Inicializando matriz auxiliar A

    sigma = y[max_delay:].dot(y[max_delay:])
              
    for s in range(M):

        ## A primeira iteração difere das iterações seguintes (1)
        if s == 0:

            g_1 = np.zeros(M)             
            q_1 = np.zeros((N - max_delay, M)) 
            ERR_1 = []

            for m in range(M):

                #Tome o m-ésimo regressor original para compor o 1º regressor ortogonal
                q_1[:, m] = candidatos[:, m]

                #Estime por mínimos quadrados o respectivo coeficiente
                g_1[m] = q_1[:, m].dot(y[max_delay:]) / q_1[:, m].dot(q_1[:, m])

                #Determine o ERR de cada possível regressor, candidato ao 1º regressor ortogonal
                ERR_1.append(g_1[m] ** 2 * q_1[:, m].dot(q_1[:, m]) / sigma)

            #Escolha para ser o 1º regressor ortogonal, aquele com maior ERR
            l.append(np.argmax(ERR_1))
            #Este é o 1º regressor ortogonal, que é a primeira coluna de Q
            Q = candidatos[:, l[0]].reshape(-1, 1)
            #Guarde o g do 1º regressor
            g.append(g_1[l[0]])
            #Guarde o ERR do 1º regressor
            ERR_total[0] = ERR_1[l[0]]
            #Construa a matriz A
            A[0,0] = 1

        ## Para s = 2, ..., M0 (s)
        else:
            ERR_s = []
            g_s = []
            q_s = []
            ## Para m = 1, ..., M, m =/= l_1, ..., m =/= l_m-1
            ## Ou seja, até chegar no critério de parada, para todos os regressores que ainda não foram escolhidos
            for m in range(M):

                if m not in l:
                    alpha = np.zeros(s)
                    
                    somatorio = 0
                    for r in range(s):

                        #Estime por mínimos quadrados os coeficientes dos regressores ortogonais escolhidos até a presente iteração m
                        alpha[r] = Q[:,r].dot(candidatos[:, m]) / Q[:,r].dot(Q[:,r])

                        somatorio = somatorio + alpha[r] * Q[:,r]

                    #Determine o próximo regressor ortogonal (candidato) eliminando de um regressor original o efeito dos s-1 regressores
                    #ortogonais escolhidos até o presente
                    q_s.append(candidatos[:, m] - somatorio)
                    
                    #Calcule o coeficiente do regressor ortogonal determinado no passo anterior
                    g_s.append(np.dot(q_s[m], y[max_delay:]) / np.dot(q_s[m], q_s[m]))
                    
                    #Determine o valor do respectivo ERR 
                    ERR_s.append(g_s[m] ** 2 * q_s[-1].dot(q_s[m]) / sigma)
                else:
                    ERR_s.append(0)  #Preciso zerar o que eu escolher para completar o número s
                    q_s.append(candidatos[:, m] * 0)
                    g_s.append(0)


            #Escolha para ser o s-ésimo regressor ortogonal aquele regressor, entre os restantes, com maior ERR
            l.append(np.argmax(ERR_s))

            #Esse é o s-ésimo regressor ortogonal regressor ortogonal, que é a s-ésima coluna de Q
            # Q.append(q_k[l[s]])
            Q = np.column_stack((Q, q_s[l[s]]))
            
            #Guarde o g do s-ésimo regressor
            g.append(g_s[l[s]])

            #Guarde o ERR do s-ésimo regressor
            ERR_total[s] = ERR_s[l[s]]

            #Construa a matriz A
            for r in range(s):
                A[r,s] = np.dot(Q[:,r], candidatos[:, l[s]]) / np.dot(Q[:,r], Q[:,r])
            A[s,s] = 1
                

            esr = 1 - sum(ERR_total)
            if esr < rho:
                M_0 = s + 1
                break
    
  
    A = A[:M_0, :M_0]
    
    theta = scipy.linalg.solve_triangular(A, g, unit_diagonal = True)

    return l, theta


