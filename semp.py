import numpy as np

def semp(candidatos, y, max_delay, rho):
    """  
    Entrada:
        candidatos - matriz com regressores candidatos (P) (N - max_delay x M)
        y - vetor y(t) de saída do sistema (N x 1)
        max_delay - max(nu, ny)
        rho - limiar
        
    Saída:
        l - índice que indica a ordem encontrada dos regressores
        theta - vetor com thetas escolhidos
    """
    
    N = len(y)                    # Tamanho do vetor y(t)
    n_theta = candidatos.shape[1] # Número de regressores candidatos

    J_old = 100                   # Valor de custo inicial, qualquer número grande

    P = np.empty((N - max_delay, 0))  # Regressores incluídos no modelo final
    Q = candidatos                # Regressores a serem analisados

    idx_final = []                # Índices (colunas de P) que foram selecionados
    indices = list(range(n_theta))# Lista com os índices de todos os regressores candidatos
    y_var = y.dot(y) / N          # Variância do sinal de saída y(t)
    SRR_final = []                # Lista com SRRs

    for i in range(n_theta):

        J = []
        SRR = []
        theta_list = []

        ### SIMULATION ###

        for j in range(Q.shape[1]):     

            P_test = np.hstack((P, Q[:, [j]]))
            theta_test = np.linalg.inv(P_test.T @ P_test) @ P_test.T @ y[max_delay:]
            y_test = P_test @ theta_test

            theta_list.append(theta_test)
            J.append(np.sum((y[max_delay:] - y_test) ** 2) / (N - max_delay))
            SRR.append((J_old - J[j]) / y_var)


        l = np.argmax(SRR) 
  

        if J[l] < J_old and abs(J_old - J[l]) > rho:
        
            P = np.hstack((P, Q[:, l].reshape(-1, 1)))
            Q = np.delete(Q, l, axis=1)

            idx_final.append(indices[l]) 
            SRR_final.append(SRR[l])
            indices.pop(l)
            J_old = J[l]                                            
            theta = theta_list[l]
        else:
            break

        ### PRUNING ###

        if P.shape[1] > 1:

            for k in range(P.shape[1]):

                idx_pruning = [m for m in range(P.shape[1]) if m != k]

                R = P[:, idx_pruning]
                theta_R = np.linalg.inv(R.T @ R) @ R.T @ y[max_delay:]
                y_R = R @ theta_R
                J_p = np.sum((y[max_delay:] - y_R) ** 2) / (N - max_delay)

                if J_p < J_old:

                    P = np.delete(P, k, axis = 1)
                    idx_final.pop(k)
                    theta = theta_R
                    J_old = J_p
                    break

    return idx_final, theta
