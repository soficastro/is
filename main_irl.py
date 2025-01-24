import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
from frols import frols
from utils import matriz_candidatos

print('--')

data_folder = 'is\data'

for filename in os.listdir(data_folder):
    system = os.path.join(data_folder, filename)
    if os.path.isfile(system):
        
        if "ballbeam" in system:
            continue
            data = pd.read_csv(system, delimiter='\t', header = None)
            system = 'Ball-beam'
            u = data[data.columns[0]].to_numpy()
            y = data[data.columns[1]].to_numpy()
            ### Hiperparâmetros
            nu = 3 #atraso da entrada
            ny = 3 #atraso da saída
            l = 3 #grau de nao linearidade

        elif "dataBenchmark" in system:
            continue
            data = scipy.io.loadmat(system)
            system = 'Tanks'
            u_train = np.reshape(data['uEst'],1024)
            y_train = np.reshape(data['yEst'],1024)
            u_test = np.reshape(data['uVal'],1024)
            y_test = np.reshape(data['yVal'],1024)
            ### Hiperparâmetros
            nu = 4 #atraso da entrada
            ny = 2 #atraso da saída
            l = 3  #grau de não linearidade
        
        elif "exchanger" in system:
            data = pd.read_csv(system, delimiter='\t', header = None)
            system = 'Heat Exchanger'
            time_steps = data[data.columns[0]].to_numpy()
            u = data[data.columns[1]].to_numpy()
            y = data[data.columns[2]].to_numpy()
            ### Hiperparâmetros
            nu = 4 #atraso da entrada
            ny = 2 #atraso da saída
            l = 3  #grau de nao linearidade

        elif "cassini" in system:
            data = scipy.io.loadmat(system)
            system = 'Soldering Iron'
            u = np.reshape(data['u'],2520)
            y = np.reshape(data['y'],2520)
            ### Hiperparâmetros
            nu = 2 #atraso da entrada
            ny = 2 #atraso da saída
            l = 2  #grau de nao linearidade


        print("System:", system)
  
        if system != 'Tanks':
            ### Treino em 75% dos dados
            n_train = int(0.75 * u.shape[0])
            u_train, u_test = u[:n_train], u[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]



        ### TREINO ###

        max_delay = max(nu, ny)
        rho = 0.0001

        candidatos, regressor_names = matriz_candidatos(input = u_train, output = y_train, nu = nu, ny = ny, l = l)
        idx_order, theta = frols(candidatos = candidatos, y = y_train, max_delay = max_delay, rho = rho)
        
        ### MODELO ###
        
        n_theta = len(theta)

        chosen_regressors = []
        for i in range(n_theta):
            chosen_regressors.append(regressor_names[idx_order[i]])

        terms = [f"{theta[i]} * {chosen_regressors[i]}" for i in range(n_theta)]
        model = "y(k) = " + " + ".join(terms)
        print(model)

        ### SIMULAÇÃO UM PASSO À FRENTE ###

        candidatos, regressor_names = matriz_candidatos(input = u_test, output = y_test, nu = nu, ny = ny, l = l)

        n_test = len(u_test)

        Psi_test = np.zeros((n_test - max_delay, n_theta))
        for i in range(n_theta):
            Psi_test[:,i] = candidatos[:,idx_order[i]]
        y_hat_one_step = Psi_test @ theta
        
        ### SIMULAÇÃO LIVRE ###

        candidatos, regressor_names = matriz_candidatos(input = u_test, output = y_test, nu = nu, ny = ny, l = l)

        y_hat = np.zeros(n_test)

        y_hat[:max_delay] = y_test[:max_delay]

        Psi_full = candidatos[0]

        Psi_frols = np.zeros(n_theta)

        for i in range(n_theta):
            Psi_frols[i] = Psi_full[idx_order[i]]
                
        y_hat[max_delay] = Psi_frols.dot(theta)

        for k in range(max_delay + 1, n_test):
            
            candidatos, regressor_names = matriz_candidatos(input = u_test[:k+1], output = y_hat[:k+1], nu = nu, ny = ny, l = l)

            Psi_full = candidatos[-1]

            for i in range(n_theta):
                Psi_frols[i] = Psi_full[idx_order[i]]

            y_hat[k] = Psi_frols.dot(theta)


        ### PLOT ###

        plt.figure(figsize=(10, 6))
        plt.plot(y_test[max_delay:], label="Saída esperada", color="b")
        plt.plot(y_hat[max_delay:], label="Saída simulação livre", color="r")
        plt.plot(y_hat_one_step, label="Saída um passo à frente", color="g", linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title(f"{system}")
        plt.legend()
        plt.grid(True)
        plt.show()