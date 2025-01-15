import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
from frols import frols
from utils import matriz_candidatos

print('--')

data_folder = 'C:/Users/Sofia/Documents/Projeto/data'

for filename in os.listdir(data_folder):
    system = os.path.join(data_folder, filename)
    if os.path.isfile(system):
        if system == 'C:/Users/Sofia/Documents/Projeto/data\\ballbeam.dat':
            data = pd.read_csv(system, delimiter='\t', header = None)
            system = 'Ball-beam'
            u = data[data.columns[0]].to_numpy()
            y = data[data.columns[1]].to_numpy()
            ### Hiperparâmetros
            nu = 3 #atraso da entrada
            ny = 3 #atraso da saída
            l = 3  #grau de nao linearidade

        elif system == 'C:/Users/Sofia/Documents/Projeto/data\\dataBenchmark.mat':
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
        
        elif system == 'C:/Users/Sofia/Documents/Projeto/data\\exchanger.dat':
            data = pd.read_csv(system, delimiter='\t', header = None)
            system = 'Heat Exchanger'
            time_steps = data[data.columns[0]].to_numpy()
            u = data[data.columns[1]].to_numpy()
            y = data[data.columns[2]].to_numpy()
            ### Hiperparâmetros
            nu = 2 #atraso da entrada
            ny = 4 #atraso da saída
            l = 2  #grau de nao linearidade

        elif system == 'C:/Users/Sofia/Documents/Projeto/data\\robot_arm.dat':
            data = pd.read_csv(system, delimiter='\t', header = None)
            system = 'Robot Arm'
            u = data[data.columns[0]].to_numpy()
            y = data[data.columns[1]].to_numpy()
            ### Hiperparâmetros
            nu = 4 #atraso da entrada
            ny = 3 #atraso da saída
            l = 2  #grau de nao linearidade

        elif system == 'C:/Users/Sofia/Documents/Projeto/data\\SNLS80mV.mat':
            data = scipy.io.loadmat(system)
            system = 'Silverbox'
            u = np.reshape(data['V1'],131072)
            y = np.reshape(data['V2'],131072)
            ### Hiperparâmetros
            nu = 3 #atraso da entrada
            ny = 3 #atraso da saída
            l = 3  #grau de nao linearidade

        print("System:", system)
  
        if system != 'Tanks':
            ### Treino em 75% dos dados
            n_train = int(0.75 * u.shape[0])
            u_train, u_test = u[:n_train], u[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]


        ## Normalizando dados de treino e teste
        u_train = u_train/np.max(np.abs(u_train))
        y_train = y_train/np.max(np.abs(y_train))
        u_test = u_test/np.max(np.abs(u_test))
        y_test = y_test/np.max(np.abs(y_test))  


        ################
        #### Treino ####

        candidatos, M, regressor_names = matriz_candidatos(input = u_train, output = y_train, nu = nu, ny = ny, l = l)
        n = len(u_train)
        grau = max(nu, ny)
        h, ERR_total, theta = frols(candidatos = candidatos, M = M, output = y_train, grau = grau)
        n_theta = len(theta)

        chosen_regressors = []
        for i in range(n_theta):
            chosen_regressors.append(regressor_names[h[i]])


        terms = [f"{theta[i]} * {chosen_regressors[i]}" for i in range(n_theta)]
        model = "y(k) = " + " + ".join(terms)
        print(model)


        #############
        ##  Teste  ##

        n_test = len(u_test)
        y_hat_test_free = np.zeros(n_test, dtype='float64')
        y_hat_test_onestep = np.zeros(n_test, dtype='float64')
        y_hat_test_free[:grau] = y_test[:grau]
        y_hat_test_onestep[:grau] = y_test[:grau]

        


        

        




