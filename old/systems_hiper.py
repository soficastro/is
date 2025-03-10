import pandas as pd
import numpy as np
import scipy.io
import os
import re
from frols import frols
from utils import matriz_candidatos

print('--')

df_list = []

data_folder = 'C:/Users/Sofia/Documents/Projeto/data'
for filename in os.listdir(data_folder):
    system = os.path.join(data_folder, filename)
    if os.path.isfile(system):
        if system == 'C:/Users/Sofia/Documents/Projeto/data\\exchanger.dat':
            data = pd.read_csv(system, delimiter='\t', header = None)
            system = 'Heat Exchanger'
            time_steps = data[data.columns[0]].to_numpy()
            u = data[data.columns[1]].to_numpy()
            y = data[data.columns[2]].to_numpy()
            ### Hiperparâmetros
            nu = 2 #atraso da entrada
            ny = 1 #atraso da saída
            l = 3  #grau de nao linearidade

        elif system == 'C:/Users/Sofia/Documents/Projeto/data\\ballbeam.dat':
            data = pd.read_csv(system, delimiter='\t', header = None)
            system = 'Ball-beam'
            u = data[data.columns[0]].to_numpy()
            y = data[data.columns[1]].to_numpy()
            ### Hiperparâmetros
            nu = 1 #atraso da entrada
            ny = 1 #atraso da saída
            l = 1  #grau de nao linearidade

        elif system == 'C:/Users/Sofia/Documents/Projeto/data\\robot_arm.dat':
            data = pd.read_csv(system, delimiter='\t', header = None)
            system = 'Robot Arm'
            u = data[data.columns[0]].to_numpy()
            y = data[data.columns[1]].to_numpy()
            ### Hiperparâmetros
            nu = 2 #atraso da entrada
            ny = 1 #atraso da saída
            l = 3  #grau de nao linearidade

        elif system == 'C:/Users/Sofia/Documents/Projeto/data\\SNLS80mV.mat':
            data = scipy.io.loadmat(system)
            system = 'Silverbox'
            u = np.reshape(data['V1'],131072)
            y = np.reshape(data['V2'],131072)
            ### Hiperparâmetros
            nu = 2 #atraso da entrada
            ny = 1 #atraso da saída
            l = 2  #grau de nao linearidade

        elif system == 'C:/Users/Sofia/Documents/Projeto/data\\dataBenchmark.mat':
            data = scipy.io.loadmat(system)
            system = 'Tanks'
            u_train = np.reshape(data['uEst'],1024)
            y_train = np.reshape(data['yEst'],1024)
            u_test = np.reshape(data['uVal'],1024)
            y_test = np.reshape(data['yVal'],1024)
            
            ### Hiperparâmetros
            nu = 2 #atraso da entrada
            ny = 1 #atraso da saída
            l = 2  #grau de nao linearidade


        print("System:", system)

        
        if system != 'Tanks':
            ### Treino em 75% dos dados
            n_train = int(0.75 * u.shape[0])
            u_train, u_test = u[:n_train], u[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]

        ### Normalizando dados de treino e teste
        u_train = u_train/np.max(np.abs(u_train))
        y_train = y_train/np.max(np.abs(y_train))
        u_test = u_test/np.max(np.abs(u_test))
        y_test = y_test/np.max(np.abs(y_test))
        


        ################
        #### Treino ####

        nu_range = range(1,5)
        ny_range = range(1,5)
        l_range = range(2,5)

        import itertools
        for nu, ny, l in itertools.product(nu_range, ny_range, l_range):

            print(f"nu: {nu}, ny: {ny}, l: {l}")

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

            #print(model)


            ### Criando matriz de regressores final de tamanho n_theta
            #Psi = np.zeros((n-grau,n_theta))
            #for i in range(n_theta):
            #    Psi[:,i] = candidatos[:,h[i]]
            #y_hat = Psi @ theta


            #############
            ##  Teste  ##

            n_test = len(u_test)
            y_hat_test_free = np.zeros(n_test)
            y_hat_test_onestep = np.zeros(n_test)
            y_hat_test_free[:grau] = y_test[:grau]


            for k in range(grau, n_test):
                regressor_values_free = []
                regressor_values_onestep = []
                for reg in chosen_regressors:
                    matches = re.compile(r'([yu])\(k-(\d+)\)').findall(reg)
                product_free = 1
                product_onestep = 1
                for var, delay_str in matches:
                    delay = int(delay_str)
                    if var == 'y':
                        product_free *= y_hat_test_free[k - delay]
                        product_onestep *= y_test[k - delay]
                    elif var == 'u':
                        product_free *= u_test[k - delay]
                        product_onestep *= u_test[k - delay]

                regressor_values_free.append(product_free)
                regressor_values_onestep.append(product_onestep)
            
                y_hat_test_free[k] = sum(theta * reg for theta, reg in zip(theta, regressor_values_free))
                y_hat_test_onestep[k] = sum(theta * reg for theta, reg in zip(theta, regressor_values_onestep))


            #Erro Quadrático Médio
            mse_free = np.mean((y_hat_test_free - y_test)**2)
            variance_free = np.var(y_test)
            nmse_free = mse_free / variance_free
            print("Normalized Mean Squared Error Free Simulation:", nmse_free)
            mse_onestep = np.mean((y_hat_test_onestep - y_test)**2)
            variance_onestep = np.var(y_test)
            nmse_onestep = mse_onestep / variance_onestep
            print("Normalized Mean Squared Error One Step Prediction:", nmse_onestep)

            
            df_list.append({
                "system": system,
                "model": model,
                "nu": nu,
                "ny": ny,
                "l": l,
                "mse_free": mse_free,
                "nmse_free": nmse_free,
                "mse_onestep": mse_onestep,
                "nmse_onestep": nmse_onestep
            })

df = pd.DataFrame(df_list)
df.to_csv("model_results.csv", index=False)
print("Results saved to model_results.csv")

            
            


