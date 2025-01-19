import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.io


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

        grau = max(nu, ny)


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
        


        #############
        ##  Teste  ##

        n_test = len(u_test)
        y_hat_test_free = np.zeros(n_test, dtype='float64')
        y_hat_test_onestep = np.zeros(n_test, dtype='float64')
        y_hat_test_free[:grau] = y_test[:grau]
        y_hat_test_onestep[:grau] = y_test[:grau]


        if system == 'Ball-beam':
            # Free simulation
            for k in range(grau, n_test):
                y_hat_test_free[k] =  -40388018.5 * y_test[k-1] + -5879909333.0 * y_test[k-3] + -5806957672.0 * u_test[k-3] + -106088407199.5 * u_test[k-1] + -106646930959.5 * y_test[k-2] + -106644391546.5 * u_test[k-3] * u_test[k-3] * u_test[k-3] + -19787848710.5 * y_test[k-3] * u_test[k-2] * u_test[k-3] + 942776162802.5 * y_test[k-2] * u_test[k-2] * u_test[k-3] + 455570541354.5 * u_test[k-1] * u_test[k-2] * u_test[k-2] + -2321855772272.5 * y_test[k-1] * u_test[k-3] * u_test[k-3] + 63440099248.5 * y_test[k-1] * y_test[k-1] * y_test[k-1] + 5785186421917.5 * y_test[k-3] * y_test[k-3] * y_test[k-3] + 36896796219237.5 * y_test[k-2] * y_test[k-3] * y_test[k-3] + 33414820791641.5 * y_test[k-1] * y_test[k-1] * u_test[k-1] + 1125987881170731.0 * y_test[k-3] * y_test[k-3] * u_test[k-3] + 2.132190155272623e+16 * y_test[k-2] * y_test[k-2] * u_test[k-3] + 2.0644108669246984e+16 * y_test[k-1] * u_test[k-1] * u_test[k-1] + -28431288827043.5 * y_test[k-3] * y_test[k-3] + -3251792832271079.5 * y_test[k-2] * y_test[k-2] + -4.384883872691589e+16 * y_test[k-1] * y_test[k-3] + -4.404808219138314e+16 * y_test[k-2] * y_test[k-2] * y_test[k-2] + -5.176935237510885e+16 * y_test[k-1] * y_test[k-3] * u_test[k-3] + 1.143523429935379e+16 * y_test[k-3] * u_test[k-3] + 9454379288227438.0 * u_test[k-1] * u_test[k-3] + 9453498161148014.0 * u_test[k-1] * u_test[k-1] * u_test[k-2] + 80245634410318.88 * y_test[k-2] * u_test[k-2] + 105563071670094.34 * u_test[k-3] * u_test[k-3] + 105452095802579.69 * y_test[k-3] * u_test[k-2] * u_test[k-2] + -156096636024797.3 * y_test[k-1] * y_test[k-1] * y_test[k-2] + -203796976199877.06 * y_test[k-1] * y_test[k-1] * u_test[k-3] + 27594827574797.47 * y_test[k-2] * y_test[k-3] + 27605091720140.6 * y_test[k-1] * y_test[k-2] * u_test[k-3] + 15043150654098.725 * y_test[k-2] * y_test[k-3] * u_test[k-3] + 461806734355.20807 * u_test[k-1] * u_test[k-2] + 480603181372.54816 * y_test[k-1] * y_test[k-1] * y_test[k-3] + 480882052424.6461 * u_test[k-2] * u_test[k-2] * u_test[k-2] + -42752192700.34241 * y_test[k-3] * u_test[k-1] * u_test[k-3] + -42615721124.2175 * u_test[k-1] * u_test[k-1] * u_test[k-1] + 65053155608.54492 * y_test[k-2] * u_test[k-1] * u_test[k-3] + -50903064504.77722 * y_test[k-1] * u_test[k-1] * u_test[k-2] + -29521265111.039127 * y_test[k-2] * u_test[k-3] * u_test[k-3] + -16373564659.548445 * y_test[k-1] * u_test[k-2] * u_test[k-3] + 11275103683.90919 * y_test[k-2] * y_test[k-2] * y_test[k-3] + 12140605827.066471 * y_test[k-3] * u_test[k-3] * u_test[k-3] + 6848681263.718408 * y_test[k-1] * u_test[k-1] * u_test[k-3] + 2078484663.7003756 * y_test[k-1] * y_test[k-3] * y_test[k-3] + 2714449172.465214 * y_test[k-1] * y_test[k-2] * y_test[k-2] + 27037486123.048218 * y_test[k-1] * y_test[k-2] * y_test[k-3] + 147518460.55205247 * y_test[k-1] * u_test[k-2] + 86450579.98421784 * y_test[k-3] * u_test[k-2] + 41138526.1047672 * y_test[k-1] * u_test[k-3] + 62876194.222821645 * u_test[k-1] * u_test[k-1] + 43001609.96126571 * u_test[k-2] * u_test[k-3] + -3747114802.508436 * u_test[k-2] * u_test[k-2] + -3783418854.782421 * y_test[k-3] * u_test[k-1] * u_test[k-1] + -3175200779.7974925 * y_test[k-1] * u_test[k-2] * u_test[k-2] + -462882226.0368101 * y_test[k-3] * y_test[k-3] * u_test[k-1] + -943385224.6450307 * y_test[k-2] * y_test[k-2] * u_test[k-1] + 5649778954.438977 * y_test[k-1] * y_test[k-2] + -1985441211.8000631 * y_test[k-1] * y_test[k-1] + -1929424821.4228961 * y_test[k-1] * u_test[k-1] + 193107766276.531 * y_test[k-3] * u_test[k-1] + 192061180774.77872 * y_test[k-3] * y_test[k-3] * u_test[k-2] + 5055458862225.991 * y_test[k-2] * y_test[k-2] * u_test[k-2] + 5218192901561.273 * u_test[k-2] * u_test[k-3] * u_test[k-3] + 366696892451824.7 * u_test[k-2] * u_test[k-2] * u_test[k-3] + 7351907713.976022 * u_test[k-2] + 7351863101.615578 * y_test[k-1] * y_test[k-2] * u_test[k-2] + 7384213204.161502 * y_test[k-3] * u_test[k-1] * u_test[k-2] + 4109830734.777588 * u_test[k-1] * u_test[k-1] * u_test[k-3] + 8435970595.37293 * u_test[k-1] * u_test[k-2] * u_test[k-3] + -2964741120.422981 * u_test[k-1] * u_test[k-3] * u_test[k-3] + 78679418.72399142 * y_test[k-1] * y_test[k-3] * u_test[k-1] + 109785717.34703153 * y_test[k-2] * y_test[k-3] * u_test[k-1] + 56772205.919439234 * y_test[k-1] * y_test[k-2] * u_test[k-1] + 51920785.06243362 * y_test[k-2] * u_test[k-1] * u_test[k-1] + -4242869.130395196 * y_test[k-2] * u_test[k-1] + -2194767.546757064 * y_test[k-2] * u_test[k-3] + -2214365.779229915 * y_test[k-2] * u_test[k-2] * u_test[k-2] + 2220727.7677553757 * y_test[k-2] * u_test[k-1] * u_test[k-2] + -92546.39332357633 * y_test[k-2] * y_test[k-3] * u_test[k-2] + 40788.36937542814 * y_test[k-1] * y_test[k-3] * u_test[k-2] + 15877.02783205536 * y_test[k-1] * y_test[k-1] * u_test[k-2]
            # One-step ahead prediction (for comparison)
            for k in range(grau, n_test):
                y_hat_test_onestep[k] = -40388018.5 * y_test[k-1] + -5879909333.0 * y_test[k-3] + -5806957672.0 * u_test[k-3] + -106088407199.5 * u_test[k-1] + -106646930959.5 * y_test[k-2] + -106644391546.5 * u_test[k-3] * u_test[k-3] * u_test[k-3] + -19787848710.5 * y_test[k-3] * u_test[k-2] * u_test[k-3] + 942776162802.5 * y_test[k-2] * u_test[k-2] * u_test[k-3] + 455570541354.5 * u_test[k-1] * u_test[k-2] * u_test[k-2] + -2321855772272.5 * y_test[k-1] * u_test[k-3] * u_test[k-3] + 63440099248.5 * y_test[k-1] * y_test[k-1] * y_test[k-1] + 5785186421917.5 * y_test[k-3] * y_test[k-3] * y_test[k-3] + 36896796219237.5 * y_test[k-2] * y_test[k-3] * y_test[k-3] + 33414820791641.5 * y_test[k-1] * y_test[k-1] * u_test[k-1] + 1125987881170731.0 * y_test[k-3] * y_test[k-3] * u_test[k-3] + 2.132190155272623e+16 * y_test[k-2] * y_test[k-2] * u_test[k-3] + 2.0644108669246984e+16 * y_test[k-1] * u_test[k-1] * u_test[k-1] + -28431288827043.5 * y_test[k-3] * y_test[k-3] + -3251792832271079.5 * y_test[k-2] * y_test[k-2] + -4.384883872691589e+16 * y_test[k-1] * y_test[k-3] + -4.404808219138314e+16 * y_test[k-2] * y_test[k-2] * y_test[k-2] + -5.176935237510885e+16 * y_test[k-1] * y_test[k-3] * u_test[k-3] + 1.143523429935379e+16 * y_test[k-3] * u_test[k-3] + 9454379288227438.0 * u_test[k-1] * u_test[k-3] + 9453498161148014.0 * u_test[k-1] * u_test[k-1] * u_test[k-2] + 80245634410318.88 * y_test[k-2] * u_test[k-2] + 105563071670094.34 * u_test[k-3] * u_test[k-3] + 105452095802579.69 * y_test[k-3] * u_test[k-2] * u_test[k-2] + -156096636024797.3 * y_test[k-1] * y_test[k-1] * y_test[k-2] + -203796976199877.06 * y_test[k-1] * y_test[k-1] * u_test[k-3] + 27594827574797.47 * y_test[k-2] * y_test[k-3] + 27605091720140.6 * y_test[k-1] * y_test[k-2] * u_test[k-3] + 15043150654098.725 * y_test[k-2] * y_test[k-3] * u_test[k-3] + 461806734355.20807 * u_test[k-1] * u_test[k-2] + 480603181372.54816 * y_test[k-1] * y_test[k-1] * y_test[k-3] + 480882052424.6461 * u_test[k-2] * u_test[k-2] * u_test[k-2] + -42752192700.34241 * y_test[k-3] * u_test[k-1] * u_test[k-3] + -42615721124.2175 * u_test[k-1] * u_test[k-1] * u_test[k-1] + 65053155608.54492 * y_test[k-2] * u_test[k-1] * u_test[k-3] + -50903064504.77722 * y_test[k-1] * u_test[k-1] * u_test[k-2] + -29521265111.039127 * y_test[k-2] * u_test[k-3] * u_test[k-3] + -16373564659.548445 * y_test[k-1] * u_test[k-2] * u_test[k-3] + 11275103683.90919 * y_test[k-2] * y_test[k-2] * y_test[k-3] + 12140605827.066471 * y_test[k-3] * u_test[k-3] * u_test[k-3] + 6848681263.718408 * y_test[k-1] * u_test[k-1] * u_test[k-3] + 2078484663.7003756 * y_test[k-1] * y_test[k-3] * y_test[k-3] + 2714449172.465214 * y_test[k-1] * y_test[k-2] * y_test[k-2] + 27037486123.048218 * y_test[k-1] * y_test[k-2] * y_test[k-3] + 147518460.55205247 * y_test[k-1] * u_test[k-2] + 86450579.98421784 * y_test[k-3] * u_test[k-2] + 41138526.1047672 * y_test[k-1] * u_test[k-3] + 62876194.222821645 * u_test[k-1] * u_test[k-1] + 43001609.96126571 * u_test[k-2] * u_test[k-3] + -3747114802.508436 * u_test[k-2] * u_test[k-2] + -3783418854.782421 * y_test[k-3] * u_test[k-1] * u_test[k-1] + -3175200779.7974925 * y_test[k-1] * u_test[k-2] * u_test[k-2] + -462882226.0368101 * y_test[k-3] * y_test[k-3] * u_test[k-1] + -943385224.6450307 * y_test[k-2] * y_test[k-2] * u_test[k-1] + 5649778954.438977 * y_test[k-1] * y_test[k-2] + -1985441211.8000631 * y_test[k-1] * y_test[k-1] + -1929424821.4228961 * y_test[k-1] * u_test[k-1] + 193107766276.531 * y_test[k-3] * u_test[k-1] + 192061180774.77872 * y_test[k-3] * y_test[k-3] * u_test[k-2] + 5055458862225.991 * y_test[k-2] * y_test[k-2] * u_test[k-2] + 5218192901561.273 * u_test[k-2] * u_test[k-3] * u_test[k-3] + 366696892451824.7 * u_test[k-2] * u_test[k-2] * u_test[k-3] + 7351907713.976022 * u_test[k-2] + 7351863101.615578 * y_test[k-1] * y_test[k-2] * u_test[k-2] + 7384213204.161502 * y_test[k-3] * u_test[k-1] * u_test[k-2] + 4109830734.777588 * u_test[k-1] * u_test[k-1] * u_test[k-3] + 8435970595.37293 * u_test[k-1] * u_test[k-2] * u_test[k-3] + -2964741120.422981 * u_test[k-1] * u_test[k-3] * u_test[k-3] + 78679418.72399142 * y_test[k-1] * y_test[k-3] * u_test[k-1] + 109785717.34703153 * y_test[k-2] * y_test[k-3] * u_test[k-1] + 56772205.919439234 * y_test[k-1] * y_test[k-2] * u_test[k-1] + 51920785.06243362 * y_test[k-2] * u_test[k-1] * u_test[k-1] + -4242869.130395196 * y_test[k-2] * u_test[k-1] + -2194767.546757064 * y_test[k-2] * u_test[k-3] + -2214365.779229915 * y_test[k-2] * u_test[k-2] * u_test[k-2] + 2220727.7677553757 * y_test[k-2] * u_test[k-1] * u_test[k-2] + -92546.39332357633 * y_test[k-2] * y_test[k-3] * u_test[k-2] + 40788.36937542814 * y_test[k-1] * y_test[k-3] * u_test[k-2] + 15877.02783205536 * y_test[k-1] * y_test[k-1] * u_test[k-2]
            
        elif system == 'Tanks':
            # Free simulation
            for k in range(grau, n_test):
                y_hat_test_free[k] = 3.410568431291334 * y_hat_test_free[k-1] - 2.4119831321059597 * y_hat_test_free[k-2]
            # One-step ahead prediction (for comparison)
            for k in range(grau, n_test):
                y_hat_test_onestep[k] = 3.410568431291334 * y_test[k-1] - 2.4119831321059597 * y_test[k-2]
        elif system == 'Heat Exchanger':
            # Free simulation
            for k in range(grau, n_test):
                y_hat_test_free[k] = 1.0453869439783245 * y_hat_test_free[k-1] - -0.00046680950956340915 * y_hat_test_free[k-2] * y_hat_test_free[k-4] 
            # One-step ahead prediction (for comparison)
            for k in range(grau, n_test):
                y_hat_test_onestep[k] = 1.0453869439783245 * y_test[k-1] - -0.00046680950956340915 * y_test[k-2] * y_test[k-4] 
        elif system == 'Robot Arm':
            # Free simulation
            for k in range(grau, n_test):
                y_hat_test_free[k] = 3.57884872014148852 * y_hat_test_free[k-3] - 4.149451261352542 * u_test[k-2] + 3.5822804838520295 * y_hat_test_free[k-1] + 0.6380716527774113 * y_hat_test_free[k-2]
            # One-step ahead prediction (for comparison)
            for k in range(grau, n_test):
                y_hat_test_onestep[k] = 3.5788487201414885 * y_test[k-3] - 4.149451261352542 * u_test[k-2] + 3.5822804838520295 * y_test[k-1] + 0.6380716527774113 * y_test[k-2]
        elif system == 'Silverbox':
            # Free simulation
            for k in range(grau, n_test):
                y_hat_test_free[k] = 1.6668790238423223 * y_hat_test_free[k-1] - 0.25038834009614064 * y_hat_test_free[k-2] 
            # One-step ahead prediction (for comparison)
            for k in range(grau, n_test):
                y_hat_test_onestep[k] = 1.6668790238423223 * y_test[k-1] - 0.25038834009614064 * y_test[k-2]
        


        #Erro Quadrático Médio
        rmse_free = np.sqrt(np.mean(np.square(y_hat_test_free - y_test)))
        print("Root Mean Squared Error Free Simulation:", rmse_free)
        rmse_onestep = np.sqrt(np.mean(np.square(y_hat_test_onestep - y_test)))
        print("Root Mean Squared Error One Step Prediction:", rmse_onestep)

        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label="Saída esperada", color="b")
        plt.plot(y_hat_test_onestep, label="Saída estimada", color="r", linestyle="--")
        plt.xlabel("Amostras")
        plt.ylabel("Amplitude")
        plt.title(f"{system} - Passo")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label="Saída esperada", color="b")
        plt.plot(y_hat_test_free, label="Saída estimada", color="r", linestyle="--")
        plt.xlabel("Amostras")
        plt.ylabel("Amplitude")
        plt.title(f"{system} - Livre")
        plt.legend()
        plt.grid(True)
        plt.show()

    
           

            
            