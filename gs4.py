import pandas as pd
import numpy as np
from gramschmidt import *
import matplotlib.pyplot as plt
import scipy.io
from sklearn.metrics import mean_squared_error

data = scipy.io.loadmat('ab2\dataBenchmark.mat')

uEst = np.reshape(data['uEst'],1024)
yEst = np.reshape(data['yEst'],1024)
uVal = np.reshape(data['uVal'],1024)
yVal = np.reshape(data['yVal'],1024)

timestamp = np.arange(0, uEst.shape[0] * 0.1, 0.1)

### Normalizando dados
uEst = uEst/np.max(np.abs(uEst))
yEst = yEst/np.max(np.abs(yEst))
uVal = yEst/np.max(np.abs(uVal))
yVal = yEst/np.max(np.abs(yVal))

'''
plt.plot(timestamp, uEst, label="entrada treino normalizada")
plt.plot(timestamp, yEst, label="saida treino normalizada")
plt.plot(timestamp, uVal, label="entrada teste normalizada")
plt.plot(timestamp, yVal, label="saida teste normalizada")
plt.title("Entrada e saída normalizadas")
plt.legend()
plt.show()
'''

### Hiperparâmetros
nu = 3 #atraso da entrada
ny = 4 #atraso da saida
l = 3  #grau de nao linearidade
n_theta = 4 #quantos regressores usar

'''
print(f'Hiperparâmetros NARX: nu = {nu}, ny = {ny}, l = {l}, n_theta = {n_theta}')
'''

################
#### Treino ####

candidatos, M, combinations = matriz_candidatos(uEst, yEst, nu, ny, l)
n = len(uEst)
grau = max(nu, ny)
h, ERR_total = melhor_ordem(candidatos, M, uEst, yEst, grau, n_theta)

'''
print(h)
print(combinations)
print(ERR_total)
'''

### Criando matriz de regressores final de tamanho ic
Psi = np.zeros((n-grau,n_theta))
for i in range(n_theta):
    Psi[:,i] = candidatos[:,h[i]]

### Mínimos Quadrados
theta = np.linalg.inv(Psi.T @ Psi) @ Psi.T @ yEst[:-max(nu, ny)]
y_hat = Psi @ theta

'''
print(theta)
'''

'''
plt.plot(timestamp[:-grau], yEst[:-grau], label="y")
plt.plot(timestamp[:-grau], y_hat, label="y_hat")
plt.title("Treino")
plt.legend()
plt.show()
'''

mse_treino = mean_squared_error(yEst[:-grau], y_hat)
print('erro treino')
print(mse_treino)


#############
##  Teste  ##

#############################
# Teste com Mínimos Quadrados
# Usa h e theta originais
candidatos_teste, n_theta_teste, combinations = matriz_candidatos(uVal, yVal, nu, ny, l)
n_teste = len(uVal)
Psi_teste = np.zeros((n_teste-grau,n_theta))
for i in range(n_theta):
    Psi_teste[:,i] = candidatos_teste[:,h[i]]
y_hat_teste = Psi_teste @ theta

mse_teste = mean_squared_error(yVal[:-grau], y_hat_teste)
print('erro teste com minimos quadrados')
print(mse_teste)

'''
plt.plot(timestamp[:-grau], yVal[:-grau], label="y")
plt.plot(timestamp[:-grau], y_hat_teste, label="y_hat")
plt.title("Teste")
plt.legend()
plt.show()
'''