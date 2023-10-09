import pandas as pd
import numpy as np
from gramschmidt import *
import matplotlib.pyplot as plt
import scipy.io
from sklearn.metrics import mean_squared_error

data = scipy.io.loadmat('ab2\SNLS80mV.mat')

input_V1 = np.reshape(data['V1'],131072)
output_V2 = np.reshape(data['V2'],131072)

### Normalizando dados
input_V1 = input_V1/np.max(np.abs(input_V1))
output_V2 = output_V2/np.max(np.abs(output_V2))

timestamp = np.arange(0, input_V1.shape[0] * 0.1, 0.1)

'''
plt.plot(timestamp, input_V1, label="entrada normalizada")
plt.plot(timestamp, output_V2, label="saida normalizada")
plt.title("Entrada e saída normalizadas")
plt.legend()
plt.show()
'''

### Treino em 20% dos dados
n_train = int(0.2 * input_V1.shape[0])

t_train, t_test = timestamp[-n_train:], timestamp[:-n_train]
u_train, u_test = input_V1[-n_train:], input_V1[:-n_train]
y_train, y_test = output_V2[-n_train:], output_V2[:-n_train]

### Hiperparâmetros
nu = 2 #atraso da entrada
ny = 3 #atraso da saida
l = 2  #grau de nao linearidade
n_theta = 4 #quantos regressores usar

'''
print(f'Hiperparâmetros NARX: nu = {nu}, ny = {ny}, l = {l}, n_theta = {n_theta}')
'''

################
#### Treino ####

candidatos, M, combinations = matriz_candidatos(u_train, y_train, nu, ny, l)
n = len(u_train)
grau = max(nu, ny)
h, ERR_total = melhor_ordem(candidatos, M, u_train, y_train, grau, n_theta)

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
theta = np.linalg.inv(Psi.T @ Psi) @ Psi.T @ y_train[:-max(nu, ny)]
y_hat = Psi @ theta

print(theta)


plt.plot(t_train[:-grau], y_train[:-grau], label="y")
plt.plot(t_train[:-grau], y_hat, label="y_hat")
plt.title("Treino")
plt.legend()
plt.show()


mse_treino = mean_squared_error(y_train[:-grau], y_hat)
print('erro treino')
print(mse_treino)


#############
##  Teste  ##

#############################
# Teste com Mínimos Quadrados
# Usa h e theta originais
candidatos_teste, n_theta_teste, combinations = matriz_candidatos(u_test, y_test, nu, ny, l)
n_teste = len(u_test)
Psi_teste = np.zeros((n_teste-grau,n_theta))
for i in range(n_theta):
    Psi_teste[:,i] = candidatos_teste[:,h[i]]
y_hat_teste = Psi_teste @ theta

mse_teste = mean_squared_error(y_test[:-grau], y_hat_teste)
print('erro teste com minimos quadrados')
print(mse_teste)


plt.plot(t_test[:-grau], y_test[:-grau], label="y")
plt.plot(t_test[:-grau], y_hat_teste, label="y_hat")
plt.title("Teste")
plt.legend()
plt.show()
