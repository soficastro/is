import pandas as pd
import numpy as np
from gramschmidt import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data = pd.read_csv('ab2/ballbeam.dat', delimiter='\t', header = None)

u = data[data.columns[0]].to_numpy()
y = data[data.columns[1]].to_numpy()

timestamp = np.arange(0, u.shape[0] * 0.1, 0.1)

### Normalizando dados
u = u/np.max(np.abs(u))
y = y/np.max(np.abs(y))

'''
plt.plot(timestamp, u, label="entrada normalizada")
plt.plot(timestamp, y, label="saida normalizada")
plt.title("Entrada e saída normalizadas")
plt.legend()
plt.show()
'''

### Treino em 20% dos dados
n_train = int(0.2 * u.shape[0])

t_train, t_test = timestamp[-n_train:], timestamp[:-n_train]
u_train, u_test = u[-n_train:], u[:-n_train]
y_train, y_test = y[-n_train:], y[:-n_train]

### Hiperparâmetros
nu = 3 #atraso da entrada
ny = 4 #atraso da saida
l = 2  #grau de nao linearidade
n_theta = 3 #quantos regressores usar

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
print(combinations)
print(h)
print(ERR_total)
'''


### Criando matriz de regressores final de tamanho ic
Psi = np.zeros((n-grau,n_theta))
for i in range(n_theta):
    Psi[:,i] = candidatos[:,h[i]]

### Mínimos Quadrados
theta = np.linalg.inv(Psi.T @ Psi) @ Psi.T @ y_train[:-max(nu, ny)]
y_hat = Psi @ theta

'''
plt.plot(t_train[:-grau], y_train[:-grau], label="y")
plt.plot(t_train[:-grau], y_hat, label="y_hat")
plt.title("Treino")
plt.legend()
plt.show()
'''
'''
print(theta)
'''

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

'''
plt.plot(t_test[:-grau], y_test[:-grau], label="y")
plt.plot(t_test[:-grau], y_hat_teste, label="y_hat")
plt.title("Teste")
plt.legend()
plt.show()
'''