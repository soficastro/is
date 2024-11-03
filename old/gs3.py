import pandas as pd
import numpy as np
from gramschmidt import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data = pd.read_csv('ab2\\robot_arm.dat', delimiter='\t', header = None)

u = data[data.columns[0]].to_numpy()
y = data[data.columns[1]].to_numpy()

timestamp = np.arange(0, u.shape[0] * 0.1, 0.1)

'''
plt.plot(timestamp, u, label="entrada original")
plt.plot(timestamp, y, label="saida original")
plt.title("Entrada e saída originais")
plt.legend()
plt.show()
'''

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
nu = 2 #atraso da entrada
ny = 2 #atraso da saida
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
print(h)
print(combinations)
print(ERR_total)
'''


### Criando matriz de regressores final de tamanho ic
Psi = np.zeros((n-grau,n_theta))
for i in range(n_theta):
    Psi[:,i] = candidatos[:,h[i]]

### Mínimos Quadrados
theta = np.linalg.inv(Psi.T @ Psi) @ Psi.T @ y_train[:-grau]
y_hat = Psi @ theta


plt.plot(t_train[:-grau], y_train[:-grau], label="y")
plt.plot(t_train[:-grau], y_hat, label="y_hat")
plt.title("Treino")
plt.legend()
plt.show()


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
candidatos_teste_1, n_theta_teste_1, combinations_1 = matriz_candidatos(u_test, y_test, nu, ny, l)
n_teste_1 = len(u_test)
Psi_teste_1 = np.zeros((n_teste_1-grau,n_theta))
for i in range(n_theta):
    Psi_teste_1[:,i] = candidatos_teste_1[:,h[i]]
y_hat_teste_1 = Psi_teste_1 @ theta

mse_teste_1 = mean_squared_error(y_test[:-grau], y_hat_teste_1)
print('erro teste com minimos quadrados')
print(mse_teste_1)

#######################
# Teste com modelo NARX
n_teste_2 = len(u_test)
y_hat_teste_2 = np.zeros((n_teste_2,1))
for k in range(grau,n_teste_2):
    y_hat_teste_2[k,0] = theta[0]*y_test[k-1]+theta[1]*y_test[k-2]+theta[2]*u_test[k-2]*u_test[k-2]

mse_teste_2 = mean_squared_error(y_test[grau:], y_hat_teste_2[grau:])
print('erro teste com modelo')
print(mse_teste_2)


plt.plot(t_test[:-grau], y_test[:-grau], label="y")
plt.plot(t_test[:-grau], y_hat_teste_1, label="y_hat")
plt.title("Teste")
plt.legend()
plt.show()
