import pandas as pd
import numpy as np
from frols import melhor_ordem, matriz_candidatos
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data\exchanger.dat', delimiter='\t', header = None)

time_steps = data[data.columns[0]].to_numpy()
input_q = data[data.columns[1]].to_numpy()
output_th = data[data.columns[2]].to_numpy()

### Normalizando dados
input_q = input_q/np.max(np.abs(input_q))
output_th = output_th/np.max(np.abs(output_th))

'''
plt.plot(time_steps, input_q, label="entrada normalizada")
plt.plot(time_steps, output_th, label="saida normalizada")
plt.title("Entrada e saída normalizadas")
plt.legend()
plt.show()
'''

### Treino em 20% dos dados
n_train = int(0.2 * input_q.shape[0])

t_train, t_test = time_steps[-n_train:], time_steps[:-n_train]
u_train, u_test = input_q[-n_train:], input_q[:-n_train]
y_train, y_test = output_th[-n_train:], output_th[:-n_train]

### Hiperparâmetros
nu = 3 #atraso da entrada
ny = 2 #atraso da saida
ne = 1 # quantidade de erro
l = 2  #grau de nao linearidade
n_theta = 5 #quantos regressores usar


print(f'Hiperparâmetros NARX: nu = {nu}, ny = {ny}, l = {l}, n_theta = {n_theta}')

################
#### Treino ####

########
### NARX
candidatos, M, combinations = matriz_candidatos(input = u_train, output = y_train, nu = nu, ny = ny, l = l)
n = len(u_train)
grau = max(nu, ny)
h, ERR_total = melhor_ordem(candidatos = candidatos, M = M, input = u_train, output = y_train, grau = grau, n_theta = n_theta)


print(combinations)
print(h)
print(ERR_total)


##########
### NARMAX
#candidatos_narmax, M_narmax, combinations_narmax = matriz_candidatos_narmax(input = u_train, output = y_train, nu = nu, ny = ny, ne = ne, l = l)
#grau_narmax = max(nu,ny, ne)
#h_narmax = melhor_ordem(candidatos = candidatos_narmax, M = M_narmax, input = u_train, output = y_train, grau = grau_narmax, n_theta = n_theta)



### Criando matriz de regressores final de tamanho n_theta
Psi = np.zeros((n-grau,n_theta))
for i in range(n_theta):
    Psi[:,i] = candidatos[:,h[i]]

### Mínimos Quadrados
theta = np.linalg.inv(Psi.T @ Psi) @ Psi.T @ y_train[:-grau]
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



### Criando matriz de regressores final de tamanho n_theta
#Psi_narmax = np.zeros((n-grau_narmax,n_theta))
#for i in range(n_theta):
#    Psi_narmax[:,i] = candidatos_narmax[:,h_narmax[i]]

### Mínimos Quadrados
#theta_narmax = np.linalg.inv(Psi_narmax.T @ Psi_narmax) @ Psi_narmax.T @ y_train[:-grau_narmax]
#y_hat_narmax = Psi_narmax @ theta_narmax

'''
plt.plot(t_train[:-grau], y_train[:-grau], label="y")
plt.plot(t_train[:-grau], y_hat, label="y_hat")
plt.title("Treino")
plt.legend()
plt.show()
'''


#mse_treino_narmax = mean_squared_error(y_train[:-grau_narmax], y_hat_narmax)
#print('erro treino narmax')
#print(mse_treino_narmax)


#############
##  Teste  ##

#############################
# Teste com Mínimos Quadrados
# Usa h e theta originais
candidatos_teste, M_teste, combinations = matriz_candidatos(u_test, y_test, nu, ny, l)
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
