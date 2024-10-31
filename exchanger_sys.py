import pandas as pd
import numpy as np
from frols import frols
from matriz_candidatos import matriz_candidatos
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data\exchanger.dat', delimiter='\t', header = None)

time_steps = data[data.columns[0]].to_numpy()
input_q = data[data.columns[1]].to_numpy()
output_th = data[data.columns[2]].to_numpy()

### Normalizando dados
input_q = input_q/np.max(np.abs(input_q))
output_th = output_th/np.max(np.abs(output_th))

### Treino em 20% dos dados
n_train = int(0.2 * input_q.shape[0])

t_train, t_test = time_steps[-n_train:], time_steps[:-n_train]
u_train, u_test = input_q[-n_train:], input_q[:-n_train]
y_train, y_test = output_th[-n_train:], output_th[:-n_train]

### Hiperparâmetros
nu = 3 #atraso da entrada
ny = 2 #atraso da saida
l = 2  #grau de nao linearidade


################
#### Treino ####

########
### NARX
candidatos, M, combinations = matriz_candidatos(input = u_train, output = y_train, nu = nu, ny = ny, l = l)
n = len(u_train)
grau = max(nu, ny)
h, ERR_total, theta = frols(candidatos = candidatos, M = M, output = y_train, grau = grau)
n_theta = len(theta)

print(theta)

### Criando matriz de regressores final de tamanho n_theta
Psi = np.zeros((n-grau,n_theta))
for i in range(n_theta):
    Psi[:,i] = candidatos[:,h[i]]


y_hat = Psi @ theta

mse_treino = mean_squared_error(y_train[:-grau], y_hat)
print('erro treino')
print(mse_treino)




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

