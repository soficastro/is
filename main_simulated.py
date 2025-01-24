import numpy as np
import matplotlib.pyplot as plt
from frols import frols
from utils import matriz_candidatos

print('--')

n_steps = 1000 
np.random.seed(0)

nu = 2                  # Atraso da entrada
ny = 2                  # Atraso da saída
l = 2                   # Grau de não linearidade
max_delay = max(nu, ny)

# Sinal pseudo-aleatório u(k) com média 0.5
u = np.random.uniform(-0.5, 1.5, n_steps)

# Sinal y(k)
y = np.zeros(n_steps)
for k in range(max_delay, n_steps):
    y[k] = (
        1.3920 * y[k-1]
        - 0.4235 * y[k-2]
        - 0.4388 * u[k-2] * y[k-1]
        + 0.3756 * u[k-2] * y[k-2]
        + 0.0454 * u[k-1]**2
        + 0.0218 * u[k-2]**2
        + 0.0097
        + u[k-2] * u[k-1]
    )


### TREINO ###

n_train = int(0.75 * u.shape[0])
u_train, u_test = u[:n_train], u[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

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
#print(model)

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
plt.title("Ferro de solda")
plt.legend()
plt.grid(True)
plt.show()