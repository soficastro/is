import itertools
import numpy as np

def matriz_candidatos(input, output, nu, ny, l):
    n = len(input)
    grau = max(nu, ny)
    linear_psi = np.zeros((n - grau, nu + ny))

    # Criar a matriz Psi linear com os valores de entrada e saída
    for i in range(ny):
        linear_psi[:, ny - 1 - i] = output[i:n - grau + i]
    for i in range(nu):
        linear_psi[:, ny + nu - 1 - i] = input[i:n - grau + i]


    # Gerar combinações de termos até grau l
    array = np.arange(nu + ny)
    combinations = find_combinations(array, l)

    M = len(combinations)

    # Criar matriz de candidatos Psi
    candidatos = np.array([np.prod(linear_psi[:, comb], axis=1) for comb in combinations]).T

    regressor_names = generate_regressor_names(nu, ny, combinations)

    return candidatos, M, regressor_names


def find_combinations(arr, l):
    combinations = []
    for r in range(1, l + 1):
        combinations.extend(list(itertools.combinations_with_replacement(arr, r)))
    return combinations


def generate_regressor_names(nu, ny, combinations):
    names = []
    for comb in combinations:
        term_names = []
        for idx in comb:
            if idx < ny:
                term_names.append(f"y_test[k-{idx + 1}]")
            else:
                term_names.append(f"u_test[k-{idx - ny + 1}]")
        names.append(" * ".join(term_names))
    return names


'''
            plt.figure(figsize=(10, 6))
            plt.plot(y_test[:-grau], label="Saída esperada", color="b")
            plt.plot(y_hat_test, label="Saída estimada", color="r", linestyle="--")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title(f"{system}")
            plt.legend()
            plt.grid(True)
            plt.show()
            '''

'''def prediction(u_test, y_test, grau, theta, chosen_regressors, n_test, y_hat_test_free, y_hat_test_onestep):
    for k in range(grau, n_test):
        regressor_values_free = []
        regressor_values_onestep = []
        for reg in chosen_regressors:
            matches = re.compile(r'([yu])\(k-(\d+)\)').findall(reg)
            print(matches)
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
        '''