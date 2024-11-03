import itertools
import numpy as np

def matriz_candidatos(input, output, nu, ny, l):
    n = len(input)
    grau = max(nu, ny)
    linear_psi = np.zeros((n - grau, nu + ny))

    # Criar a matriz Psi linear com os valores de entrada e saída
    for i in range(ny):
        linear_psi[:, ny - 1 - i] = output[i + 1:n - grau + i + 1]
    for i in range(nu):
        linear_psi[:, ny + nu - 1 - i] = input[i + 1:n - grau + i + 1]

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
                term_names.append(f"y(k-{idx + 1})")
            else:
                term_names.append(f"u(k-{idx - ny + 1})")
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