import itertools
import numpy as np

def matriz_candidatos(input, output, nu, ny, l):
    n = len(input)
    max_delay = max(nu, ny)
    linear_psi = np.zeros((n - max_delay, nu + ny))

    # Criar a matriz Psi linear com os valores de entrada e saída
    for i in range(ny):
        linear_psi[:, ny - 1 - i] = output[i:n - max_delay + i]
    for i in range(nu):
        linear_psi[:, ny + nu - 1 - i] = input[i:n - max_delay + i]

    # Gerar combinações de termos até grau l
    array = np.arange(nu + ny)
    combinations = []
    for r in range(1, l + 1):
        combinations.extend(list(itertools.combinations_with_replacement(array, r)))

    # Criar matriz de candidatos Psi
    candidatos = np.array([np.prod(linear_psi[:, comb], axis=1) for comb in combinations]).T

    regressor_names = generate_regressor_names(nu, ny, combinations)

    return candidatos, regressor_names


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
