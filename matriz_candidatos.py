import itertools
import numpy as np

def matriz_candidatos(input, output, nu, ny, l):
    n = len(input)
    grau = max(nu, ny)
    linear_psi = np.zeros((n - grau, nu + ny))

    # Create candidate regressors using input and output delays
    for i in range(ny):
        linear_psi[:, ny - 1 - i] = output[i + 1:n - grau + i + 1]
    for i in range(nu):
        linear_psi[:, ny + nu - 1 - i] = input[i + 1:n - grau + i + 1]

    # Generate combinations of terms up to degree l
    array = np.arange(nu + ny)
    combinations = find_combinations(array, l)
    M = len(combinations)

    # Create the candidate matrix
    candidatos = np.array([np.prod(linear_psi[:, comb], axis=1) for comb in combinations]).T

    return candidatos, M, combinations


def find_combinations(arr, l):
    combinations = []
    for r in range(1, l + 1):
        combinations.extend(list(itertools.combinations_with_replacement(arr, r)))
    return combinations