import pandas as pd

df = pd.read_csv('C:/Users/Sofia/Documents/Projeto/results/model_results.csv')

min_mse_free = df.loc[df.groupby('system')['mse_free'].idxmin()]
min_mse_onestep = df.loc[df.groupby('system')['mse_onestep'].idxmin()]

result = pd.concat([min_mse_free, min_mse_onestep])

result = result.reset_index(drop=True)

result.to_csv('C:/Users/Sofia/Documents/Projeto/results/best_results.csv')
