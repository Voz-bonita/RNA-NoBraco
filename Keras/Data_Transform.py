import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

# Saindo dos scripts em R dispom-se de dados em .csv
# Surge o objetivo de guardar esses dados de forma comoda para uso posterior
path = Path(os.getcwd()).parent.absolute()


# Os indices sao carregados como colunas, logo,
# devem ser descartados
dados = pd.read_csv(f"{path}/Data/Dados.csv")
dados.drop(dados.columns[0], axis=1, inplace=True)


# Input e respectivo valor observado
divisor = 80000
X_treino = np.array(dados[dados.columns[0:2]][:divisor])
y_treino = np.array(dados[dados.columns[3]][:divisor])

X_teste = np.array(dados[dados.columns[0:2]][divisor:])
y_teste = np.array(dados[dados.columns[3]][divisor:])

pickle.dump(X_treino, open(f"{path}/Data/X_treino.pickle", "wb"))
pickle.dump(y_treino, open(f"{path}/Data/y_treino.pickle", "wb"))
pickle.dump(X_teste, open(f"{path}/Data/X_teste.pickle", "wb"))
pickle.dump(y_teste, open(f"{path}/Data/y_teste.pickle", "wb"))
