import pickle
import os
from pathlib import Path

path = Path(os.getcwd()).parent.absolute()
temp = open(f"{path}/Data/X_treino.pickle", "rb")
X_treino = pickle.load(temp)
temp.close()

temp = open(f"{path}/Data/y_treino.pickle", "rb")
y_treino = pickle.load(temp)
temp.close()

temp = open(f"{path}/Data/X_teste.pickle", "rb")
X_teste = pickle.load(temp)
temp.close()

temp = open(f"{path}/Data/y_teste.pickle", "rb")
y_teste = pickle.load(temp)
temp.close()
