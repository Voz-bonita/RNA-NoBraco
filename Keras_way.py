import pandas as pd
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Os indices sao carregados como colunas, logo,
# devem ser descartados
dados = pd.read_csv("Dados.csv")
dados.drop(dados.columns[0], axis=1, inplace=True)

# Input e respectivo valor observado
divisor = 80000
X_treino = np.array(dados[dados.columns[0:2]][:divisor])
y_treino = np.array(dados[dados.columns[3]][:divisor])

X_teste = np.array(dados[dados.columns[0:2]][divisor:])
y_teste = np.array(dados[dados.columns[3]][divisor:])


# Pesos e vieses fornecidos
W1_Beta1 = [np.empty(shape=(2, 2), dtype=np.float32), np.empty(shape=2, dtype=np.float32)]
W1_Beta1[0][0] = 0
W1_Beta1[0][1] = 0
W1_Beta1[1][0] = 0

W2_Beta3 = [np.empty(shape=(2, 1), dtype=np.float32), np.empty(shape=1, dtype=np.float32)]
W1_Beta1[0][0] = 0
W1_Beta1[1][0] = 0


modelo = Sequential()

modelo.add(Dense(2, input_shape=X_treino.shape,
                 activation="sigmoid"))
modelo.layers[0].set_weights(W1_Beta1)


modelo.add(Dense(1, activation="linear"))
modelo.layers[1].set_weights(W2_Beta3)


modelo.compile(loss="mse",
               optimizer=SGD(learning_rate=0.1))

# Note o batch_size
# Assim como no caso anterior, utiliza-se o banco inteiro
modelo.fit(X_treino, y_treino,
           batch_size=len(X_treino),
           epochs=100,
           validation_data=(X_teste, y_teste))

modelo.save("2x1-Dense.model")
