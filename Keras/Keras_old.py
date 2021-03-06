import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from Load_Data import X_treino, X_teste, y_treino, y_teste
import time

# Callback para salvar apenas o melhor modelo
MCP_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# Pesos e vieses de partida
W1_Beta1 = [np.empty(shape=(2, 2), dtype=np.float32), np.empty(shape=2, dtype=np.float32)]
W1_Beta1[0][0] = 0
W1_Beta1[0][1] = 0
W1_Beta1[1][0] = 0

W2_Beta3 = [np.empty(shape=(2, 1), dtype=np.float32), np.empty(shape=1, dtype=np.float32)]
W2_Beta3[0][0] = 0
W2_Beta3[1][0] = 0


TenBoard = TensorBoard(log_dir=f"logs/Old-Dense-2x1")

# Vamos fazer essa operacao 20x para obter uma especie de benchmark
tempo_total = 0
i_benchmark = 1
for i in range(i_benchmark):
    modelo = Sequential()

    modelo.add(Dense(2, input_shape=X_treino.shape,
                     activation="sigmoid"))
    modelo.layers[0].set_weights(W1_Beta1)

    modelo.add(Dense(1, activation="linear"))
    modelo.layers[1].set_weights(W2_Beta3)

    modelo.compile(loss="mse",
                   optimizer=SGD(learning_rate=0.1))

    # Marcando o tempo mais precisamente


    inicio = time.time()

    # Note o batch_size
    # Assim como no caso anterior, utiliza-se o banco inteiro
    modelo.fit(X_treino, y_treino,
               batch_size=len(X_treino),
               epochs=100,
               validation_data=(X_teste, y_teste),
               callbacks=[TenBoard, MCP_save])

    fim = time.time()
    tempo_total += fim-inicio

print(f"Demorou-se em media {tempo_total/i_benchmark:.2f} segundos")

# Apenas o melhor modelo e seu vetor de pesos
model = load_model(".mdl_wts.hdf5")
print(*model.get_weights())
