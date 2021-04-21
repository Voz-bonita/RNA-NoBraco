import numpy as np
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam, Nadam, Ftrl, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from Load_Data import X_treino, X_teste, y_treino, y_teste

# Algumas possiveis combinacoes de modelos
h_layers = range(1, 4)
neurons = [1, 2, 4, 16, 32, 64]

# Prepara os callbacks de parada
Estop = EarlyStopping(monitor="loss", patience=8)


for neuron in neurons:
    for layers in h_layers:

        # Cria-se uma pasta especificando o modelo
        nome = f"Dense-{layers}-{neuron}"
        TenBoard = TensorBoard(log_dir=f"logs/{nome}")

        # Cria-se um novo modelo
        model = Sequential()
        model.add(Dense(neuron, activation="sigmoid", input_shape=X_treino.shape[1:]))

        # Adiciona "layer" camadas escondidas
        for layer in range(layers):
            model.add(Dense(neuron, activation="relu"))

        model.add(Dense(1, activation="linear"))

        model.compile(loss="mse",
                      optimizer=SGD(learning_rate=0.01))

        model.fit(X_treino, y_treino,
                  batch_size=64,
                  epochs=100,
                  callbacks=[TenBoard, Estop],
                  validation_data=(X_teste, y_teste))
