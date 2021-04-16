import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from Load_Data import X_treino, X_teste, y_treino, y_teste

model = Sequential()

model.add(Dense(16, activation="tanh", input_shape=X_treino.shape[1:]))

model.add(Dense(16, activation="sigmoid"))

model.add(Dense(1, activation="linear"))

model.compile(loss="mse",
              optimizer=Adam(learning_rate=0.1))

model.fit(X_treino, y_treino,
          batch_size=64,
          epochs=100,
          callbacks=[tensorboard],
          validation_data=(X_teste, y_teste))

