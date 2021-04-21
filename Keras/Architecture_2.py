from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from Load_Data import X_treino, X_teste, y_treino, y_teste

Estop = EarlyStopping(monitor="loss", patience=8)

# Melhores quantidade de neuronios
# neurons = [16, 32, 64]
# Pos comparacao de lr
neurons = [64]

# Taxas de aprendizado variadas em torno de 0.01
# lr_s = [0.01, 0.03, 0.005, 0.001]
# Pos comparacao de arquitetura + lr
lr_s = [0.01, 0.03]

for lr in lr_s:
    for neuron in neurons:

        nome = f"Dense-1-{neuron}-{lr}"
        TenBoard = TensorBoard(log_dir=f"Optim/{nome}")

        model = Sequential()

        model.add(Dense(neuron, activation="sigmoid", input_shape=X_treino.shape[1:]))

        model.add(Dense(1, activation="linear"))

        model.compile(loss="mse",
                      optimizer=SGD(lr=lr))

        model.fit(X_treino, y_treino,
                  batch_size=64,
                  epochs=100,
                  callbacks=[Estop, TenBoard],
                  validation_data=(X_teste, y_teste))
