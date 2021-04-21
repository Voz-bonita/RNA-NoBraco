from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from Load_Data import X_treino, X_teste, y_treino, y_teste

neurons = [32, 64]
lr_s = [0.003]

for lr in lr_s:
    for neuron in neurons:

        nome = f"Dense-2x{neuron}-lr-{lr}"
        TenBoard = TensorBoard(log_dir=f"Optim/{nome}")
        MCP_save = ModelCheckpoint(f'{nome}.melhor', save_best_only=True, monitor='val_loss', mode='min')

        model = Sequential()

        model.add(Dense(neuron, activation="tanh", input_shape=X_treino.shape[1:]))
        model.add(Dense(neuron, activation="relu"))

        model.add(Dense(1, activation="linear"))

        model.compile(loss="mse",
                      optimizer=SGD(lr=lr))

        model.fit(X_treino, y_treino,
                  batch_size=64,
                  epochs=100,
                  callbacks=[TenBoard, MCP_save],
                  validation_data=(X_teste, y_teste))



