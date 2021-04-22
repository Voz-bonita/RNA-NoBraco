from tensorflow.keras.models import load_model
from Load_Data import X_treino, X_teste, y_treino, y_teste
import numpy as np

model = load_model("Dense-2x32-lr-0.003.melhor")
yhat = model.predict(X_teste)

print(len(yhat))
np.savetxt("optim_predict.csv", yhat, delimiter=",")

