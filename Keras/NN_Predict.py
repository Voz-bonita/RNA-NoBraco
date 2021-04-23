from tensorflow.keras.models import load_model
from Load_Data import X_teste
from itertools import combinations
import numpy as np

model = load_model("Dense-2x32-lr-0.003.melhor")

# Item c)
yhat = model.predict(X_teste)
np.savetxt("optim_predict.csv", yhat, delimiter=",")


# Item d)
x = np.array([np.ones(2)])
print("d) Previsao:", model.predict(x))

# A rede tem muitos pesos, cuidado com a bagunca
# print("d) Pesos:", model.get_weights())


# Item e)

# Em R, equivale a
# seq(-3, 3, length.out=n)
n = 100
x1 = x2 = np.array([np.linspace(-3, 3, num=n)])

dados_grid = set(combinations(np.append(x1, x2), r=2))
dados_grid = sorted(dados_grid, key=lambda x: (x[1], x[0]))
dados_grid_arr = np.array([np.array(comb) for comb in dados_grid])


dados_prediction = model.predict(dados_grid_arr)
np.savetxt("Dados-grid-pred.csv", dados_prediction, delimiter=",")
