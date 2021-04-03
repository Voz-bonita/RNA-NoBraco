# Title     : RNA
# Objective : Ajustar uma RNA do jeito dificil
# Created by: Jose Vitor
# Created on: 15/03/2021

library(dplyr)

set.seed(2.2020)
m.obs <- 100000
dados <- tibble(
  x1.obs=runif(m.obs, -3, 3),
  x2.obs=runif(m.obs, -3, 3)) %>%
    mutate(mu=abs(x1.obs^3 - 30*sin(x2.obs) + 10),
    y=rnorm(m.obs, mean=mu, sd=1))


# Carrega todas as funcoes customizadas utilizadas nesse script
source("Functions.R")

# Item a)

## Theta fornecido
theta <- rep(0.1,9)

## Para o vetor x = (1,1) tem-se que
exemplar_pred <- prediction(1,1)

## Resposta ao item a)
exemplar_pred$yhat


## Item b)

treinamento <- dados[1:80000,]
teste <- dados[-1:-80000,]

teste_pred <- prediction(teste$x1.obs, teste$x2.obs)

## Resposta ao item b)
custo <- sum(MSE(teste$y, teste_pred$yhat))
custo


# Item d)

# Resposta ao item d)
gradJ <- grad(matrix(c(treinamento$x1.obs, treinamento$x2.obs), ncol = 2), rep(0.1, 9), treinamento$y)
gradJ


# Itemd e)

# A funcao grad_desc foi definida para otimizar os pesos sobre o conjunto de treino
# Trata-se entao de verificar a descida de gradiente utilizando como base o conjunto de teste no lugar do de treino
grad_desc.teste <- grad_desc(teste, indexes = c(1,2,4), lr = 0.1, theta = rep(0, 9), iterations = 100)

# Resposta ao item e)

# Menor custo obtido durante a descida de gradiente
# Vale lembrar que o conjunto de teste foi utilizado como base
# Por isso mesmo pede-se a coluna `MSE-train`, pois o conjunto de teste esta no lugar do conjunto de treino
min(grad_desc.teste$`MSE-train`)

# Iteracao da ocorrencia
index <- which(grad_desc.teste$`MSE-train` == min(grad_desc.teste$`MSE-train`))
index

# Vetor de pesos estimado
grad_desc.teste[index, 1:9]

# Item f)

grad_desc.normal <- grad_desc(treinamento, teste, indexes = c(1,2,4), lr = 0.1, theta = rep(0, 9), iterations = 100)