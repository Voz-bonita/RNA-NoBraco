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


# Itens e) & f)

# Novo theta fornecido
theta.n <- rep(0, 9)


# Em preparativo para o item f) guarda-se os dados do custo desde o primeiro theta
treinamento_pred.0 <- prediction(treinamento$x1.obs, treinamento$x2.obs, theta.n)
teste_pred.0 <- prediction(teste$x1.obs, teste$x2.obs, theta.n)

custo.trein.0 <- MSE(treinamento$y, treinamento_pred.0$yhat)
custo.teste.0 <- MSE(teste$y, teste_pred.0$yhat)


# Guarda-se o primeiro custo na linha "0", por isso sao 101 linhas de observacoes e nao 100
grad_custo <- data.frame(
   matrix(ncol = 11, nrow = 101)
)
grad_custo[1,] <- c(theta.n, sum(custo.trein.0), sum(custo.teste.0))
names(grad_custo) <- c("w1","w2","w3","w4","w5","w6","b1","b2","b3","MSE-treino","MSE-teste")


# Definido o Learning Rate, pode-se calcular a descida de gradiente
lr <- 0.1
for (n in 2:101) {
    gradJ.n <- grad(matrix(c(treinamento$x1.obs, treinamento$x2.obs), ncol = 2), theta.n, treinamento$y)
    theta.n <- theta.n - lr*gradJ.n

    treinamento_pred.n <- prediction(treinamento$x1.obs, treinamento$x2.obs, theta.n)
    teste_pred.n <- prediction(teste$x1.obs, teste$x2.obs, theta.n)

    custo.trein.n <- MSE(treinamento$y, treinamento_pred.n$yhat)
    custo.teste.n <- MSE(teste$y, teste_pred.n$yhat)

    grad_custo[n,] <- c(theta.n, sum(custo.trein.n), sum(custo.teste.n))

}
grad_custo <- as_tibble(grad_custo)