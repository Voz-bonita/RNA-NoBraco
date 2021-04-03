# Title     : RNA
# Objective : Ajustar uma RNA do jeito dificil
# Created by: Jose Vitor
# Created on: 15/03/2021

library(dplyr)
library(ggplot2)
library(reshape2)
library(latex2exp)

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

grad_desc.graph <- as_tibble(melt(grad_desc.normal, id.vars=c("Iteracao","w1","w2","w3","w4","w5","w6","b1","b2","b3")))
names(grad_desc.graph)[11:12] <- c("Banco", "Custo")

ggplot(data = grad_desc.graph, aes(x=Iteracao, y=Custo, col=Banco)) +
  geom_line(size=1.5) +
  labs(x = "Iteracao",
       y = "MSE") +
  scale_color_discrete(labels = c("Treino", "Teste")) +
  ggsave("Gradient_desc.png")

# Item g)

index <- which(grad_desc.normal$`MSE-validation` == min(grad_desc.normal$`MSE-validation`))
pesos_otimos <- unlist(grad_desc.normal[index,2:10])

teste_otimizado <- prediction(teste$x1.obs, teste$x2.obs, theta = pesos_otimos)
residuos <- teste$y - teste_otimizado$yhat

teste_final <- tibble(
  x1 = teste$x1.obs,
  x2 = teste$x2.obs,
  y = teste$y,
  yhat = teste_otimizado$yhat,
  residuos = residuos
)

grafico_base <- ggplot(teste_final, aes(x=x1, y=x2)) +
  coord_cartesian(expand=F) +
  theme_dark() +
  xlab(TeX("X_1")) + ylab(TeX("X_2"))

grafico_base +
  geom_point(aes(colour=residuos), size=2, shape=15) +
  scale_colour_gradient(low="springgreen",
                        high="red",
                        name=TeX("Redisuos|(X_1, X_2)")) +
  ggsave("Residuos.png")

grafico_base +
  geom_point(aes(colour=yhat), size=2, shape=15) +
  scale_colour_gradient(low="springgreen",
                        high="red",
                        name=TeX("$f(X_1,X_2\\;|\\;\\theta\\;)$")) +
  ggsave("Previsoes.png")
