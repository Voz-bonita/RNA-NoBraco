# Title     : RNA
# Objective : Ajustar uma RNA do jeito dificil
# Created by: Jose Vitor
# Created on: 15/03/2021

library(dplyr)
library(ggplot2)
library(reshape2)
library(latex2exp)
library(microbenchmark)

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
gradJ_treino <- grad(matrix(c(treinamento$x1.obs, treinamento$x2.obs), ncol = 2), rep(0.1, 9), treinamento$y)
gradJ_treino


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
  geom_line(size=3.0) +
  labs(x = "Iteracao",
       y = "MSE") +
  scale_color_discrete(labels = c("Treino", "Teste")) +
  ggsave("Gradient_desc.png")


# Item g)

# Toma-se o melhor vetor de pesos obtido
index <- which(grad_desc.normal$`MSE-validation` == min(grad_desc.normal$`MSE-validation`))
pesos_otimos <- unlist(grad_desc.normal[index,2:10])

teste_otimizado <- prediction(teste$x1.obs, teste$x2.obs, theta = pesos_otimos)
residuos <- teste$y - teste_otimizado$yhat


# Conjunto de teste contendo valores iniciais e previsoes pos otimizacao
teste_final <- tibble(
  x1 = teste$x1.obs,
  x2 = teste$x2.obs,
  y = teste$y,
  yhat = teste_otimizado$yhat,
  residuos = residuos
)

grafico_res <- ggplot(teste_final, aes(x=x1, y=x2)) +
  coord_cartesian(expand=F) +
  geom_point(aes(colour=residuos), size=2, shape=15) +
  scale_colour_gradient(low="springgreen",
                        high="red",
                        name=TeX("Redisuos|(Y, \\hat{Y})")) +
  theme_dark() +
  xlab(TeX("X_1")) + ylab(TeX("X_2"))

grafico_res + ggsave("Residuos.png")


# Item h)

YxYhat <- ggplot(teste_final, aes(x=yhat, y=y)) +
  geom_point() +
  xlab(TeX("$\\hat{y}$")) + ylab(TeX("y"))

YxYhat + ggsave("YxYhat.png")


# Item i)

k <- 300
dJdw1 <- data.frame(matrix(ncol = 2, nrow = k))
names(dJdw1) <- c("k", "Derivada")
for (i in 1:k) {
  amostra <- dados[1:i,]
  gradiente <- grad(matrix(c(amostra$x1.obs, amostra$x2.obs), ncol = 2), theta = rep(0.1,9), amostra$y)
  dJdw1[i,] <- c(i, gradiente[1])
}
dJdw1 <- as_tibble(dJdw1)

grad_xk <- ggplot(data = dJdw1, aes(x=k, y=Derivada)) +
  geom_line(color="white") +
  geom_hline(aes(yintercept = gradJ_treino[1]), color="red") +
  annotate(x=255, y=-0.07, label=TeX("J_{w_1} = -0.170096"),
           geom = "text", angle = 0, vjust = 1, hjust=-0.1, size = 3.5)+
  theme_dark() +
  xlab("k amostra(s)") + ylab(TeX("J_{w_1}"))

grad_xk + ggsave("GradW1xK.png")

# Note que a amostra de tamanho 100000 trata-se do proprio banco original
amostra_300 <- dados[1:300,]

benchmark <- microbenchmark(grad(matrix(c(amostra_300$x1.obs, amostra_300$x2.obs), ncol = 2), theta = rep(0.1,9), amostra_300$y),
                            grad(matrix(c(dados$x1.obs, dados$x2.obs), ncol = 2), theta = rep(0.1,9), dados$y))
benchmark

# Item j)

mod.lm1 <- lm(y ~ 1 + x1.obs + x2.obs, data = treinamento)
betas_mod1 <- mod.lm1$coefficients
names(betas_mod1) <- c("B0", "B1", "B2")

# A funcao lm juntaria os argumentos semelhantes para calcular os coeficientes
# A alternativa encontrada foi criar artifialmente covariaveis quadraticas
treinamento_especial <- treinamento %>%
  mutate(x1sq = x1.obs^2,
         x2sq = x2.obs^2)
mod.lm2 <- lm(y ~ 1 + 1*x1.obs + x2.obs + x1sq + x2sq + x1.obs*x2.obs, data = treinamento_especial)

# Aqui e necessario cuidado ao mudar os nomes do coeficientes
# Os coeficientes nao aparecem necessariamente na ordem da formula
betas_mod2 <- mod.lm2$coefficients
names(betas_mod2) <- c("B0", "B2", "B3", "B4", "B1", "B5")


# Custo no banco de teste
mod.lm1_pred <- linear1(teste$x1.obs, teste$x2.obs, betas_mod1[["B0"]], unname(betas_mod1[c("B1","B2")]))
custo.lm1 <- sum(MSE(teste$y, mod.lm1_pred))

mod.lm2_pred <- linear2(teste$x1.obs, teste$x2.obs, betas_mod2[["B0"]], unname(betas_mod2[c("B1","B2","B3","B4","B5")]))
custo.lm2 <- sum(MSE(teste$y, mod.lm2_pred))


custo.lm1
custo.lm2
min(grad_desc.normal$`MSE-train`)


# Item l)

# Rede neural
sigmahat <- sqrt(min(grad_desc.normal$`MSE-validation`))
captura_obs(medias = teste_otimizado$yhat, sd = sigmahat, obs = teste$y)


# Os modelos lineares foram treinados sobre o conjunto de treino
# Vale observar seus erros amostrais para calcular os novos intervalos de confianca
lm1.conf <- as_tibble(predict(mod.lm1, interval = "prediction"))
ErroAmostral.lm1 <- mean(lm1.conf$fit - lm1.conf$lwr)
sd.lm1 <- ErroAmostral.lm1/qnorm(0.975)

captura_obs(medias = mod.lm1_pred, sd = sd.lm1, obs = teste$y)


lm2.conf <- as_tibble(predict(mod.lm2, interval = "prediction"))
ErroAmostral.lm2 <- mean(lm2.conf$fit - lm2.conf$lwr)
sd.lm2 <- ErroAmostral.lm2/qnorm(0.975)

captura_obs(medias = mod.lm2_pred, sd = sd.lm2, obs = teste$y)