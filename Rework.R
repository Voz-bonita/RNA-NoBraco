library(dplyr)
library(plyr)
library(tibble)


set.seed(2.2020)
m.obs <- 100000
dados <- tibble(
  x1.obs=runif(m.obs, -3, 3),
  x2.obs=runif(m.obs, -3, 3)) %>%
  mutate(mu=abs(x1.obs^3 - 30*sin(x2.obs) + 10),
         y=rnorm(m.obs, mean=mu, sd=1))

corte <- 80000
treinamento <- dados[1:corte,]
teste <- dados[(corte+1):(nrow(dados)),]

sigmoid <- function(x) {
  activation <- 1 / (1 + exp(-x))
  returnValue(activation)
}

MSE <- function (y, yhat) {
  custo <-sum((y-yhat)^2)/length(y)
  returnValue(custo)
}

ME <- function (y, yhat) {
  custo <- (y-yhat)/length(y)
  returnValue(custo)
}

# Forward_propagation com dropout
forward_prop <- function(x, theta = rep(0.1, 9), dropout = rep(T, 4)) {
  ifelse(is.data.frame(x), x <- matrix(unlist(x), ncol = 2), x <- matrix(x, ncol = 2))
  n <- nrow(x)
  W1 <- matrix(theta[1:4], ncol = 2, nrow = 2)
  W2 <- matrix(theta[5:6], ncol = 1, nrow = 2)
  B1 <- matrix(theta[7:8], ncol = 2, nrow = n)
  b3 <- last(theta)

  ### Dropout acontecendo
  ## Zerar os neuronios altera o back-propagation
  ## Zeram-se os pesos
  if (!dropout[1]) W1[,1] <- 0
  if (!dropout[2]) W1[,2] <- 0


  a <- x %*% t(W1) + B1
  h <- sigmoid(a)

  if (!dropout[3]) {W2[1,] <- 0}
  if (!dropout[4]) {W2[2,] <- 0}

  yhat <- h %*% W2 + b3

  predict_df <- tibble(
    a1 = a[,1],
    a2 = a[,2],
    h1 = h[,1],
    h2 = h[,2],
    yhat = yhat[,1]
  )

  returnValue(predict_df)
}

back_prop <- function(X, theta, y, dropout = rep(T, 4)) {
  X <- matrix(X, ncol = 2)
  pred <- forward_prop(X, theta = theta, dropout = dropout)


  dCdY <- -2*ME(y, pred$yhat)
  dYdH1 <- theta[5]
  dYdH2 <- theta[6]


  # w1, w3, w5, b1
  K1 <- dCdY * pred$h1
  tau1 <- K1 * (1-pred$h1) * dYdH1

  dJdW5 <- sum(K1)
  dJdW3 <- sum(tau1 * X[,2])
  dJdW1 <- sum(tau1 * X[,1])
  dJdB1 <- sum(tau1)

  # w2, w4, w6, b2
  K2 <- dCdY * pred$h2
  tau2 <- K2 * (1-pred$h2) * dYdH2

  dJdW6 <- sum(K2)
  dJdW4 <- sum(tau2 * X[,2])
  dJdW2 <- sum(tau2 * X[,1])
  dJdB2 <- sum(tau2)

  # b3
  dJdB3 <- sum(dCdY)

  # Gradiente obtido
  grad <- c(dJdW1, dJdW2, dJdW3, dJdW4, dJdW5, dJdW6, dJdB1, dJdB2, dJdB3)
  return(grad)
}


grad_desc <- function(X_treino, y_treino,
                      X_teste = matrix(0, ncol = 2), y_teste = matrix(0, ncol = 2),
                      lr = 0.1, theta, epochs,
                      dropout.ratio = 1){


  train_pred <- forward_prop(X_treino, theta)
  val_pred <- forward_prop(X_teste, theta)

  cost.train <- MSE(y_treino, train_pred$yhat)
  cost.val <- MSE(y_teste, val_pred$yhat)

  # Note mais a frente que o custo da validacao estabelece dominancia
  menor.val.cost <- cost.val
  menor.train.cost <- cost.train
  melhor.peso <- theta
  melhor.epoch <- 0

  for (epoch in 1:epochs) {

    dropout <- rbinom(4,1,0.6)
    gradJ <- back_prop(X = X_treino, theta = theta, y = y_treino, dropout = dropout)

    theta <- theta - lr*gradJ

    # Calculam-se a n-esimas previsoes e seus custos
    train_pred.n <- forward_prop(X_treino, theta, dropout = dropout)
    val_pred.n <- forward_prop(X_teste, theta, dropout = dropout)

    train_cost.n <- MSE(y_treino, train_pred.n$yhat)
    val_cost.n <- MSE(y_teste, val_pred.n$yhat)

    # Aqui esta a dominancia da validacao
    if (val_cost.n < menor.val.cost) {
      menor.val.cost <- val_cost.n
      menor.train.cost <- train_cost.n
      melhor.peso <- theta
      melhor.epoch <- epoch
    }
  }
  melhor.modelo <- list(menor.train.cost, menor.val.cost, melhor.peso, melhor.epoch)
  names(melhor.modelo) <- c("MSE_treino", "MSE_val", "Theta", "Epoch")
  names(melhor.modelo$Theta) <- c("w1","w2","w3",
                                  "w4","w5","w6",
                                  "b1","b2","b3")
  returnValue(melhor.modelo)

}

dropout_opt <- grad_desc(c(treinamento$x1.obs, treinamento$x2.obs), treinamento$y,
                          c(teste$x1.obs, teste$x2.obs), teste$y,
                          lr = 0.1,
                          theta = rep(0, 9),
                          epochs = 100,
                          dropout = 0.4)


# Item b)

familia_gen <- function(X, theta, n.previsoes, dropout.ratio = 0.6) {
  X <- as.vector(X)

  ## Memoria X Velocidade
  # Redes repetidas levam a operações redundantes
  dropout <- rbinom(n.previsoes*4,1,dropout.ratio)

  dropout_df <- matrix(dropout, ncol = 4, nrow = 200) %>%
    as_tibble(.name_repair = "minimal")
  names(dropout_df ) <- c("D1", "D2", "D3", "D4")

  # Redes unicas contadas
  dropout_df <- ddply(dropout_df, .(D1, D2, D3, D4), nrow) %>%
    add_column(cumsum(.[,5]))


  # Familia de previsoes
  familia <- numeric(n.previsoes)

  unicas <- nrow(dropout_df)
  for (i in 1:(unicas)) {
    # Previsao unica ocorre uma unica vez
    previsao <- forward_prop(X, theta=theta, dropout=dropout_df[i,1:4])$yhat

    # Onde e quantas previsoes alocar
    reps <- dropout_df[i,5]
    ocupado <- dropout_df[i-1,6]

    if (i == 1) ocupado <- 0

    familia[(ocupado+1):(ocupado+reps)] <- rep(previsao, reps)
  }
  return(familia)
}

# Pesos obtidos no item anterior
theta <- dropout_opt$Theta

X_teste <- teste %>%
  select(x1.obs, x2.obs)

familia <- familia_gen(X_teste[1,], theta, 200)
estimativa <- mean(familia)
estimativa

# (IC) Limite inferior e superior respectiva
paste(quantile(familia, 0.025), quantile(familia, 0.975))


# Item c)

# n.est <- nrow(X_teste)
# estimativas_treino <- numeric(n.est)
# for (i in 1:n.est) {
#   print(i)
#   familia <- familia_gen(X_teste[i,], theta, 200)
#   estimativa.pontual <- mean(familia)
#   estimativas_treino[i] <- estimativa.pontual
# }
#
# Salva as previsoes para nao precisar executar duas vezes
# write.csv(estimativas_treino, "estimativas.csv")


# Item d)

WSIR <- function (X, theta, dropout.ratio) {
  theta[1:8] <- theta[1:8]*dropout.ratio
  previsao <- forward_prop(X, theta)$yhat
  return(previsao)
}

previsao <- WSIR(X_teste, theta, 0.6)

estimativas_c <- read.csv("estimativas.csv")
MSE(teste$y, estimativas_c[,2])
MSE(teste$y, previsao)