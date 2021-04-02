library(dplyr)

# Item a)
# Funcao de ativacao
sigmoid <- function(x) {
    activation <- 1 / (1 + exp(-x) )
    returnValue(activation)
}

# Predicao basica
prediction <- function(x1, x2, theta = rep(0.1, 9)) {
    x <- c(x1, x2)
    n <- length(x)/2
    W1 <- matrix(theta[1:4], ncol = 2, nrow = 2)
    W2 <- matrix(theta[5:6], ncol = 1, nrow = 2)
    B1 <- matrix(theta[7:8], ncol = 2, nrow = n)
    b3 <- last(theta)

    X <- matrix(x, ncol = 2)

    a <- X %*% W1 + B1
    h <- sigmoid(a)
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

# Item b)

# Funcao de custo
MSE <- function (y, yhat) {
    custo <-(y-yhat)^2/length(y)
    returnValue(custo)
}

# Item d)

# Ao inves de calcular o MSE e aplicar a raiz quadrada, calcula-se o "erro medio"
ME <- function (y, yhat) {
    custo <- (y-yhat)/length(y)
    returnValue(custo)
}
# A necessidade do "erro medio" surge da derivada parcial do custo com respeito à variavel preditoria

grad <- function(X, theta, y) {

    pred <- prediction(X[,1], X[,2], theta)

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
    nabla <- c(dJdW1, dJdW2, dJdW3, dJdW4, dJdW5, dJdW6, dJdB1, dJdB2, dJdB3)
    return(nabla)
}
