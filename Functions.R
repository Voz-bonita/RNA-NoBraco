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


# Itens e) & f)

grad_desc <- function(train_db, validation_db = FALSE, indexes, lr = 0.1, theta, epochs) {

    train_db <- as.data.frame(train_db)
    # Um conjunto de validacao nao e estritamente necessario
    # Se nao for desejado, entao sera um data frame vazio que permite a descida de gradiente no conjunto de treino
    if (is.logical(validation_db)) {
        validation_db <- data.frame(matrix(ncol = ncol(train_db), nrow = nrow(train_db)))
    }
    validation_db <- as.data.frame(validation_db)

    # Usuario deve informar em quais colunas do banco se encontram os inputs e o valor esperado
    x1.col <- indexes[[1]]
    x2.col <- indexes[[2]]
    y.col <- indexes[[3]]


    train_db_pred <- prediction(train_db[,x1.col], train_db[,x2.col], theta)
    validation_db_pred <- prediction(validation_db[,x1.col], validation_db[,x2.col], theta)

    cost.train.0 <- MSE(train_db[,y.col], train_db_pred$yhat)
    cost.validation.0 <- MSE(unlist(validation_db[,y.col]), validation_db_pred$yhat)


    # Guarda-se o primeiro custo na linha "0", por isso sao iteracaoes+1 linhas de observacoes e nao iteracoes linhas
    grad_custo <- data.frame(
       matrix(ncol = 12, nrow = epochs+1)
    )
    grad_custo[1,] <- c(0, theta, sum(cost.train.0), sum(cost.validation.0))
    names(grad_custo) <- c("Iteracao","w1","w2","w3","w4","w5","w6","b1","b2","b3","MSE-train","MSE-validation")


    for (n in 2:(epochs+1)) {
        # Calcula-se o n-esimo gradiente sobre o conjunto de treino
        # E em seguida o n-esimo theta
        gradJ.n <- grad(matrix(c(train_db[,x1.col], train_db[,x2.col]), ncol = 2), theta, train_db$y)
        theta <- theta - lr*gradJ.n

        # Calculam-se a n-esimas previsoes e seus custos
        train_db_pred.n <- prediction(train_db[,x1.col], train_db[,x2.col], theta)
        validation_db_pred.n <- prediction(validation_db[,x1.col], validation_db[,x2.col], theta)

        train_cost.n <- MSE(train_db[,y.col], train_db_pred.n$yhat)
        validation_cost.n <- MSE(validation_db[,y.col], validation_db_pred.n$yhat)

        grad_custo[n,] <- c(n-1, theta, sum(train_cost.n), sum(validation_cost.n))
    }
    returnValue(as_tibble(grad_custo))

}


# Item j)

# Denotando por beta_1 o vetor dos coeficientes beta sem o intercepto
# Note que o vetor beta_1 deve estar ordenado pelo indice do coeficiente

linear1 <- function(x1, x2, intercepto, beta_1) {
    warning("O primeiro elemento do vetor beta_1 deve ser o termo Beta1.\n",
            "  Se ja estiver fazendo uso correto da funcao, ignore-me.")

    X <- matrix(c(x1, x2), ncol = 2)
    yhat <- X %*% beta_1 + intercepto
    returnValue(yhat)
}


linear2 <- function(x1, x2, intercepto, beta_1) {
    warning("O primeiro elemento do vetor beta_1 deve ser o termo Beta1.\n",
            "  O segundo elemento do vetor beta_1 deve ser o termo Beta2.\n",
            "  E assim sucessivamente... Se ja estiver fazendo uso correto da funcao, ignore-me.")

    X <- matrix(c(x1, x2), ncol = 2)
    Xsq <- matrix(c(x1^2, x2^2), ncol = 2)

    yhat <- X %*% beta_1[1:2] + Xsq %*% beta_1[3:4] + beta_1[5]*x1*x2 + intercepto
    returnValue(yhat)

}


# Item l)

captura_obs <- function(medias, sd, conf_lvl = 0.95, obs) {

    conf_lateral_inf <- (1-conf_lvl)/2
    erro <- sd*abs(qnorm(conf_lateral_inf))

    lwr <- medias - erro
    upr <- medias + erro

    # Os intervalos de confianca estao em forma vetorial
    # Para agilizar o processo vamos vetorizar a funcao between
    between <- Vectorize(between)
    contem <- between(obs, lwr, upr)

    # Lembrando-se que nos niveis mais baixos da programacao 1 = TRUE e 0 = FALSE
    media <- sum(contem)/length(contem)
    returnValue(media)
}
