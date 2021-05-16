library(pacman)
p_load(ggplot2, dplyr, latex2exp, tibble, rlist, knitr)

#### a)
## Funcao fornecida
fbase <- function(x1,x2) {
  return(x1^4 + x2^4 + x1^2*x2 + x1*x2^2 - 20*x1^2 - 15*x2^2)
}

## 1 milhao de pontos no intervalo [-5,5]
n <- 100
x1 <- seq(-5, 5, length.out=n)
x2 <- seq(-5, 5, length.out=n)
x.grid <- expand.grid(x1, x2)

## Dataframe para o grafico na sequencia
fbase_df <- as_tibble(x.grid) %>%
  rename_all( ~ c("x1", "x2")) %>%
  mutate(z = fbase(x1, x2))


### Preparativos para o grafico

## Limites da escala do contorno
inf <- floor(min(fbase_df$z))
sup <- ceiling(max(fbase_df$z))

## Ponto de minimo global
minimo <- fbase_df[which(fbase_df$z == min(fbase_df$z)),]

### Grafico de contorno
ggplot(fbase_df) +
  geom_contour_filled(aes(x=x1, y=x2, z=z),
                      breaks = seq(inf, sup, 30)) +
  geom_point(aes(x=minimo$x1, y=minimo$x2),
             size=2, color="red") +
  annotate("text", x=minimo$x1, y=minimo$x2 + 0.5,
           label="Minimo global", color="red") +
  theme(panel.grid=element_blank(),
        panel.background=element_rect(fill = "transparent",colour = NA),
        panel.border=element_blank(),
        legend.position = "bottom") +
  guides(fill=guide_legend(nrow=4, byrow=TRUE)) +
  scale_x_continuous(breaks = seq(-5,5,5)) +
  scale_y_continuous(breaks = seq(-5,5,5)) +
  xlab(TeX("$X_1$")) + ylab(TeX("$X_2$"))


#### c)

grad_desc2 <- function(x1, x2, lr = 0.01, epochs = 10) {
  x1 <- as.double(x1)
  x2 <- as.double(x2)

  if (x1 == 0 & x2 == 0) {
    warning(paste0("O ponto inicial (0,0) zera o gradiente\n",
                   "impossibilitando a descida de gradiente."))
  }

  ## Valor inicial
  y0 <- fbase(x1,x2)
  menor <- y0
  melhor_ponto <- c(x1,x2)
  melhor_epoch <- 0

  for (epoch in 1:epochs) {
    # Gradiente
    ddx1 <- 4*x1^3 + 2*x1*x2 + x2^2 - 40*x1
    ddx2 <- 4*x2^3 + 2*x1*x2 + x1^2 - 30*x2

    # Atualizacao das variaveis
    x1 <- x1 - lr*ddx1
    x2 <- x2 - lr*ddx2

    # Atualizacao do valor minimo
    yi <- fbase(x1,x2)
    if (is.nan(yi)) {
      warning("A funcao divergiu e o treinamento foi interrompido")
      break
    } else if (yi < menor) {
      menor <- yi
      melhor_ponto <- c(x1,x2)
      melhor_epoch <- epoch
    }
  }

  melhor_resultado <- c("f(x1,x2)" = menor,
                        "x1" = melhor_ponto[1],
                        "x2" = melhor_ponto[2],
                        "Epoch" = melhor_epoch)

  return(melhor_resultado)
}

grad_desc2(x1 = 1, x2 = 1)
grad_desc2(x1 = 1, x2 = -1)
grad_desc2(x1 = -3, x2 = -3, lr = 0.1)


#### d)

grad_desc2(x1 = 0, x2 = 5, lr = 0.01, epochs = 100)


#### e)

### Potencias de 10
rates <- 10^(0:-4)

resultados <- list()
for (lr in rates) {
  resultados <- list.append(resultados,
                            grad_desc2(x1 = 0, x2 = 5,
                                       lr = lr, epochs = 100))
}

names(resultados) <- as.character(rates)
resultados <- as_tibble(resultados)

resultados


#### f)

grad_desc_pathwise <- function(x1, x2, lr = 0.01, epochs = 10) {
  x1 <- as.double(x1)
  x2 <- as.double(x2)

  if (x1 == 0 & x2 == 0) {
    warning(paste0("O ponto inicial (0,0) zera o gradiente\n",
                   "impossibilitando a descida de gradiente."))
  }

  ## Guarda todo o caminho percorrido pela funcao
  caminho_df <- tibble(x1 = numeric(101),
         x2 = numeric(101),
         f = numeric(101))
  caminho_df[1,] <- list(x1, x2, fbase(x1,x2))

  for (epoch in 1:epochs) {
    # Gradiente
    ddx1 <- 4*x1^3 + 2*x1*x2 + x2^2 - 40*x1
    ddx2 <- 4*x2^3 + 2*x1*x2 + x1^2 - 30*x2

    # Atualizacao das variaveis
    x1 <- x1 - lr*ddx1
    x2 <- x2 - lr*ddx2

    # Atualizacao do valor minimo
    yi <- fbase(x1,x2)
    if (is.nan(yi)) {
      warning("A funcao divergiu e o treinamento foi interrompido")
      break
    } else{
      caminho_df[(epoch+1),] <- list(x1,x2, yi)
    }
  }

  return(caminho_df)
}


set.seed(123)

## Primeira linha apenas para identificar colunas
caminhos <- tibble(x1 = 0, x2 = 0, Tentativa = 0)
for (i in 1:20) {
  ## x1 e x2 uniformemente distribuidos em [-5,5]
  x <- runif(2, -5,5)

  caminho_individual <- grad_desc_pathwise(x[1], x[2],
                                           lr = 0.01,
                                           epochs = 100)[1:2] %>%
    add_column(Tentativa = rep(i,101))

  caminhos <- caminhos %>%
    add_row(caminho_individual)
}

## Remove a primeira linha artificial
caminhos <- caminhos[2:nrow(caminhos),]

## Garante que os caminhos são agrupaveis por tentativa
caminhos$Tentativa <- as.factor(caminhos$Tentativa)

ggplot(fbase_df) +
  geom_contour_filled(aes(x=x1, y=x2, z=z),
                      breaks = seq(inf, sup, 30)) +
  geom_line(data = caminhos, aes(x=x1, y=x2,
                                 group = Tentativa,
                                 color = Tentativa),
            size = 1.2) +
  theme(panel.grid=element_blank(),
        panel.background=element_rect(fill = "transparent",colour = NA),
        panel.border=element_blank(),
        legend.position = "bottom") +
  guides(fill=FALSE,
         color=guide_legend(nrow=2, byrow=TRUE)) +
  scale_x_continuous(breaks = seq(-5,5,5)) +
  scale_y_continuous(breaks = seq(-5,5,5)) +
  xlab(TeX("$X_1$")) + ylab(TeX("$X_2$"))


#### g)

SGD_momentum <- function(x, vel, lr = 0.01, momentum = 0.1, epochs = 10) {
  x <- as.vector(x)
  x1 <- x[1]
  x2 <- x[2]

  if (x1 == 0 & x2 == 0) {
    warning(paste0("O ponto inicial (0,0) zera o gradiente\n",
                   "impossibilitando a descida de gradiente."))
  }

  ## Guarda todo o caminho percorrido pela funcao
  caminho_df <- tibble(x1 = numeric(101),
         x2 = numeric(101),
         f = numeric(101),
         epoch = numeric(101))
  caminho_df[1,] <- list(x1, x2, fbase(x1,x2), 0)

  for (epoch in 1:epochs) {
    # Gradiente
    ddx1 <- 4*x1^3 + 2*x1*x2 + x2^2 - 40*x1
    ddx2 <- 4*x2^3 + 2*x1*x2 + x1^2 - 30*x2
    grad <- c(ddx1, ddx2)

    # Inercia
    vel <- momentum*vel - lr*grad

    # Atualizacao das variaveis
    x <- x + vel
    x1 <- x[1]
    x2 <- x[2]

    # Atualizacao do valor minimo
    yi <- fbase(x1,x2)
    if (is.nan(yi)) {
      warning("A funcao divergiu e o treinamento foi interrompido")
      break
    } else{
      caminho_df[(epoch+1),] <- list(x1,x2, yi, epoch)
    }
  }

  return(caminho_df)
}

optim_inercia <- SGD_momentum(x = c(0,5), vel = c(0,0), lr = 0.01, momentum = 0.9, epochs = 100)
optim_inercia[which(optim_inercia$f == min(optim_inercia$f)),]


#### h)

RMSprop <- function(x, lr = 0.01, decay = 0.1, estabilizador = 10^(-6), epochs = 10) {
  x <- as.vector(x)
  x1 <- x[1]
  x2 <- x[2]

  if (x1 == 0 & x2 == 0) {
    warning(paste0("O ponto inicial (0,0) zera o gradiente\n",
                   "impossibilitando a descida de gradiente."))
  }

  ## Guarda todo o caminho percorrido pela funcao
  caminho_df <- tibble(x1 = numeric(epochs+1),
                       x2 = numeric(epochs+1),
                       f = numeric(epochs+1),
                       epoch = numeric(epochs+1))
  caminho_df[1,] <- list(x1, x2, fbase(x1,x2), 0)

  ## Acumulo
  r <- numeric(length(x))

  for (epoch in 1:epochs) {
    # Gradiente
    ddx1 <- 4*x1^3 + 2*x1*x2 + x2^2 - 40*x1
    ddx2 <- 4*x2^3 + 2*x1*x2 + x1^2 - 30*x2
    grad <- c(ddx1, ddx2)

    # Atualizacao do Acumulo
    r <- decay*r + (1-decay) * grad*grad

    # Variacao
    delta.x <- -lr/sqrt(estabilizador + r) * grad

    # Atualizacao das variaveis
    x <- x + delta.x
    x1 <- x[1]
    x2 <- x[2]

    # Atualizacao do valor minimo
    yi <- fbase(x1,x2)
    if (is.nan(yi)) {
      warning("A funcao divergiu e o treinamento foi interrompido")
      break
    } else{
      caminho_df[(epoch+1),] <- list(x1,x2, yi, epoch)
    }
  }

  return(caminho_df)
}

optim_rms <- RMSprop(x = c(0,5), lr = 0.001, decay = 0.9, estabilizador = 10^(-6), epochs = 100)
optim_rms[which(optim_rms$f == min(optim_rms$f)),]
