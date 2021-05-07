library(pacman)
p_load(ggplot2, dplyr, latex2exp)

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
    warning("O ponto inicial (0,0) zera o gradiente\n  impossibilitando a descida de gradiente.")
  }

  ## Valor inicial
  y0 <- fbase(x1,x2)
  menor <- y0
  melhor_ponto <- c(x1,x2)

  for (i in 1:epochs) {
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
    }
  }

  melhor_resultado <- c("f(x1,x2)" = menor,
                        "x1" = melhor_ponto[1],
                        "x2" = melhor_ponto[2])

  return(melhor_resultado)
}

grad_desc2(1,1)
grad_desc2(1,0)
grad_desc2(x1 = -3, x2 = -3, lr = 0.1)


#### d)

grad_desc2(0, 5, lr = 0.01, epochs = 100)