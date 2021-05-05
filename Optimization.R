library(pacman)
p_load(ggplot2, dplyr, latex2exp)


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
           label="Minimo Global", color="red") +
  theme(panel.grid=element_blank(),
        panel.background=element_rect(fill = "transparent",colour = NA),
        panel.border=element_blank(),
        legend.position = "bottom") +
  guides(fill=guide_legend(nrow=4, byrow=TRUE)) +
  scale_x_continuous(breaks = seq(-5,5,5)) +
  scale_y_continuous(breaks = seq(-5,5,5)) +
  xlab(TeX("$X_1$")) + ylab(TeX("$X_2$"))
