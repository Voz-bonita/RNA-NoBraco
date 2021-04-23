library(pacman)
pacman::p_load(dplyr, ggplot2, latex2exp, tibble)

path_fix <- function (path) {
  returnValue(paste0(getwd(), path))
}


dados <- read.csv("Data/Dados.csv")

corte <- 80000
treinamento <- dados[1:corte,]
teste <- dados[(corte+1):nrow(dados),]

predito <- read.csv(path_fix("/Keras/optim_predict.csv"), header = F) %>%
  rename("yhat" = "V1")

predictXobs <- tibble(
  yhat = predito$yhat,
  y = teste$y
)

preXobs.plot <- ggplot(predictXobs, aes(x=yhat, y=y)) +
  geom_point() +
  geom_abline(colour="red", size=1.3) +
  coord_fixed(ratio = 1) +
  xlab(TeX("$\\hat{y}$")) + ylab(TeX("y"))
# preXobs.plot + ggsave(path_fix("/Images/YxYhat_V2.png"))

# Item e)

n <- 100
x1 <- seq(-3, 3, length.out=n)
x2 <- seq(-3, 3, length.out=n)
x.grid <- expand.grid(x1, x2)

# Previsoes
dados.grid.yhat <- read.csv(path_fix("/Keras/Dados-grid-pred.csv"), header = F) %>%
  add_column(as_tibble(xgrid), .before = "V1") %>%
  add_column(rep("NN", nrow(x.grid))) %>%
  rename_all(~ c("x1", "x2", "mu", "origem"))


# Default
dados.grid <- as_tibble(x.grid) %>%
  add_column(rep("Default", nrow(x.grid))) %>%
  rename_all(~ c("x1", "x2", "origem")) %>%
  mutate(mu=abs(x1^3 - 30*sin(x2) + 10)) %>%
  add_row(dados.grid.yhat)


ggplot(dados.grid, aes(x=x1, y=x2)) +
  geom_point(aes(colour=mu), size=2, shape=15) +
  facet_wrap(~ origem) +
  coord_cartesian(expand=F) +
  scale_colour_gradient(low="white",
                        high="black",
                        name=TeX("E(Y|X_1, X_2)")) +
  xlab(TeX("X_1")) + ylab(TeX("X_2"))
# ggsave(path_fix("/Images/Superficie.png"))


