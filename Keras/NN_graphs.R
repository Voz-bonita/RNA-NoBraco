library(pacman)
pacman::p_load(dplyr, ggplot2, latex2exp)

path_fix <- function (path) {
  returnValue(paste0(getwd(), path))
}


dados <- read.csv("Data/Dados.csv")

corte <- 80000
treinamento <- dados[1:corte,]
teste <- dados[(corte+1):nrow(dados),]

predito <- as.double(read.csv(path_fix("/Keras/optim_predict.csv")))

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




