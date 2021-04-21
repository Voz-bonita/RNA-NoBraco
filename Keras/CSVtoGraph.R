library(ggplot2)
library(dplyr)
library(reshape2)
library(tibble)
library(plotly)
library(htmlwidgets)
library(purrr)
library(ggpubr)

# Baixe o csv do TensorBoard e leia-os
treino <- read.csv(paste0(getwd(), "/Keras/CSV-TensorBoard/run-Old-Dense-2x1_train-tag-epoch_loss.csv")) %>%
  select(Step, Value)
teste <- read.csv(paste0(getwd(), "/Keras/CSV-TensorBoard/run-Old-Dense-2x1_validation-tag-epoch_loss.csv")) %>%
  select(Step, Value)

EpochsXCusto <- right_join(treino, teste, by = "Step")
names(EpochsXCusto) <- c("Epoch", "Treino", "Teste")
EpochsXCusto <- melt(EpochsXCusto, id.vars = "Epoch")

ggplot(data = EpochsXCusto) +
  geom_line(aes(x = Epoch, y = value, color = variable), size=2) +
  guides(color = guide_legend(title = "Banco")) +
  ylab("MSE")
# ggsave(paste0(getwd(),"/Images/Old_SGD.png"))


Densemax3x4 <- tibble(Epochs = seq(1,100))
for (layer in 1:3) {
  layer <- as.character(layer)
  for (neurons in c(1,2,4)) {
    neurons <- as.character(neurons)
    custo <- read.csv(paste0(getwd(),
                             "/Keras/CSV-TensorBoard/run-Dense-",
                             layer, "-",neurons,"_validation-tag-epoch_loss.csv"))$Value

    # Quase nenhum vetor de custos tem 100 observacoes
    n <- length(custo)
    custo <- c(custo, rep(NA, 100-n))

    Densemax3x4 <- add_column(Densemax3x4, custo, .name_repair = "minimal")
  }
}
names(Densemax3x4)[2:10] <- c("D-1L-1N", "D-1L-2N", "D-1L-D4N", "D-2L-1N", "D-2L-2N", "D-2L-4N", "D-3L-1N", "D-3L-2N", "D-3L-4N")
Densemax3x4 <- melt(Densemax3x4, id.vars = "Epochs")
names(Densemax3x4)[2:3] <- c("Arquitetura", "Custo")

ggplot(data = Densemax3x4) +
  geom_line(aes(x = Epochs, y = Custo, color = Arquitetura), size = 1.1) +
  ylim(0,200) + xlim(0,30)
# ggsave(paste0(getwd(),"/Images/Arq-Simples.png"))


Densemax2x64 <- tibble(Epochs = seq(1,100))
for (layer in 1:3) {
  layer <- as.character(layer)
  for (neurons in c(16,32,64)) {
    neurons <- as.character(neurons)
    custo <- read.csv(paste0(getwd(),
                             "/Keras/CSV-TensorBoard/run-Dense-",
                             layer, "-",neurons,"_validation-tag-epoch_loss.csv"))$Value

    n <- length(custo)
    custo <- c(custo, rep(NA, 100-n))

    Densemax2x64 <- add_column(Densemax2x64, custo, .name_repair = "minimal")
  }
}
names(Densemax2x64)[2:9] <- c("D-1L-16N", "D-1L-32N", "D-1L-D64N", "D-2L-16N", "D-2L-32N", "D-2L-64N", "D-3L-16N", "D-3L-32N")
Densemax2x64 <- melt(Densemax2x64, id.vars = "Epochs")
names(Densemax2x64)[2:3] <- c("Arquitetura", "Custo")

comp_plot <- ggplot(data = Densemax2x64) +
  geom_line(aes(x = Epochs, y = Custo, color = Arquitetura), size = 1.1)

comp_plot +
  ylim(0,200) + xlim(0,90)
# ggsave(paste0(getwd(),"/Images/A2-Todas.png"))

comp_plot +
  ylim(195,200) + xlim(0,30)
# ggsave(paste0(getwd(),"/Images/A2-Sup.png"))

comp_plot +
  ylim(0,30) + xlim(0,90)
# ggsave(paste0(getwd(),"/Images/A2-Inf.png"))


LR <- tibble(Epochs = seq(1,100))
i <- 1
for (lr in c(0.01, 0.03, 0.005, 0.001)) {
  lr <- as.character(lr)

  for (neurons in c(16,32,64)) {
    neurons <- as.character(neurons)
    custo <- read.csv(paste0(getwd(),
                             "/Keras/CSV-TensorBoard/lr/run-Dense-1-",
                             neurons,"-",lr,"_validation-tag-epoch_loss.csv"))$Value

    n <- length(custo)
    custo <- c(custo, rep(NA, 100-n))

    LR <- add_column(LR, custo, .name_repair = "minimal")
    i <- i + 1
    names(LR)[i] <- paste0("D1-",neurons,"N-",lr,"lr")
  }
}

Lrs <- list(LR0.01 = LR[1:4] %>%
  melt(id.vars = "Epochs"),
     LR0.03 = LR[c(1,5:7)] %>%
  melt(id.vars = "Epochs"),
     LR0.005 = LR[c(1,8:10)] %>%
  melt(id.vars = "Epochs"),
     LR0.001 = LR[c(1,11:13)] %>%
  melt(id.vars = "Epochs"))

Lrs <- map(Lrs, ~rename(.x, "Arquitetura" = "variable", "Custo" = "value"))

pbase <- ggplot(NULL, aes(x=Epochs, y=Custo, color=Arquitetura)) +
  ylim(0,10)

p1 <- pbase + geom_line(data = Lrs[[1]], size = 1.3)
p2 <- pbase + geom_line(data = Lrs[[2]], size = 1.3)
p3 <- pbase + geom_line(data = Lrs[[3]], size = 1.3)
p4 <- pbase + geom_line(data = Lrs[[4]], size = 1.3)
ggarrange(p1,p2,p3,p4)
# ggsave(paste0(getwd(), "/Images/Multi-lr.png"))

ArcD1 <- LR[c(1,4,7,10,13)] %>%
  melt(id.vars = "Epochs") %>%
  rename("Arquitetura" = "variable", "Custo" = "value")
pbase + geom_line(data = ArcD1, size = 1.3)
# ggsave(paste0(getwd(), "/Images/Multi-lr-64N.png"))