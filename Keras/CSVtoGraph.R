library(ggplot2)
library(dplyr)
library(reshape2)
library(tibble)
library(plotly)
library(htmlwidgets)

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
ggsave(paste0(getwd(),"/Images/A2-Todas.png"))

comp_plot +
  ylim(195,200) + xlim(0,30)
ggsave(paste0(getwd(),"/Images/A2-Sup.png"))

comp_plot +
  ylim(0,30) + xlim(0,90)
ggsave(paste0(getwd(),"/Images/A2-Inf.png"))