library(ggplot2)
library(dplyr)
library(reshape2)

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