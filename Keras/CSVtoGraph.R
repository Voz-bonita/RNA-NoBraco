library(pacman)
pacman::p_load(dplyr, ggplot2, ggpubr, latex2exp, purrr, reshape2, tibble)

path_fix <- function (path) {
  returnValue(paste0(getwd(), path))
}

NN_sumary <- function (epochs, layers, neurons, extra = "", endpoint = TRUE) {
  tab <- tibble(Epochs = seq(1,epochs))

  layers <- as.character(layers)
  neurons <- as.character(neurons)

  if (extra != "") {
    extra <- as.character(extra)
    if (!startsWith(extra,"-")){
      extra <- paste0("-", extra)
    }
  }

  for (layer in layers) {
    for (neu in neurons) {
      if (!endpoint & layer == last(layers) & neu == last(neurons)) {
        break
      }

      path <- paste0("/Keras/CSV-TensorBoard/run-Dense-",
                     layer, "-",neu,extra,"_validation-tag-epoch_loss.csv")
      custo <- tryCatch({read.csv(path_fix(path))$Value},
               error = function (e) {
                 message(e)
                 break
               })
      custo

      # Quase nenhum vetor de custos tem 100 observacoes
      # Causado pelo EarlyStopping
      n <- length(custo)
      custo <- c(custo, rep(NA, 100-n))


      tab <- add_column(tab, custo, .name_repair = "minimal")

      novo_nome <- paste0("D-",layer,"L-",neu,"N",extra)
      names(tab)[length(tab)] <- novo_nome
    }
  }
  tab <- melt(tab, id.vars = "Epochs") %>%
    rename_all(~ c("Epochs", "Arquitetura", "Custo"))
}


# Baixe o csv do TensorBoard e leia-os
treino <- read.csv(path_fix("/Keras/CSV-TensorBoard/run-Old-Dense-2x1_train-tag-epoch_loss.csv")) %>%
  select(Step, Value)
teste <- read.csv(path_fix("/Keras/CSV-TensorBoard/run-Old-Dense-2x1_validation-tag-epoch_loss.csv")) %>%
  select(Step, Value)


EpochsXCusto <- right_join(treino, teste, by = "Step") %>%
  rename_all(~ c("Epoch", "Treino", "Teste")) %>%
  melt(id.vars = "Epoch")

ggplot(data = EpochsXCusto) +
  geom_line(aes(x = Epoch, y = value, color = variable), size=1.3) +
  guides(color = guide_legend(title = "Banco")) +
  ylab("MSE")
# ggsave(paste0(getwd(),"/Images/Old_SGD.png"))


Densemax3x4 <- NN_sumary(epochs=100, layers=1:3, neurons=c(1,2,4))

ggplot(data = Densemax3x4) +
  geom_line(aes(x = Epochs, y = Custo, color = Arquitetura), size = 1.1) +
  ylim(0,200) + xlim(0,30)
# ggsave(paste0(getwd(),"/Images/Arq-Simples.png"))


Densemax2x64 <- NN_sumary(epochs = 100, layers=1:3, neurons=c(16,32,64), "", endpoint = F)

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
# ggsave(path_fix("/Images/Multi-lr.png"))

ArcD1 <- LR[c(1,4,7,10,13)] %>%
  melt(id.vars = "Epochs") %>%
  rename("Arquitetura" = "variable", "Custo" = "value")
pbase + geom_line(data = ArcD1, size = 1.3)
# ggsave(path_fix("/Images/Multi-lr-64N.png"))


custo2x32 <- read.csv(path_fix(
                         "/Keras/CSV-TensorBoard/run-Dense-2x32-lr-0.003_validation-tag-epoch_loss.csv"))$Value
custo2x64 <- read.csv(path_fix(
                         "/Keras/CSV-TensorBoard/run-Dense-2x64-lr-0.003_validation-tag-epoch_loss.csv"))$Value
min(custo2x32)
min(custo2x64)