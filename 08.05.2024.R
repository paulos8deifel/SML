# Lab: Classification Methods


## The Stock Market Data

###
install.packages("ISLR2")
library(ISLR2)
install.packages("GGally")
library(GGally)
install.packages("nnet")
library(nnet)
install.packages("MASS")
library(MASS)
install.packages("glmnet")
library(glmnet)
install.packages("dplyr")
library(dplyr)
install.packages("haven")
library(haven)
install.packages("Hmisc")
library(Hmisc)
install.packages("readr")
library(readr)
install.packages("lmtest")
library(lmtest)
install.packages("MVN")
library(MVN)
install.packages("biotools")
library(biotools)
library(caret)
install.packages("gridExtra")
library(gridExtra)
install.packages("ggplot2")
library(ggplot2)
install.packages("progress")
library(progress)

# 1. Datenvorbereitung
diabetic_data_all <- read_csv("~/Documents/Dokumente - MacBook Air (10)/Dokumente/Bamberg/Semester 2/Stat. Machine Learning/diabetic_data.csv")

diabetic_data <- diabetic_data_all[, c("readmitted", "race", "gender", "age", 
                                       "time_in_hospital", "number_emergency", "number_diagnoses", "diabetesMed", "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone")]
data_plot <- diabetic_data_all[, c("time_in_hospital", "number_emergency", "number_diagnoses", "diabetesMed", "num_lab_procedures", "num_procedures", "num_medications", "readmitted", "gender")]
is.numeric(diabetic_data$number_diagnoses)

table(diabetic_data$change)
is.numeric(diabetic_data$readmitted)
anyNA(diabetic_data)
table(diabetic_data$readmitted)
##remove NAs, in diesem Fall "?"
data1 <- data1[data1$gender != "Unknown/Invalid", ]
data <- diabetic_data_all[, c("readmitted", "race", "gender", "age", 
                              "time_in_hospital", "number_emergency", "number_diagnoses", "diabetesMed", "num_lab_procedures", "num_procedures", "num_medications")]
rows_with_question_mark <- apply(diabetic_data, 1, function(row) any(row == "?"))
data1 <- data[!rows_with_question_mark, ]
data_plot <- data_plot %>% mutate(gender = factor(
  gender, levels = c("Female", "Male"), labels = c(0,1)))
data_plot$gender <- as.numeric(data_plot$gender)
data1 <- data1 %>% mutate(diabetesMed = factor(
  diabetesMed, levels = c("No", "Yes"), labels = c(0,1)))
data_plot <- data_plot %>% mutate(diabetesMed = factor(
  diabetesMed, levels = c("No", "Yes"), labels = c(0,1)))
data_plot$diabetesMed <- as.numeric(data_plot$diabetesMed)
data1$readmitted_binary <- ifelse(data1$readmitted == "NO", "No", "Yes")
#data1$readmitted_binary <- as.numeric(ifelse(data1$readmitted == "NO", 0, 1))
data_plot$readmitted <- as.numeric(ifelse(data_plot$readmitted == "NO", 0, 1))
is.numeric(data_plot$readmitted)
data1 <- data1[data1$gender != "Unknown/Invalid", ]
nrow(data1)
## 2. Explorative Datenanalyse (EDA)
xtabs(~ readmitted_binary + gender, data = data1)
xtabs(~ readmitted_binary + time_in_hospital, data = data1)

## 3. Logistische Regression
glm1 <- glm(readmitted_binary ~ time_in_hospital + number_emergency + number_diagnoses + diabetesMed  + num_lab_procedures + num_procedures + num_medications + gender,
            data = data1, family = binomial)
summary(glm1)
coef(glm1)
glm_probs <- predict(glm1, type = "response")
summary(glm_probs) 
glm_pred <- ifelse(glm_probs>0.5, "Yes", "No")
table(glm_pred, data1$readmitted_binary)
mean(glm_pred==data1$readmitted_binary) #hier readmitted_binary ausführen

#Multinominal
data1$readmitted_binary <- as.factor(data1$readmitted_binary)

formula <- readmitted_binary ~ time_in_hospital + number_emergency + number_diagnoses + diabetesMed + 
  num_lab_procedures + num_procedures + num_medications + gender

trainData <- data1

ctrl <- trainControl(method = "cv", number = 10) 
set.seed(123)
glmnet_model <- train(formula, data = trainData, method = "glmnet", trControl = ctrl, family = "binomial")
print(glmnet_model)

coef(glmnet_model$finalModel, s = glmnet_model$bestTune$lambda)

## 4. Prüfung der Annahmen für Diskriminanzanalyse (LDA und QDA):
#Normalverteilung QQ
data1$readmitted_binary <- as.factor(data1$readmitted_binary)
create_qq_plot <- function(data, variable, group) {
  ggplot(data, aes_string(sample = variable)) +
    stat_qq() +
    stat_qq_line() +
    ggtitle(paste("QQ Plot of", variable, "for group", group)) +
    theme_minimal()
}

groups <- unique(data1$readmitted_binary)
plots <- list()
for (group in groups) {
  subset_data <- data1[data1$readmitted_binary == group, ]
  for (var in c("time_in_hospital", "number_emergency", "number_diagnoses", "diabetesMed", 
                "num_lab_procedures", "num_procedures", "num_medications", 
                "gender")) {
    plot <- create_qq_plot(subset_data, var, group)
    plots <- c(plots, list(plot))
  }
}
grid.arrange(grobs = plots, ncol = 3)
#Normalverteilung hist
create_histogram <- function(data, variable, group) {
  ggplot(data, aes_string(x = variable)) +
    geom_histogram(aes(y = ..density..), bins = 30, fill = "blue", alpha = 0.5) +
    geom_density(color = "red", size = 1) +
    ggtitle(paste("Histogram of", variable, "for group", group)) +
    theme_minimal()
}
groups <- unique(data1$readmitted_binary)
hist_plots <- list()
for (group in groups) {
  subset_data <- data1[data1$readmitted_binary == group, ]
  for (var in c("time_in_hospital", "number_emergency", "number_diagnoses", "diabetesMed", 
                "num_lab_procedures", "num_procedures", "num_medications", 
                "gender")) {
    plot <- create_histogram(subset_data, var, group)
    hist_plots <- c(hist_plots, list(plot))
  }
}
grid.arrange(grobs = hist_plots, ncol = 3)

#Varianz-Kovarianz-Matrizen gleich
sapply(data1, is.numeric)
sapply(data_plot, is.numeric)

group <- data1$readmitted_binary
variables <- data1[, c("time_in_hospital", "number_emergency", "number_diagnoses", "diabetesMed", "num_lab_procedures", "num_procedures", "num_medications", "gender")]
box_m_test <- boxM(variables, group)
print(box_m_test)
#LDA
lda.fit <- lda(readmitted_binary ~ time_in_hospital + number_emergency + number_diagnoses + diabetesMed + 
                 num_lab_procedures + num_procedures + num_medications + gender, data = data1)
lda.fit
dev.new(width = 8, height = 6)
plot(lda.fit)

lda.pred <- predict(lda.fit, data1)
names(lda.pred)

lda.class <- lda.pred$class
table(lda.class, data1$readmitted_binary)
mean(lda.class==data1$readmitted_binary)
#QDA
qda.fit <- qda(readmitted_binary ~ time_in_hospital + number_emergency + number_diagnoses + diabetesMed + 
                 num_lab_procedures + num_procedures + num_medications + gender, data = data1)
qda.fit
qda.pred <- predict(qda.fit, data1)
qda.class <- predict(qda.fit, data1)$class
names(qda.pred)
table(qda.class, data1$readmitted_binary)
mean(qda.class==data1$readmitted_binary)
## 5. Vergleich
data1$readmitted <- as.factor(data1$readmitted_binary)
if(!require(RColorBrewer)) install.packages("RColorBrewer", dependencies=TRUE)
library(RColorBrewer)
colors <- brewer.pal(length(levels(data1$readmitted_binary)), "Set1")
color_vector <- as.numeric(data1$readmitted_binary)
colors_mapped <- colors[color_vector]
pairs(data_plot, col = colors_mapped, pch = 19, main = "Pairs Plot Colored by Readmission Status")
pairs(data_plot, col=data_plot$readmitted_binary)





















#testing
#vorhersage
levels(data1$age)
age_levels <- c("[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", 
                "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)")

data1$age <- factor(data1$age, levels = age_levels)

new_data <- data.frame(
  time_in_hospital = 5,          # Beispielwert für time_in_hospital
  number_emergency = 1,          # Wert für number_emergency
  number_diagnoses = 8,          # Beispielwert für number_diagnoses
  diabetesMed = "Yes")
table(data1$gender)

data1$prob_readmitted <- predict(glm1, type = "response")
par(mar = c(5, 5, 2, 2))

plot(data1$num_procedures, data1$prob_readmitted, 
     xlab = "Number of Emergency", ylab = "Probability of Readmission",
     col = rgb(0, 0, 1), pch = 1, cex = 1)
lines(lowess(data1$number_emergency, data1$prob_readmitted), col = "red", lwd = 2)
summary(wald_test)

data1$logit_transformed <- log(data1$prob_readmitted / (1 - data1$prob_readmitted))

plot(data1$num_lab_procedures, data1$prob_readmitted,
     xlab = "Number of Medications", ylab = "Logit-transformed Probability (log odds)",
     col = "blue", pch = 19)
lines(lowess(data1$num_medications, data1$prob_readmitted), col = "red", lwd = 2)
str(data1)


#graph

# Erstellen eines Data Frames für die Vorhersagen
predicted_data <- data.frame(
  probability_of_readmission = data1$prob_readmitted,
  gender = data1$gender,
  num = data1$num_lab_procedures,
  readmitted_binary = data1$readmitted_binary
)

# Plot der Vorhersagen
library(ggplot2)
ggplot(data = predicted_data, aes(x = num, y = probability_of_readmission)) +
  geom_point(aes(color = readmitted_binary), size = 5) +
  xlab("num_lab_procedures") +
  ylab("Predicted probability of readmission")




#LDA
lda.fit <- lda(readmitted ~ time_in_hospital + number_emergency + number_diagnoses + diabetesMed + 
                 num_lab_procedures + num_procedures + num_medications + number_outpatient + number_inpatient, data = data1)
lda.fit
plot(lda.fit)

lda.pred <- predict(lda.fit, data1)
names(lda.pred)

predicted_data
lda.class <- lda.pred$class

#plot
# Erstellen eines DataFrames für den Plot
plot_data <- data.frame(LD1 = lda.pred$x[, 1], LD2 = lda.pred$x[, 2], Class = data1$readmitted)


ggplot(plot_data, aes(x = LD1, y = LD2, color = Class)) +
  geom_point(size = 2) +
  labs(title = "Fisher's Discriminant Plot", x = "Linear Discriminant 1", y = "Linear Discriminant 2") +
  theme_minimal()
lda.fit


table(data1$readmitted)


#test plot
numeric_vars <- names(data1_subset)

create_histogram <- function(data, var) {
  ggplot(data1, aes_string(x = var)) +
    geom_histogram(fill = "blue", color = "black", bins = 30) +
    ggtitle(paste("Histogramm von", var)) +
    xlab(var) +
    ylab("Häufigkeit") +
    theme_minimal()
}

for (var in numeric_vars) {
  print(create_histogram(data1_subset, var))
}


data_plot$readmitted <- factor(data_plot$readmitted, levels = c("NO", ">30", "<30"))
colors <- c("NO" = "green", ">30" = "yellow", "<30" = "blue")


View(data1)
#test
data_test <- data1[, c("gender", "time_in_hospital", "number_emergency", "number_diagnoses")]

View(data_plot)

#variablen numerisch recoden
data_plot <- data_plot %>%
  mutate(readmitted = factor(
    readmitted,
    levels = c("NO", ">30", "<30"),
    labels = c(0, 0.5, 1)
  ))


pairs(data_plot)

colors <- c("red", "blue", "green")
color_vector <- colors[as.factor(data_plot$readmitted)]
pairs(data_plot, col = color_vector, pch = 19,
      main = "Pairs Plot mit eingefärbten Punkten")
cor(data_plot)
View(data_plot)

data_plot$readmitted <- as.numeric(data_plot$readmitted)
is.numeric(data_plot$time_in_hospital)
is.numeric(data_plot$number_emergency)
is.numeric(data_plot$number_diagnoses)
data_plot$diabetesMed <- as.numeric(data_plot$diabetesMed)

#recode binär
data_plot$readmitted_binary <- ifelse(data_plot$readmitted == "NO", "NO", "YES")


#plot für Übersicht

ggplot(data_plot, aes(x = number_diagnoses, y = time_in_hospital, color = readmitted_binary)) +
  geom_point() +
  labs(title = "Scatterplot von diabetesMed und time_in_hospital",
       x = "number of diagnoses",
       y = "Time in Hospital") +
  theme_minimal() + geom_jitter()
View(data_plot)

#neuer Datensatz mit den drugs

View(diabetic_data)
table(diabetic_data$metformin)
variables_to_recode <- c("metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", 
                         "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", 
                         "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", 
                         "examide", "citoglipton", "insulin")

for (var in variables_to_recode) {
  diabetic_data[[var]] <- ifelse(diabetic_data[[var]] == "No", "No", "Yes")
}
diabetic_data$readmitted_binary <- ifelse(data_plot$readmitted == "NO", "NO", "YES")
table(diabetic_data$chlorpropamide)
#log reg
class(diabetic_data$readmitted_binary)
diabetic_data$readmitted_binary <- as.factor(diabetic_data$readmitted_binary)
diabetic_data$metformin <- factor(diabetic_data$metformin, levels = c("No", "Yes"))
diabetic_data$repaglinide <- factor(diabetic_data$repaglinide, levels = c("No", "Yes"))
diabetic_data$nateglinide <- factor(diabetic_data$nateglinide, levels = c("No", "Yes"))
diabetic_data$chlorpropamide <- factor(diabetic_data$chlorpropamide, levels = c("No", "Yes"))
diabetic_data$glimepiride <- factor(diabetic_data$glimepiride, levels = c("No", "Yes"))
diabetic_data$acetohexamide <- factor(diabetic_data$acetohexamide, levels = c("No", "Yes"))
diabetic_data$glipizide <- factor(diabetic_data$glipizide, levels = c("No", "Yes"))
diabetic_data$glyburide <- factor(diabetic_data$glyburide, levels = c("No", "Yes"))
diabetic_data$tolbutamide <- factor(diabetic_data$tolbutamide, levels = c("No", "Yes"))
diabetic_data$pioglitazone <- factor(diabetic_data$pioglitazone, levels = c("No", "Yes"))
diabetic_data$rosiglitazone <- factor(diabetic_data$rosiglitazone, levels = c("No", "Yes"))
diabetic_data$acarbose <- factor(diabetic_data$acarbose, levels = c("No", "Yes"))
diabetic_data$miglitol <- factor(diabetic_data$miglitol, levels = c("No", "Yes"))
diabetic_data$troglitazone <- factor(diabetic_data$troglitazone, levels = c("No", "Yes"))
diabetic_data$tolazamide <- factor(diabetic_data$tolazamide, levels = c("No", "Yes"))
diabetic_data$examide <- factor(diabetic_data$examide, levels = c("No", "Yes"))
diabetic_data$citoglipton <- factor(diabetic_data$citoglipton, levels = c("No", "Yes"))
diabetic_data$insulin <- factor(diabetic_data$insulin, levels = c("No", "Yes"))


model <- glm(diabetic_data$readmitted_binary ~ metformin + repaglinide + nateglinide + chlorpropamide + glimepiride +
               acetohexamide + glipizide + glyburide + tolbutamide + pioglitazone +
               rosiglitazone + acarbose + miglitol + troglitazone + tolazamide +
               examide + citoglipton + insulin,
             data = diabetic_data,
             family = "binomial")

summary(model)

#test normalverteilungs-check

set.seed(111)
ggplot(data1, aes(x = num_medications)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  labs(title = "Histogram of time_in_hospital",
       x = "Time in Hospital",
       y = "Frequency")

plot(data1$time_in_hospital)

#Grafik
data1$readmitted_binary <- as.numeric(data1$readmitted_binary)
table(data1$readmitted_binary)
data1$probability2 <- ifelse(data1$prob_readmitted > 0.5, 1, 0)

ggplot(data1, aes(x = num_medications, y = probability2)) +
  geom_jitter(height = 0.05, width = 0, alpha = 0.5) +  # Fügt die Punkte hinzu, leicht vertikal gestreut
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = FALSE, color = "blue") + # Logistische Regressionslinie
  labs(x = "Time in Hospital", y = "Probability of Readmission") + # Beschriftet die Achsen
  ggtitle("Probability of Readmission vs Time in Hospital") + # Fügt einen Titel hinzu
  theme_minimal()

