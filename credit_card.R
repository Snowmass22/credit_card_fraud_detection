# ============================================================
# CREDIT CARD FRAUD DETECTION - COMPLETE CORRECTED R CODE
# ============================================================

# ── STEP 1: Install & Load Libraries ─────────────────────────
install.packages(c("tidyverse", "caret", "ROSE", "randomForest",
                   "pROC", "corrplot", "rpart", "rpart.plot"))

library(tidyverse)
library(caret)
library(ROSE)
library(randomForest)
library(pROC)
library(corrplot)
library(rpart)
library(rpart.plot)

# ── STEP 2: Load Dataset ─────────────────────────────────────
df <- read.csv("creditcard.csv")

# ── STEP 3a: Structure & Summary ─────────────────────────────
str(df)
summary(df)
cat("Rows:", nrow(df), "\nColumns:", ncol(df), "\n")

# ── STEP 3b: Missing Values ───────────────────────────────────
cat("Missing values per column:\n")
print(colSums(is.na(df)))

# ── STEP 3c: Outlier Detection (keep — may be fraud) ─────────
ggplot(df, aes(x = Amount)) +
  geom_histogram(bins = 50, fill = "steelblue") +
  ggtitle("Transaction Amount Distribution (Before Scaling)") +
  theme_minimal()

Q1      <- quantile(df$Amount, 0.25)
Q3      <- quantile(df$Amount, 0.75)
IQR_val <- Q3 - Q1
outliers <- df$Amount < (Q1 - 1.5 * IQR_val) | df$Amount > (Q3 + 1.5 * IQR_val)
cat("Number of outliers in Amount:", sum(outliers), "\n")

# ── STEP 3d: Convert Data Types ───────────────────────────────
# FIX 1: Convert ALL integer columns to numeric BEFORE scaling
#         (ROSE only accepts numeric or factor — integers cause the error)
df <- df %>% mutate(across(where(is.integer), as.numeric))

# FIX 2: scale() returns a matrix — use as.numeric() to flatten to vector
df$Amount_Scaled <- as.numeric(scale(df$Amount))
df$Time_Scaled   <- as.numeric(scale(df$Time))

# Drop original Amount and Time
df <- df %>% select(-Amount, -Time)

# FIX 3: Convert Class to factor AFTER all mutations
df$Class <- factor(df$Class, levels = c(0, 1), labels = c("Genuine", "Fraud"))

# Verify everything
str(df)
summary(df)
cat("Class distribution:\n")
print(table(df$Class))

# ── STEP 4: EDA ───────────────────────────────────────────────

# Plot 1: Class imbalance
ggplot(df, aes(x = Class, fill = Class)) +
  geom_bar() +
  scale_fill_manual(values = c("steelblue", "firebrick")) +
  ggtitle("Class Distribution: Genuine vs Fraud") +
  theme_minimal() +
  # FIX 4: after_stat() replaces deprecated ..count.. syntax
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5)

# Plot 2: Scaled Amount by Class
ggplot(df, aes(x = Class, y = Amount_Scaled, fill = Class)) +
  geom_boxplot() +
  scale_fill_manual(values = c("steelblue", "firebrick")) +
  ggtitle("Scaled Amount by Class") +
  theme_minimal()

# Plot 3: Correlation heatmap (numeric cols only)
num_df      <- df %>% select(where(is.numeric))   # FIX 5: select_if is deprecated
corr_matrix <- cor(num_df)
corrplot(corr_matrix, method = "color", type = "upper",
         tl.cex = 0.5, title = "Correlation Matrix",
         mar = c(0, 0, 1, 0))

# Plot 4: Feature distributions by class (V1–V4)
df_long <- df %>%
  select(Class, V1, V2, V3, V4) %>%
  pivot_longer(cols = V1:V4, names_to = "Feature", values_to = "Value")

ggplot(df_long, aes(x = Value, fill = Class)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~Feature, scales = "free") +
  scale_fill_manual(values = c("steelblue", "firebrick")) +
  ggtitle("Feature Distributions by Class") +
  theme_minimal()

# ── Insights ──────────────────────────────────────────────────

# Insight 1: Fraud rate
fraud_rate <- mean(df$Class == "Fraud") * 100
cat("Fraud Rate:", round(fraud_rate, 4), "%\n")

# Insight 2: Mean scaled amount per class
df %>%
  group_by(Class) %>%
  summarise(Avg_Amount_Scaled = mean(Amount_Scaled), Count = n()) %>%
  print()

# Insight 3: Most discriminating features
# FIX 6: sapply(df, is.numeric) on a tibble with factor column works,
#         but using where(is.numeric) is safer
num_cols    <- names(df)[sapply(df, is.numeric)]
fraud_means   <- colMeans(df[df$Class == "Fraud",    num_cols])
genuine_means <- colMeans(df[df$Class == "Genuine",  num_cols])
diff_means    <- abs(fraud_means - genuine_means)
cat("Top 5 most discriminating features:\n")
print(sort(diff_means, decreasing = TRUE)[1:5])

# ── STEP 5: Train-Test Split ──────────────────────────────────
set.seed(42)

trainIndex <- createDataPartition(df$Class, p = 0.7, list = FALSE)
train_data <- df[ trainIndex, ]
test_data  <- df[-trainIndex, ]

cat("Train size:", nrow(train_data),
    "\nTest size:",  nrow(test_data),
    "\nTrain fraud cases:", sum(train_data$Class == "Fraud"), "\n")

# ── STEP 6: Handle Class Imbalance with ROSE ─────────────────
# FIX 7: Ensure ALL columns are numeric or factor — no matrix/integer cols
#         This is the root fix for the ROSE error
train_data <- train_data %>%
  mutate(across(where(is.numeric), as.numeric))  # ensure no matrix cols remain

train_balanced <- ROSE(Class ~ ., data = train_data, seed = 42)$data

cat("Balanced class distribution:\n")
print(table(train_balanced$Class))

# ── STEP 7: Model 1 — Logistic Regression ────────────────────
model_lr <- glm(Class ~ ., data = train_balanced, family = binomial)

pred_lr_prob <- predict(model_lr, test_data, type = "response")
pred_lr      <- ifelse(pred_lr_prob > 0.5, "Fraud", "Genuine")
pred_lr      <- factor(pred_lr, levels = c("Genuine", "Fraud"))

cm_lr <- confusionMatrix(pred_lr, test_data$Class, positive = "Fraud")
print(cm_lr)

# ── STEP 8: Model 2 — Random Forest ─────────────────────────
model_rf <- randomForest(Class ~ ., data = train_balanced,
                         ntree = 100, importance = TRUE)

pred_rf <- predict(model_rf, test_data)
cm_rf   <- confusionMatrix(pred_rf, test_data$Class, positive = "Fraud")
print(cm_rf)

varImpPlot(model_rf, main = "Random Forest — Feature Importance")

# ── STEP 9: Model 3 — Decision Tree ──────────────────────────
model_dt <- rpart(Class ~ ., data = train_balanced,
                  method = "class", cp = 0.001)

rpart.plot(model_dt, main = "Decision Tree for Fraud Detection")

pred_dt <- predict(model_dt, test_data, type = "class")
cm_dt   <- confusionMatrix(pred_dt, test_data$Class, positive = "Fraud")
print(cm_dt)

# ── STEP 10: ROC Curves & Model Comparison ───────────────────
roc_lr <- roc(test_data$Class, pred_lr_prob,
              levels = c("Genuine", "Fraud"), quiet = TRUE)

roc_rf <- roc(test_data$Class,
              predict(model_rf, test_data, type = "prob")[, "Fraud"],
              levels = c("Genuine", "Fraud"), quiet = TRUE)

roc_dt <- roc(test_data$Class,
              predict(model_dt, test_data, type = "prob")[, "Fraud"],
              levels = c("Genuine", "Fraud"), quiet = TRUE)

# Plot ROC
plot(roc_lr, col = "blue",  main = "ROC Curve Comparison", legacy.axes = TRUE)
plot(roc_rf, col = "green", add = TRUE)
plot(roc_dt, col = "red",   add = TRUE)
legend("bottomright",
       legend = c(paste("Logistic      AUC =", round(auc(roc_lr), 3)),
                  paste("Random Forest AUC =", round(auc(roc_rf), 3)),
                  paste("Decision Tree AUC =", round(auc(roc_dt), 3))),
       col = c("blue", "green", "red"), lwd = 2)

# Summary comparison table
results <- data.frame(
  Model     = c("Logistic Regression", "Random Forest", "Decision Tree"),
  Accuracy  = round(c(cm_lr$overall["Accuracy"],
                      cm_rf$overall["Accuracy"],
                      cm_dt$overall["Accuracy"]), 4),
  Precision = round(c(cm_lr$byClass["Precision"],
                      cm_rf$byClass["Precision"],
                      cm_dt$byClass["Precision"]), 4),
  Recall    = round(c(cm_lr$byClass["Recall"],
                      cm_rf$byClass["Recall"],
                      cm_dt$byClass["Recall"]), 4),
  F1        = round(c(cm_lr$byClass["F1"],
                      cm_rf$byClass["F1"],
                      cm_dt$byClass["F1"]), 4),
  AUC       = round(c(auc(roc_lr), auc(roc_rf), auc(roc_dt)), 4)
)

print(results)