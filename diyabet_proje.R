# =========================================================
# DIABETES - Naive Bayes vs SVM (FAST)
# Outputs:
#  - model_karsilastirma_sonuclari.csv
#  - model_ciktilari.txt
#  - roc_naive_bayes.png
#  - roc_svm_fast.png
# =========================================================

# 0) Temizle
rm(list = ls())
cat("\014")  # console temizler 

# 1) Paketler
required_pkgs <- c("caret", "pROC", "e1071", "LiblineaR")
for (p in required_pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
}
library(caret)
library(pROC)
library(e1071)
library(LiblineaR)

set.seed(42)

# 2) Dosya oku
file_path <- "diabetes_prediction_dataset.csv"
if (!file.exists(file_path)) {
  stop(paste("Dosya bulunamadı:", file_path, "."))
}

data <- read.csv(file_path, stringsAsFactors = FALSE)

# 3) Hedef değişken: diabetes
if (!("diabetes" %in% names(data))) stop

# diabetes kolonunu factor yap (0/1 ise No/Yes'e çevir)
if (is.numeric(data$diabetes) || is.integer(data$diabetes)) {
  data$diabetes <- ifelse(data$diabetes == 1, "Yes", "No")
}
data$diabetes <- factor(data$diabetes, levels = c("No","Yes"))

# 4) Karakter kolonları faktöre çevir (smoking_history gibi)
char_cols <- names(data)[sapply(data, is.character)]
for (cc in char_cols) data[[cc]] <- as.factor(data[[cc]])

# 5) Train/Test böl
idx <- createDataPartition(data$diabetes, p = 0.8, list = FALSE)
train_df <- data[idx, ]
test_df  <- data[-idx, ]

cat("VERI BOYUTU:\n")
cat("Toplam:", nrow(data), "satir,", ncol(data), "sutun\n")
cat("Train :", nrow(train_df), "\n")
cat("Test  :", nrow(test_df), "\n\n")

# 6) Ortak metrik fonksiyonları
safe_div <- function(a, b) ifelse(b == 0, 0, a / b)

metrics_from_preds <- function(y_true, y_pred) {
  cm <- confusionMatrix(y_pred, y_true, positive = "Yes")
  acc <- as.numeric(cm$overall["Accuracy"])
  prec <- as.numeric(cm$byClass["Precision"])
  rec <- as.numeric(cm$byClass["Recall"])
  f1 <- ifelse(is.na(prec) || is.na(rec) || (prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
  list(confusion = cm, accuracy = acc, precision = prec, recall = rec, f1 = f1)
}

# =========================================================
# 7) NAIVE BAYES
# =========================================================
cat("===== NAIVE BAYES =====\n")
nb_model <- naiveBayes(diabetes ~ ., data = train_df)

nb_pred_class <- predict(nb_model, newdata = test_df, type = "class")
nb_pred_prob  <- predict(nb_model, newdata = test_df, type = "raw")[, "Yes"]

nb_m <- metrics_from_preds(test_df$diabetes, nb_pred_class)
nb_roc <- roc(response = test_df$diabetes, predictor = nb_pred_prob, levels = c("No","Yes"), quiet = TRUE)
nb_auc <- as.numeric(auc(nb_roc))

print(nb_m$confusion)
cat("Naive Bayes AUC:", nb_auc, "\n\n")

# ROC (ekrana + PNG)
plot(nb_roc, main = "Naive Bayes ROC Curve")
png("roc_naive_bayes.png", width = 900, height = 650)
plot(nb_roc, main = "Naive Bayes ROC Curve")
dev.off()

# =========================================================
# 8) SVM (FAST) - LiblineaR
# =========================================================
cat("===== SVM (FAST - LiblineaR) =====\n")
max_train <- 20000
if (nrow(train_df) > max_train) {
  train_fast <- train_df[sample(seq_len(nrow(train_df)), max_train), ]
} else {
  train_fast <- train_df
}

x_train_df <- subset(train_fast, select = -diabetes)
x_test_df  <- subset(test_df,    select = -diabetes)

y_train <- train_fast$diabetes
y_test  <- test_df$diabetes

# Dummy encode
x_train_mm <- model.matrix(~ . - 1, data = x_train_df)
x_test_mm  <- model.matrix(~ . - 1, data = x_test_df)

# Kolon hizalama
missing_cols <- setdiff(colnames(x_train_mm), colnames(x_test_mm))
if (length(missing_cols) > 0) {
  add <- matrix(0, nrow = nrow(x_test_mm), ncol = length(missing_cols))
  colnames(add) <- missing_cols
  x_test_mm <- cbind(x_test_mm, add)
}
x_test_mm <- x_test_mm[, colnames(x_train_mm), drop = FALSE]

# Scale
x_train_sc <- scale(x_train_mm)
center <- attr(x_train_sc, "scaled:center")
scal   <- attr(x_train_sc, "scaled:scale")
x_test_sc  <- scale(x_test_mm, center = center, scale = scal)

# LiblineaR label 0/1
y_train_bin <- ifelse(y_train == "Yes", 1, 0)

# type=0: logistic regression + proba
svm_model <- LiblineaR(data = x_train_sc, target = y_train_bin, type = 0)

svm_pred_obj <- predict(svm_model, x_test_sc, proba = TRUE)
svm_prob <- svm_pred_obj$probabilities[, "1"]

svm_pred_class <- ifelse(svm_prob >= 0.5, "Yes", "No")
svm_pred_class <- factor(svm_pred_class, levels = c("No","Yes"))

svm_m <- metrics_from_preds(y_test, svm_pred_class)
svm_roc <- roc(response = y_test, predictor = svm_prob, levels = c("No","Yes"), quiet = TRUE)
svm_auc <- as.numeric(auc(svm_roc))

print(svm_m$confusion)
cat("SVM (FAST) AUC:", svm_auc, "\n\n")

plot(svm_roc, main = "SVM (FAST - LiblineaR) ROC Curve")
png("roc_svm_fast.png", width = 900, height = 650)
plot(svm_roc, main = "SVM (FAST - LiblineaR) ROC Curve")
dev.off()

# =========================================================
# 9) Sonuç tablosu + dosyalara yaz
# =========================================================
results <- data.frame(
  Model     = c("Naive_Bayes", "SVM_Fast_LiblineaR"),
  Accuracy  = c(nb_m$accuracy,  svm_m$accuracy),
  Precision = c(nb_m$precision, svm_m$precision),
  Recall    = c(nb_m$recall,    svm_m$recall),
  F1        = c(nb_m$f1,        svm_m$f1),
  AUC       = c(nb_auc,         svm_auc)
)

cat("===== MODEL KARSILASTIRMA =====\n")
print(results)

write.csv(results, "model_karsilastirma_sonuclari.csv", row.names = FALSE)

sink("model_ciktilari.txt")
cat("VERI BOYUTU:\n")
cat("Toplam:", nrow(data), "satir,", ncol(data), "sutun\n")
cat("Train :", nrow(train_df), "\n")
cat("Test  :", nrow(test_df), "\n\n")

cat("===== NAIVE BAYES =====\n")
print(nb_m$confusion)
cat("\nNaive Bayes AUC:", nb_auc, "\n\n")

cat("===== SVM (FAST - LiblineaR) =====\n")
print(svm_m$confusion)
cat("\nSVM (FAST) AUC:", svm_auc, "\n\n")

cat("===== MODEL KARSILASTIRMA =====\n")
print(results)

cat("\nOlusan dosyalar:\n")
cat("- model_karsilastirma_sonuclari.csv\n")
cat("- model_ciktilari.txt\n")
cat("- roc_naive_bayes.png\n")
cat("- roc_svm_fast.png\n")
sink()

cat("\nBitti ✅  Dosyalar olustu:\n")
cat("- model_karsilastirma_sonuclari.csv\n")
cat("- model_ciktilari.txt\n")
cat("- roc_naive_bayes.png\n")
cat("- roc_svm_fast.png\n")