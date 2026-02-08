# INITIAL SETUP & LIBRARIES
set.seed(123)

# Load library
library (tidyverse)
library (caret)
library (reshape2)
library (randomForest)
library (themis)
library (recipes)
library (pROC)
library (xgboost)
library (Ckmeans.1d.dp)

# DATA PREPROCESSING & CLEANING
# Load Data
raw_data = read.csv("loan_data_2007_2014.csv")

# Pemilihan variabel dan cleaning
df_clean = raw_data %>% 
  select(
    # Target (Variabel Y)
    loan_status, 
    # Karakteristik Pinjaman (Variabel X)
    loan_amnt, term, int_rate, sub_grade, 
    # Kemampuan Finansial (Variabel X)
    annual_inc, dti, emp_length, home_ownership, total_acc,
    # Riwayat Kredit (Variabel X)
    delinq_2yrs, inq_last_6mths, revol_util, purpose
  ) %>%
  mutate(
    # Labeling : 1 = Bad Loan (Risiko), 0 = Good Loan (Aman)
    loan_status = ifelse(loan_status %in% c("Charged Off", "Default", 
                                            "Does not meet the credit policy. Status:Charged Off", "Late (31-120 days)"), 1, 0),
    # Cleaning numerik
    term = as.numeric(gsub("[^0-9]", "", term)),
    emp_length = case_when(
      emp_length == "10+ years" ~ 10,
      emp_length == "< 1 year"  ~ 0,
      TRUE ~ as.numeric(gsub("[^0-9]", "", emp_length))
    ),
    emp_length = ifelse(is.na(emp_length), 0, emp_length)
  ) %>%
  drop_na() %>%
  mutate(across(c(sub_grade, home_ownership, purpose, loan_status), as.factor))
colSums(is.na(df_clean))
glimpse(df_clean)
table(df_clean$loan_status)

# EXPLORATORY DATA ANALYSIS
# Statistika Deskriptif
summary(df_clean)

# Deteksi Visual (Boxplot Panel) 
numeric_vars <- df_clean %>% select_if(is.numeric)
df_long <- numeric_vars %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value")
ggplot(df_long, aes(x = variable, y = value)) + 
  geom_boxplot(
    fill = "steelblue",           
    color = "black",              
    outlier.colour = "darkorange",
    outlier.shape = 16, 
    outlier.alpha = 0.5
  ) +
  facet_wrap(~variable, scales = "free") + 
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_blank(), 
    text = element_text(family = "serif"),
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    strip.background = element_rect(fill = "gray95") 
  ) + 
  labs(title = "Deteksi Outlier pada Seluruh Variabel Numerik", x = "", y = "Nilai")

# Penanganan Outlier : Capping (Winsorizing)
cap_outliers <- function(x) {
  qntl <- quantile(x, 0.99, na.rm = TRUE)
  x[x > qntl] <- qntl
  return(x)
}
df_clean <- df_clean %>%
  mutate(across(where(is.numeric), cap_outliers))

# Univariat Analysis
# Distribusi Loan status
ggplot(df_clean, aes(x = loan_status, fill = loan_status)) +
  geom_bar() +
  scale_fill_manual(values = c("0" = "darkblue", "1" = "darkorange")) +
  labs(title = "Distribusi Good (0) vs Bad (1) Loans", x = "Status", y = "Jumlah") +
  theme(text = element_text(family = "serif"), 
        plot.title = element_text(face = "bold", size = 14, hjust = 0.5))

# Analisis korelasi
# Korelasi bunga dengan loan status
ggplot(df_clean, aes(x = loan_status, y = int_rate, fill = loan_status)) +
  geom_boxplot() +
  scale_fill_manual(values = c("0" = "darkblue", "1" = "darkorange")) +
  labs(title = "Hubungan Suku Bunga dengan Risiko", x = "0: Good, 1: Bad", y = "Interest Rate") +
  theme(text = element_text(family = "serif"), 
        plot.title = element_text(face = "bold", size = 14, hjust = 0.5))

# Korelasi grade dengan loan status
ggplot(df_clean, aes(x = sub_grade, fill = loan_status)) +
  geom_bar(position = "fill") + 
  scale_fill_manual(values = c("0" = "darkblue", "1" = "darkorange")) +
  theme(axis.text.x = element_text(angle = 90),
        text = element_text(family = "serif"), 
        plot.title = element_text(face = "bold", size = 14, hjust = 0.5)) +
  labs(title = "Proporsi Gagal Bayar per Sub-Grade", y = "Persentase")

# Heatmap korelasi variabel
numeric_cols <- df_clean %>% select_if(is.numeric)
corr_matrix <- cor(numeric_cols, use = "complete.obs")
melted_corr <- melt(corr_matrix)
ggplot(melted_corr, aes(Var1, Var2, fill=value)) +
  geom_tile() +
  scale_fill_gradient2(low="darkblue", high="darkorange", mid="white", midpoint=0) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        text = element_text(family = "serif"), 
        plot.title = element_text(face = "bold", size = 14, hjust = 0.5)) +
  labs(title = "Heatmap Korelasi Variabel Numerik")

# Analisis multivariat
# Korelasi gaji vs pinjaman
ggplot(df_clean, aes(x = annual_inc, y = loan_amnt, color = loan_status)) +
  geom_point(alpha = 0.3) +
  scale_x_log10() + 
  scale_color_manual(values = c("0" = "navy", "1" = "darkorange"),
                     labels = c("Good Loan", "Bad Loan")) + 
  labs(title = "Hubungan Gaji vs Jumlah Pinjaman", 
       subtitle = "Visualisasi korelasi pendapatan terhadap besar pinjaman",
       x = "Gaji (Log Scale)", 
       y = "Jumlah Pinjam",
       color = "Status Pinjaman") + 
  theme_minimal() +
  theme(
    text = element_text(family = "serif"), 
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5),
    panel.grid.minor = element_blank(), 
    legend.position = "bottom" 
  )

# MODELING & FEATURE ENGINEERING
# Data Splitting
index = createDataPartition(df_clean$loan_status, p = 0.8, list = FALSE)
train_data = df_clean[index, ]
test_data  = df_clean[-index, ]
table(train_data$loan_status)

# Penanganan Imbalance dengan SMOTE
rec_smote <- recipe(loan_status ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_smote(loan_status) %>%
  prep()
train_smote <- juice(rec_smote)
test_baked  <- bake(rec_smote, test_data)


# MODELING (Comparing)
# 1. Baseline model (tanpa penanganan)- Logistic Regression
model_baseline = glm(loan_status ~ ., data = train_data, family = "binomial")

# 2. SMOTE Model (Logistic Regression tapi pakai data SMOTE)
model_smote = glm(loan_status ~ ., data = train_smote, family = "binomial")

# 3. Random Forest
train_small = train_data %>%
  group_by(loan_status) %>%
  sample_n(min(sum(train_data$loan_status == "1"), 5000)) %>% ungroup()
model_rf = randomForest(loan_status ~ ., 
                        data = train_small, 
                        ntree = 200)

# 4. XGBoost
train_matrix <- as.matrix(sapply(train_smote %>% select(-loan_status), as.numeric))
test_matrix  <- as.matrix(sapply(test_baked %>% select(-loan_status), as.numeric))
train_label  <- as.numeric(as.character(train_smote$loan_status))
test_label   <- as.numeric(as.character(test_baked$loan_status))
dtrain       <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest        <- xgb.DMatrix(data = test_matrix, label = test_label)

params_tuned <- list(
  objective = "binary:logistic", 
  learning_rate = 0.05, 
  max_depth = 6 
)

model_xgb <- xgb.train(
  params = params_tuned, 
  data = dtrain, 
  nrounds = 500,             
  watchlist = list(val=dtest, train=dtrain), 
  early_stopping_rounds = 20, 
  print_every_n = 50,
  verbose = 1
)


# EVALUATION & PERFORMANCE COMPARISON
# 1. Tabel akurasi 

# Prediksi Probabilitas
prob_baseline = predict(model_baseline, test_data, type = "response")
prob_smote    = predict(model_smote, test_baked, type = "response")
prob_rf       = predict(model_rf, test_data, type = "prob")[,2]
prob_xgb      = predict(model_xgb, dtest) 

# Konversi ke Klasifikasi (Threshold 0.5)
pred_baseline <- as.factor(ifelse(prob_baseline > 0.5, 1, 0))
pred_smote    <- as.factor(ifelse(prob_smote > 0.5, 1, 0))
pred_rf       <- as.factor(ifelse(prob_rf > 0.5, 1, 0))
pred_xgb      <- as.factor(ifelse(prob_xgb > 0.5, 1, 0))

# Hitung AUC masing-masing
auc_baseline <- round(auc(roc(test_data$loan_status, prob_baseline, quiet = TRUE)), 3)
auc_smote    <- round(auc(roc(test_data$loan_status, prob_smote, quiet = TRUE)), 3)
auc_rf       <- round(auc(roc(test_data$loan_status, prob_rf, quiet = TRUE)), 3)
auc_xgb      <- round(auc(roc(test_label, prob_xgb, quiet = TRUE)), 3)

# Buat tabel perbandingan
tabel_perbandingan <- data.frame(
  Metode = c("Baseline (Logistic)", "SMOTE (Logistic)", "Random Forest", "XGBoost Tuned"),
  AUC = c(auc_baseline, auc_smote, auc_rf, auc_xgb)
)

# Tambahkan Metrik Lain (Accuracy & Recall)
calc_metrics <- function(pred, actual) {
  cm <- confusionMatrix(pred, actual, positive = "1")
  return(c(
    Acc = round(cm$overall["Accuracy"], 3),
    Recall = round(cm$byClass["Sensitivity"], 3)
  ))
}

# Hitung metrik 
m_base  <- calc_metrics(pred_baseline, test_data$loan_status)
m_smote <- calc_metrics(pred_smote, test_data$loan_status)
m_rf    <- calc_metrics(pred_rf, test_data$loan_status)
m_xgb   <- calc_metrics(pred_xgb, as.factor(test_label))

# Gabungkan ke tabel
tabel_perbandingan$Accuracy <- c(m_base[1], m_smote[1], m_rf[1], m_xgb[1])
tabel_perbandingan$Recall   <- c(m_base[2], m_smote[2], m_rf[2], m_xgb[2])
print(tabel_perbandingan)

# 2. Confusion matrix
final_cm <- confusionMatrix(pred_smote, test_data$loan_status, positive = "1")
par(mfrow = c(1, 1), family = "serif", mar = c(2,2,2,2))
fourfoldplot(final_cm$table, color = c("darkblue", "darkorange"), 
             conf.level = 0, margin = 1, main = "")
mtext("Confusion Matrix: Model SMOTE", side = 3, line = 1, adj = 0.5, cex = 1.2, font = 2)

# Analisis ROC curve
roc_base <- roc(test_data$loan_status, prob_baseline, quiet = TRUE)
roc_smote <- roc(test_data$loan_status, prob_smote, quiet = TRUE)
roc_rf    <- roc(test_data$loan_status, prob_rf, quiet = TRUE)
roc_xgb   <- roc(test_label, prob_xgb, quiet = TRUE)

plot(roc_xgb, col = "darkorange", lwd = 4, main = "", family = "serif") 
plot(roc_smote, col = "darkblue", lwd = 2, add = TRUE)
plot(roc_rf, col = "forestgreen", lwd = 2, add = TRUE)
plot(roc_base, col = "yellow", lwd = 2, lty = 2, add = TRUE) 

abline(a = 0, b = 1, lty = 3, col = "red")
mtext("ROC Curve Comparison: All Models", side = 3, line = 1, adj = 0.5, cex = 1.4, font = 2)
legend("bottomright", 
       legend = c(paste("XGBoost (AUC:", round(auc(roc_xgb), 3), ")"),
                  paste("SMOTE (AUC:", round(auc(roc_smote), 3), ")"),
                  paste("Random Forest (AUC:", round(auc(roc_rf), 3), ")"),
                  paste("Baseline (AUC:", round(auc(roc_base), 3), ")")),
       col = c("darkorange", "darkblue", "forestgreen", "yellow"), 
       lwd = c(4, 2, 2, 2), 
       lty = c(1, 1, 1, 2),
       bty = "n",
       cex = 0.9)

# KS Statistic Analysis
ks_base  <- max(roc_base$sensitivities + roc_base$specificities - 1)
ks_smote <- max(roc_smote$sensitivities + roc_smote$specificities - 1)
ks_rf    <- max(roc_rf$sensitivities + roc_rf$specificities - 1)
ks_xgb   <- max(roc_xgb$sensitivities + roc_xgb$specificities - 1)

df_ks <- data.frame(
  Metode = c("Baseline", "SMOTE", "Random Forest", "XGBoost"),
  KS_Value = c(ks_base, ks_smote, ks_rf, ks_xgb)
)

ggplot(df_ks, aes(x = reorder(Metode, KS_Value), y = KS_Value, fill = Metode)) +
  geom_bar(stat = "identity", width = 0.7) +
  geom_text(aes(label = round(KS_Value, 3)), vjust = -0.5, fontface = "bold") +
  scale_fill_manual(values = c("Baseline"="darkred", "SMOTE"="navy", 
                               "Random Forest"="grey", "XGBoost"="darkorange")) +
  labs(title = "Perbandingan KS Statistic: Model Power",
       subtitle = "Semakin tinggi KS, semakin baik model memisahkan Good vs Bad Loan",
       x = "Metode", y = "Nilai KS") +
  theme_minimal() +
  theme(
    text = element_text(family = "serif"),
    legend.position = "none",
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5)
  ) +
  ylim(0, max(df_ks$KS_Value) * 1.2)