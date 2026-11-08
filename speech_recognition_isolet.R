# ==============================================================================
# PROJEKT ISOLET: KLASYFIKACJA SYGNAŁÓW AUDIO (MASTER SCRIPT)
# Autorzy: Mateusz Panek, Konrad Serocki
# Opis: Pełna historia eksperymentów, diagnostyka nieliniowości i model końcowy.
# ==============================================================================

# 1. KONFIGURACJA I BIBLIOTEKI
#setwd('C:\\Users\\seroc\\Downloads')

library(tidyverse)
library(tidymodels)
library(doParallel)
library(kernlab)   # SVM
library(ranger)    # Random Forest
library(xgboost)   # XGBoost
library(baguette)  # Bagging
library(stacks)    # Stacking
library(vip)       # Ważność zmiennych
library(ggplot2)

# Turbo Mode (rdzenie - 1 dla stabilności systemu)
cores <- parallel::detectCores(logical = FALSE)
cl <- makePSOCKcluster(cores - 1)
registerDoParallel(cl)
message(paste0("Rdzenie aktywne: ", cores - 1))

# --- UNIWERSALNA FUNKCJA SMART_RUN ---
smart_run <- function(file_name, expr) {
  if (file.exists(file_name)) {
    message("Wczytano: ", file_name)
    return(readRDS(file_name))
  } else {
    message("Sartuję proces dla: ", file_name)
    result <- eval(expr) 
    saveRDS(result, file_name)
    return(result)
  }
}

# ==============================================================================
# 2. PRZYGOTOWANIE DANYCH I RECEPTUR
# ==============================================================================
if(!file.exists("isolet_data.csv")) stop("Nie znaleziono pliku isolet_data.csv!")

data_raw <- read.csv("isolet_data.csv")
data_clean <- data_raw %>% 
  select(-matches("^X$")) %>% 
  mutate(klasa = as.factor(klasa))

set.seed(123) 
split <- initial_split(data_clean, prop = 0.8, strata = klasa)
train_data <- training(split)
test_data  <- testing(split)
cv_folds   <- vfold_cv(train_data, v = 5, strata = klasa)

# Receptury (Preprocessing)
rec_basic <- recipe(klasa ~ ., data = train_data) %>% step_nzv(all_predictors())
rec_norm  <- rec_basic %>% step_normalize(all_numeric_predictors())

# ==============================================================================
# 3. EKSPERYMENTY
# ==============================================================================

# 3.1. Regresja Logistyczna (Baseline)
fit_lr <- smart_run("model_isolet_final.rds", {
  last_fit(workflow() %>% add_recipe(rec_norm) %>% 
             add_model(multinom_reg(penalty = 0.01) %>% set_engine("glmnet")), split)
})

# 3.2. Random Forest (Baseline)
fit_rf <- smart_run("model_rf_final.rds", {
  last_fit(workflow() %>% add_recipe(rec_basic) %>% 
             add_model(rand_forest(trees = 500) %>% set_engine("ranger", importance = "impurity")), split)
})

# 3.3. XGBoost (zakomentowany)
# fit_xgb <- smart_run("model_xgb_attempt.rds", {
#   last_fit(workflow() %>% add_recipe(rec_basic) %>% 
#            add_model(boost_tree(trees = 500) %>% set_engine("xgboost")), split)
# })

# 3.4. SVM Etap 1 (Auto Grid)
fit_svm_auto <- smart_run("model_svm.rds", {
  spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>% set_engine("kernlab") %>% set_mode("classification")
  res  <- tune_grid(workflow() %>% add_recipe(rec_norm) %>% add_model(spec), 
                    resamples = cv_folds, grid = grid_regular(cost(), rbf_sigma(), levels = 3))
  last_fit(finalize_workflow(workflow() %>% add_recipe(rec_norm) %>% add_model(spec), select_best(res, "accuracy")), split)
})

# 3.5. SVM Etap 2 (Snajper - Manualny Grid)
fit_svm_final <- smart_run("model_svm_manual.rds", {
  spec <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>% set_engine("kernlab") %>% set_mode("classification")
  grid_manual <- expand.grid(cost = c(20, 32, 45, 60), rbf_sigma = c(0.000008, 0.00001, 0.00002))
  res  <- tune_grid(workflow() %>% add_recipe(rec_norm) %>% add_model(spec), resamples = cv_folds, grid = grid_manual)
  last_fit(finalize_workflow(workflow() %>% add_recipe(rec_norm) %>% add_model(spec), select_best(res, "accuracy")), split)
})

# 3.6. Bagging
fit_bag <- smart_run("model_bagging.rds", {
  spec <- bag_tree() %>% set_engine("rpart", times = 25) %>% set_mode("classification")
  last_fit(workflow() %>% add_recipe(rec_basic) %>% add_model(spec), split)
})

# 3.7. Stacking (SVM + RF)
fit_stack <- smart_run("model_stacking_final.rds", {
  ctrl <- control_stack_resamples()
  m1   <- fit_resamples(extract_workflow(fit_svm_final), resamples = cv_folds, control = ctrl)
  m2   <- fit_resamples(workflow() %>% add_recipe(rec_basic) %>% 
                          add_model(rand_forest(trees = 500) %>% set_engine("ranger")), resamples = cv_folds, control = ctrl)
  stacks() %>% add_candidates(m1) %>% add_candidates(m2) %>% blend_predictions() %>% fit_members()
})

# ==============================================================================
# 4. RANKING KOŃCOWY
# ==============================================================================
get_acc <- function(fit_obj) {
  if (inherits(fit_obj, "linear_stack")) {
    preds <- predict(fit_obj, test_data) %>% bind_cols(test_data)
    return(accuracy(preds, truth = klasa, estimate = .pred_class)$.estimate)
  }
  collect_metrics(fit_obj) %>% filter(.metric == "accuracy") %>% pull(.estimate)
}

ranking <- tibble(
  Model = c("SVM Manual", "SVM Auto", "Stacking", "Random Forest", "Bagging", "Logistic Regression"),
  Accuracy = map_dbl(list(fit_svm_final, fit_svm_auto, fit_stack, fit_rf, fit_bag, fit_lr), get_acc)
) %>% arrange(desc(Accuracy))

cat("\n--- RANKING SKUTECZNOŚCI (ZBIÓR TESTOWY) ---\n")
print(ranking)

# ==============================================================================
# 5. DIAGNOSTYKA I AUDYT MODELU
# ==============================================================================
message("Generowanie diagnostyki dla modelu zwycięskiego (SVM Manual)...")

# 5.1. Analiza Overfittingu
final_wf_model <- extract_workflow(fit_svm_final)
train_results  <- predict(final_wf_model, new_data = train_data) %>% bind_cols(train_data %>% select(klasa))
test_results   <- collect_predictions(fit_svm_final)

acc_train <- accuracy(train_results, truth = klasa, estimate = .pred_class)
acc_test  <- accuracy(test_results, truth = klasa, estimate = .pred_class)

cat("\n--- AUDYT OVERFITTINGU (Manual) ---\n")
cat("Accuracy (TRENING): ", round(acc_train$.estimate * 100, 2), "%\n")
cat("Accuracy (TEST):    ", round(acc_test$.estimate * 100, 2), "%\n")
cat("Różnica (Gap):      ", round((acc_train$.estimate - acc_test$.estimate) * 100, 2), " pkt proc.\n")

# 5.2. Macierz Pomyłek
cm_plot <- test_results %>%
  conf_mat(truth = klasa, estimate = .pred_class) %>%
  autoplot(type = "heatmap") +
  labs(title = "Macierz Pomyłek - Zbiór Testowy",
       subtitle = "Model SVM Manual") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
print(cm_plot)

# 5.3. Wykres VIP (Ważność zmiennych na bazie Random Forest)
vip_plot <- extract_workflow(fit_rf) %>%
  extract_fit_parsnip() %>%
  vip(num_features = 20, geom = "point", aesthetics = list(color = "#800080", size = 3)) +
  theme_minimal() +
  labs(title = "Top 20 Predyktorów (Ranking Ważności)", 
       subtitle = "Analiza wkładu zmiennych audio (Model Random Forest)") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))
print(vip_plot)

# ==============================================================================
# 6. MODEL PRODUKCYJNY (TRENOWANY NA 100% DANYCH)
# ==============================================================================
message("Trenowanie modelu produkcyjnego na 100% danych...")

model_produkcyjny <- fit(extract_workflow(fit_svm_final), data = data_clean)
saveRDS(model_produkcyjny, "model_production.rds")


stopCluster(cl); registerDoSEQ()