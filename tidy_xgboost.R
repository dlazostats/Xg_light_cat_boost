### XGboost tidymodels
### Autor: Diego Lazo
### Fecha: 09/08/2022
#-------------------------------------------------
# Librerias
library(dplyr)
library(readxl)
library(dials)
library(tune)
library(yardstick)
library(tidymodels)
library(MLmetrics)
library(parsnip)

# Working directory
script_name <- 'tidy_xgboost.R'
ruta <- gsub(rstudioapi::getActiveDocumentContext()$path,pattern = script_name,replacement = '')
setwd(ruta)

# Data
db<-read.csv("data_perf_betterv1.csv",sep=",")
db_0<-db %>% select(-fecha,-dim,-peso,-dim_equiv,-year)
db_f<-sapply(db_0,as.numeric) %>% as.data.frame()
dbm<-db_f %>% select(-"fluenc",-"res_trac") #fluenc #res_trac alarg

# Train/test set
set.seed(3456)
split<-initial_split(dbm,prop=0.8,strata = alarg)
train<-training(split)
test<-testing(split)

# Recipe and CV
ml_folds <- vfold_cv(train,v=5)

# Model Specification
xgboost_model <- boost_tree(mode = "regression",
                            trees = 1000,
                            min_n = tune(),
                            tree_depth = tune(),
                            learn_rate = tune(),
                            loss_reduction = tune()) %>%
                 set_engine("xgboost",
                            objective = "reg:squarederror")

# grid specification
xgboost_params <- parameters(min_n(),
                             tree_depth(),
                             learn_rate(),
                             loss_reduction())
xgboost_grid <- grid_max_entropy(xgboost_params, 
                                 size = 60)
# Workflow
xgboost_wf <- workflow() %>%
              add_model(xgboost_model) %>% 
              add_formula(alarg ~ .)

# Tune Model
xgboost_tuned <- tune_grid( object = xgboost_wf,
                            resamples = ml_folds,
                            grid = xgboost_grid,
                            metrics = metric_set(rmse, rsq, mae),
                            control = control_grid(verbose = TRUE))

# Best metrics
xgboost_tuned %>% show_best(metric = "rmse") 
xgboost_best_params <- xgboost_tuned %>% select_best("rmse")
xgboost_model_final <- xgboost_model %>% finalize_model(xgboost_best_params)

# Evaluate performance 
#=======================
## Train
prep_recp <- recipe(alarg ~ .,
                    data = training(split)) %>% 
                    prep()
train_processed <- bake(prep_recp,
                        new_data = training(split))
train_prediction <- xgboost_model_final %>%
                    fit(formula = alarg ~ ., 
                        data    = train_processed) %>%
                    predict(new_data = train_processed) %>%
                    bind_cols(training(split))
xgboost_score_train <- train_prediction %>%
                       metrics(alarg, .pred) %>%
                       mutate(.estimate = format(round(.estimate, 2),
                                                 big.mark = ","))
xgboost_score_train

## Test
test_processed <- bake(prep_recp,
                       new_data = testing(split))
test_prediction <- xgboost_model_final %>%
                   fit(formula = alarg ~ ., 
                        data    = test_processed) %>%
                   predict(new_data = test_processed) %>%
                   bind_cols(testing(split))
xgboost_score_test <- test_prediction %>%
                      metrics(alarg, .pred) %>%
                      mutate(.estimate = format(round(.estimate, 2),
                                                big.mark = ","))
xgboost_score_test




























