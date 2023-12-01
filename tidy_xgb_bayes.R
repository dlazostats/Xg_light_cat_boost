## Tidy Models done right
#=========================
library(dplyr)
library(tidymodels)
library(MASS)
library(doParallel)
library(bundle)
library(butcher)

# Loading Data
dt0<-Boston

# Train/Test set
set.seed(122)
split<- initial_split(dt0, strata = medv,prop=0.8)
train<-training(split)
test<-testing(split)

# CV and Recipe
folds <- vfold_cv(train,v=4,repeats=3)
mdl_rec<-recipe(medv~.,data=train) %>% step_nzv(all_predictors())

# Model specification
xgboost_model <- boost_tree(mode = "regression",
                            trees = 1000, 
                            sample_size= tune(),
                            min_n = tune(),
                            tree_depth = tune(),
                            mtry = tune(),
                            learn_rate = tune(),
                            loss_reduction = tune()) %>%
                 set_engine("xgboost",objective = "reg:squarederror")
xgboost_model

# Workflow
xgboost_wf <- workflow() %>%
              add_model(xgboost_model) %>% 
              add_recipe(mdl_rec)
xgb_params <- parameters(workflow(medv~., xgboost_model)) %>% update(mtry = finalize(mtry(), train))

# Bayesian Hyperparameter tunig
set.seed(345)
registerDoParallel()
grid_fit <- tune_bayes(xgboost_wf, 
                       resamples = folds,
                       param_info = xgb_params,
                       initial = 20, 
                       iter    = 20, 
                       metrics = metric_set(rmse), 
                       control = control_bayes(no_improve = 15, verbose = TRUE))
stopImplicitCluster()
grid_fit %>% show_best("rmse")
best_params <- grid_fit %>% select_best("rmse")

## final model
xgb_final<- xgboost_wf %>% finalize_workflow(best_params)
final_fit <- xgb_final %>% last_fit(split)
final_fit %>% collect_metrics() 

## Save model 
savemdl_xgb <-final_fit %>%            
              extract_workflow() %>%   
              butcher() %>% 
              bundle()
saveRDS(savemdl_xgb,"xgb_model.RDS")
