### Predicción tiempo de espera de una embarcación
### Autor: Diego Lazo
### Fecha: 22/04/2022
#-------------------------------------------------
# Librerias
library(dplyr)
library(readxl)
library(missForest)
library(data.table)
library(RLightGBM)
library(mltools)
library(lightgbm)
library(catboost)
library(MLmetrics)

# Working directory
script_name <- 'predict_wt.R'
ruta <- gsub(rstudioapi::getActiveDocumentContext()$path,pattern = script_name,replacement = '')
setwd(ruta)

# Data
dbte0<-read_xlsx("tiempo_espera.xlsx") %>% as.data.frame()

# Preprocessing
## Imputation
dbte0$Planta<-factor(dbte0$Planta)
imp_res<-missForest(dbte0)
dbte2<-imp_res$ximp

## Anomaly detection
distances <- mahalanobis(x = dbte2[,c(1:5)] ,
                         center = colMeans(dbte2[,c(1:5)]),
                         cov = cov(dbte2[,c(1:5)]))
cutoff <- qchisq(p = 0.95 , df = ncol(dbte2)-1)
dbte3<-dbte2[distances < cutoff ,]

## One hot enconding
df.encoded <- one_hot(as.data.table(dbte3)) %>% as.data.frame()
df.cat <- dbte3

## ML modelling
##--------------
### Train/test set
set.seed(3456)
trainIndex <- createDataPartition(df.encoded$tiempo_esp, p = .7,list=F)
train <- df.encoded[trainIndex,]
test  <- df.encoded[-trainIndex,]

## Train Control
ctrl <- trainControl(method = "repeatedcv",search = "random",
                     number = 5, repeats = 5,allowParallel = TRUE)
x_db<-train %>% dplyr::select(-"tiempo_esp") %>% as.matrix()
y_db<- train %>% dplyr::select("tiempo_esp") %>% as.matrix()
x_db_test <- test %>% dplyr::select(-"tiempo_esp") %>% as.matrix()
y_db_test <- test %>% dplyr::select("tiempo_esp") %>% as.matrix()
  
### XGboost
xgb_model <- train(x = x_db, 
                   y = y_db[,1],
                   method = "xgbTree",
                   tuneLength =10,
                   trControl = ctrl)
pred_xgb <-predict(xgb_model, x_db_test)
fit_xgb <- predict(xgb_model,x_db)

### Lightgbm
dtrain <- lgb.Dataset(x_db, label = y_db)
dtest <- lgb.Dataset.create.valid(dtrain, data = x_db_test, label = y_db_test)
valids <- list(eval = dtest, train = dtrain)
train_params <- list(learning_rate = 0.1,num_iterations=100,
                     objective = "regression")
lgb_model <- lgb.train(params = train_params,
                       dtrain, 
                       valids = valids)
pred_light <- predict(lgb_model, x_db_test)
fit_light <- predict(lgb_model,x_db)

### Catboost
set.seed(3456)
trainIndexc <- createDataPartition(df.cat$tiempo_esp, p = .7,list=F)
trainc <- df.cat[trainIndex,]
testc  <- df.cat[-trainIndex,]
x_db<-trainc %>% dplyr::select(-"tiempo_esp") 
y_db<- trainc %>% dplyr::select("tiempo_esp")
x_db_test<-testc %>% dplyr::select(-"tiempo_esp")
grid <- expand.grid(depth = c(4, 6, 8),
                    learning_rate = 0.1,
                    iterations = 200,
                    l2_leaf_reg = 1e-3,
                    rsm = 0.95,
                    border_count = 64)
cat_model <- train(x_db, y_db$tiempo_esp,
                   method = catboost.caret,
                   logging_level = 'Verbose', 
                   preProc = NULL,
                   tuneGrid = grid, 
                   trControl = ctrl)
pred_cat <- predict(cat_model, x_db_test)
fit_cat <- predict(cat_model, trainc)

## Metrics
#----------
## On Train set
df_rst<-data.frame(model=c("XGBoost","Lightgbm","Catboost"),
                  rmse=c(RMSE(fit_xgb,train$tiempo_esp),RMSE(fit_light,train$tiempo_esp),RMSE(fit_cat,train$tiempo_esp)),
                  MAE=c(MAE(fit_xgb,train$tiempo_esp),MAE(fit_light,train$tiempo_esp),MAE(fit_cat,train$tiempo_esp)),
                  R2=c(R2(fit_xgb,train$tiempo_esp),R2(fit_light,train$tiempo_esp),R2(fit_cat,train$tiempo_esp)))
df_rst

## On test set
df_rs<-data.frame(model=c("XGBoost","Lightgbm","Catboost"),
                  rmse=c(RMSE(pred_xgb,test$tiempo_esp),RMSE(pred_light,test$tiempo_esp),RMSE(pred_cat,test$tiempo_esp)),
                  MAE=c(MAE(pred_xgb,test$tiempo_esp),MAE(pred_light,test$tiempo_esp),MAE(pred_cat,test$tiempo_esp)),
                  R2=c(R2(pred_xgb,test$tiempo_esp),R2(pred_light,test$tiempo_esp),R2(pred_cat,test$tiempo_esp)))
df_rs


