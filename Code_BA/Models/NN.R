set.seed(123)

train_data_scaled <- train_data %>%
  mutate(across(-CPI, scale))

test_data_scaled <- test_data %>%
  mutate(across(-CPI, scale))



x_train <- model.matrix(CPI ~ . - 1, data = train_data_scaled)
y_train <- train_data$CPI


x_test <- model.matrix(CPI ~ . - 1, data = test_data_scaled)
y_test <- test_data$CPI


############################################################################


create_model <- function(units = 32, dropout_rate = 0.2, lr = 0.001) {
  model <- keras_model_sequential() %>%
    layer_dense(units = units, activation = 'relu', input_shape = ncol(x_train)) %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 1)  # Output layer for regression
  
  # Compile the model
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = lr),  # Specify learning_rate here
    loss = 'mean_squared_error',
    metrics = list('mean_absolute_error')
  )
  
  return(model)
}


#################################################

k <- 5
set.seed(123)
folds <- createFolds(y_train, k = k, list = TRUE, returnTrain = TRUE)

# Define hyperparameter grid
hyper_grid <- expand.grid(
  units = c(32, 64, 128),
  dropout_rate = c(0.2, 0.3, 0.4),
  lr = c(0.001, 0.01, 0.1)
)

results <- data.frame()

for (i in 1:nrow(hyper_grid)) {
  units <- hyper_grid$units[i]
  dropout_rate <- hyper_grid$dropout_rate[i]
  lr <- hyper_grid$lr[i]
  
  cv_results <- data.frame(MSE = numeric(k), MAE = numeric(k))
  
  for (j in 1:k) {
    fold_train_indices <- folds[[j]]
    fold_test_indices <- setdiff(seq_len(nrow(x_train)), fold_train_indices)
    
    fold_x_train <- x_train[fold_train_indices, , drop = FALSE]
    fold_y_train <- y_train[fold_train_indices]
    fold_x_test <- x_train[fold_test_indices, , drop = FALSE]
    fold_y_test <- y_train[fold_test_indices]
    
    model <- create_model(units, dropout_rate, lr)
    history <- model %>% fit(
      fold_x_train, fold_y_train,
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2,
      verbose = 0,
      callbacks = list(callback_early_stopping(patience = 10))  ###########i included this to save some computational time
      #                                                         # the final model was build however without stopping criteria
    )
    
    fold_predictions <- model %>% predict(fold_x_test)
    cv_results$MSE[j] <- mean((fold_y_test - fold_predictions)^2)
    cv_results$MAE[j] <- mean(abs(fold_y_test - fold_predictions))
  }
  
  avg_mse <- mean(cv_results$MSE)
  avg_mae <- mean(cv_results$MAE)
  
  results <- rbind(results, data.frame(units, dropout_rate, lr, MSE = avg_mse, MAE = avg_mae))
}

# Select the best hyperparameters
best_params <- results[which.min(results$MAE), ]




best_model <- create_model(best_params$units, best_params$dropout_rate, best_params$lr)
history <- best_model %>% fit(
  x = x_train, y = y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 0
)

test_predictions <- predict(best_model, x_test)

rmse <- sqrt(mean((y_test - test_predictions)^2))
mae <- mean(abs(y_test - test_predictions))

cat("Test RMSE:", rmse, "\n")
cat("Test MAE:", mae, "\n")

#############################################################
#Training Error:

train_predictions <- predict(best_model, x_train)

rmse <- sqrt(mean((y_train - train_predictions)^2))
mae <- mean(abs(y_train - train_predictions))

cat("Test RMSE:", rmse, "\n")
cat("Test MAE:", mae, "\n")



############################################################
###############
#Without CV

create_model <- function(units = 32, dropout_rate = 0.2, lr = 0.001) {
  model <- keras_model_sequential() %>%
    layer_dense(units = units, activation = 'relu', input_shape = ncol(x_train)) %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 1)  # Output layer for regression
  
  optimizer <- optimizer_adam(learning_rate = lr)
  
  model %>% compile(
    optimizer = optimizer,
    loss = 'mean_squared_error',
    metrics = list('mean_absolute_error')
  )
  
  return(model)
}



#################################
hyper_grid <- expand.grid(
  units = c(32, 64, 128),
  dropout_rate = c(0.2, 0.3, 0.4),
  lr = c(0.001, 0.01, 0.1)
)

results <- data.frame()

for (i in 1:nrow(hyper_grid)) {
  units <- hyper_grid$units[i]
  dropout_rate <- hyper_grid$dropout_rate[i]
  lr <- hyper_grid$lr[i]
  
  model <- create_model(units, dropout_rate, lr)
  history <- model %>% fit(
    x = x_train, y = y_train,
    epochs = 100,
    batch_size = 32,
    validation_data = list(x_test, y_test),
    verbose = 0
  )
  
  val_loss <- history$metrics$val_loss[length(history$metrics$val_loss)]
  val_mae <- history$metrics$val_mean_absolute_error[length(history$metrics$val_mean_absolute_error)]
  
  results <- rbind(results, data.frame(units, dropout_rate, lr, val_loss, val_mae))
}

# Select the best hyperparameters
best_params <- results[which.min(results$val_mae), ]



#############
best_model <- create_model(best_params$units, best_params$dropout_rate, best_params$lr)
history <- best_model %>% fit(
  x = x_train, y = y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 0
)

test_predictions <- best_model %>% predict(x_test)

mse <- mean((y_test - test_predictions)^2)
mae <- mean(abs(y_test - test_predictions))

cat("Test MSE:", mse, "\n")
cat("Test MAE:", mae, "\n")



################################################### SHAPRLEY VALUES:
library(iml)


x_train <- as.data.frame(x_train)

x_interest <- as.data.frame(x_test[1, , drop = FALSE])
predictor <- Predictor$new(best_model, data = x_train, y = y_train)

shapley_nn <- Shapley$new(predictor, x.interest = x_interest)

