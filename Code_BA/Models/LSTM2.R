install.packages("remotes")
remove.packages("keras")
library(remotes)
remotes::install_github("rstudio/tensorflow")

install.packages("keras3") # or remotes::install_github("rstudio/keras")

library(keras3) 

library(tensorflow)
install_tensorflow(envname = "r-tensorflow")


######LSTM: DATA

set.seed(123)
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
train_data <- train_data %>% mutate(across(everything(), normalize))
test_data <- test_data %>% mutate(across(everything(), normalize))

###############Sequence creation###############################################
create_sequences <- function(data, sequence_length) {
  x <- array(NA, dim = c(nrow(data) - sequence_length, sequence_length, ncol(data) - 1))
  y <- numeric(nrow(data) - sequence_length)
  for (i in 1:(nrow(data) - sequence_length)) {
    x[i,,] <- as.matrix(data[i:(i + sequence_length - 1), -which(names(data) == "CPI")])
    y[i] <- data[i + sequence_length, which(names(data) == "CPI")]
  }
  return(list(x = x, y = y))
}


sequence_length <- 10
target_column <- "CPI"  # Specify the column name for the target variable
train_sequences <- create_sequences(train_data, sequence_length, target_column)
test_sequences <- create_sequences(test_data, sequence_length, target_column)


x_train <- train_sequences$x
y_train <- train_sequences$y
x_test <- test_sequences$x
y_test <- test_sequences$y




cat("x_train shape: ", dim(train_data), "\n")
cat("y_train shape: ", length(test_data), "\n")
cat("x_test shape: ", dim(x_test), "\n")
cat("y_test shape: ", length(y_test), "\n")

###############################LSTM MODEL
build_model <- function(lstm_units, dense_units, dropout_rate) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = lstm_units, input_shape = c(sequence_length, ncol(train_data) - 1)) %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = dense_units, activation = 'relu') %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_adam()
  )
  
  return(model)
}



#########################################HYPERparameter optimization kfoldCV

hyper_grid <- expand.grid(
  lstm_units = c(50, 100),
  dense_units = c(20, 50),
  dropout_rate = c(0.2, 0.5)
)

k <- 5
set.seed(123)
cv_splits <- vfold_cv(train_data, v = k)

results <- data.frame()

for (i in 1:nrow(hyper_grid)) {
  params <- hyper_grid[i, ]
  fold_metrics <- c()
  
  for (j in 1:k) {
    cv_train <- analysis(cv_splits$splits[[j]])
    cv_test <- assessment(cv_splits$splits[[j]])
    
    cv_train_sequences <- create_sequences(cv_train, sequence_length)
    cv_test_sequences <- create_sequences(cv_test, sequence_length)
    
    x_cv_train <- cv_train_sequences$x
    y_cv_train <- cv_train_sequences$y
    x_cv_test <- cv_test_sequences$x
    y_cv_test <- cv_test_sequences$y
    
    model <- build_model(params$lstm_units, params$dense_units, params$dropout_rate)
    
    history <- model %>% fit(
      x = x_cv_train, y = y_cv_train,
      epochs = 50, batch_size = 32,
      validation_data = list(x_cv_test, y_cv_test),
      verbose = 0
    )
    
    metrics <- model %>% evaluate(
      x = x_cv_test, y = y_cv_test,
      verbose = 0
    )
    fold_metrics <- c(fold_metrics, metrics)
  }
  
  results <- rbind(results, c(params, mean(fold_metrics)))
}

# Get the best hyperparameters
best_params <- results[which.min(results$mean), ]


final_model <- build_model(best_params$lstm_units, best_params$dense_units, best_params$dropout_rate)

final_history <- final_model %>% fit(
  x = x_train, y = y_train,
  epochs = 50, batch_size = 32,
  validation_data = list(x_test, y_test)
)

test_metrics <- final_model %>% evaluate(
  x = x_test, y = y_test
)
print(test_metrics)

last_sequence <- tail(test_data, sequence_length) %>% 
  select(-CPI) %>%
  as.matrix() %>%
  array(dim = c(1, sequence_length, ncol(test_data) - 1))

# Predict the next CPI value
predicted_cpi <- final_model %>% predict(last_sequence)
print(predicted_cpi)