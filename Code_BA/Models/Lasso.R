set.seed(123)
train_data_scaled <- train_data %>%
  mutate(across(-CPI, scale))

test_data_scaled <- test_data %>%
  mutate(across(-CPI, scale))



x_train <- model.matrix(CPI ~ . - 1, data = train_data_scaled)
y_train <- train_data$CPI


x_test <- model.matrix(CPI ~ . - 1, data = test_data_scaled)
y_test <- test_data$CPI



####IMPORTANT: alpha is reversed here as compared to my paper and the literature.
# so for alpha=0 we have RR and for alpha=1 we have LASSO
#They only swapped the formula putting 1-alpha in front instead of alpha (see more with help(glmnet)

#build in cv

#lasso_model <- glmnet(x_train, y_train, alpha = 1)

# Perform cross-validation to find the optimal lambda
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)

# Get the best lambda
best_lambda <- cv_lasso$lambda.min

# Fit the final model on the training data using the best lambda
final_model <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda)


predictions <- predict(final_model, s = best_lambda, newx = x_test)

# Eval
mse <- mean((y_test - predictions)^2)
print(mse)

rmse <- sqrt(mean((predictions - y_test)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

mae <- mean(abs(predictions - y_test))
print(paste("MAE:", mae))

################################################################################
#Trainset eval:
#### Predict:

predictions_T <- predict(final_model, s = best_lambda, newx = x_train)

# Eval

rmse <- sqrt(mean((y_train - predictions_T)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

mae <- mean(abs(y_train - predictions_T))
print(paste("MAE:", mae))









###################################################################
####################################################traincontrolcv


# Set up outer cross-validation
outer_cv <- trainControl(method = "repeatedcv", number = 10, repeats = 10)



# Define the grid of hyperparameters for LASSO
lambda_grid <- expand.grid(alpha = 1, lambda = 10^seq(-4, 1, length = 100))

# Perform nested cross-validation #glmnet performs 10fold CV
lasso_model <- caret::train(
  x_train, y_train,
  method = "glmnet",
  trControl = outer_cv,
  tuneGrid = lambda_grid
)


################
best_lambda <- lasso_model$bestTune[2]
best_lambda <- best_lambda %>%
  pull()
# Use best lambda to train final Ridge Regression model
lasso_model_cv <- glmnet(x_train, y_train, alpha = 0, lambda = best_lambda)
####################################
# Make predictions on the test set
predictions <- predict(lasso_model_cv, newx = x_test)

# Evaluate the model performance
test_rmse <- sqrt(mean((predictions - y_test)^2))
print(paste("Test RMSE:", test_rmse))

test_mae <- mean(abs(predictions - y_test))
print(paste("MAE:", test_mae))


################################################### Eval Train data
predictions_T <- predict(final_model, s = best_lambda, newx = x_train)

# Eval

rmse <- sqrt(mean((y_train - predictions_T)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

mae <- mean(abs(y_train - predictions_T))
print(paste("MAE:", mae))










###FEATURE IMPORTANCE
lasso_model_cv$beta
final_model$beta

variable_importance_lasso <- broom::tidy(final_model) %>%
  slice(-1) %>%
  arrange(desc(abs(estimate)))
################################################################

ggplot(variable_importance_lasso, aes(x = abs(estimate), y = reorder(term, abs(estimate)))) +
  geom_point() +
  labs(title = "Variable Importance in LASSO Regression",
       x = "Absolute Coefficient Value",
       y = "Variable") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 8)) 





########################################################Testing a different algo
##CAN BE DISREGARDED


# Define the outer cross-validation folds
set.seed(123)
outer_folds <- createFolds(y_train, k = 5, list = TRUE)

# Initialize a vector to store the RMSE values from each outer fold
outer_rmse <- numeric(length(outer_folds))

# Loop over each outer fold
for (i in seq_along(outer_folds)) {
  # Split the data into training and validation sets for the outer loop
  outer_train_index <- unlist(outer_folds[-i])
  outer_valid_index <- unlist(outer_folds[i])
  
  outer_x_train <- x_train[outer_train_index, ]
  outer_y_train <- y_train[outer_train_index]
  outer_x_valid <- x_train[outer_valid_index, ]
  outer_y_valid <- y_train[outer_valid_index]
  
  # Define the inner cross-validation
  inner_cv <- trainControl(method = "cv", number = 5)
  
  # Define the grid of hyperparameters for LASSO
  lambda_grid <- expand.grid(alpha = 1, lambda = 10^seq(-4, 1, length = 100))
  
  # Train the LASSO model with inner cross-validation
  lasso_model <- caret::train(
    outer_x_train, outer_y_train,
    method = "glmnet",
    trControl = inner_cv,
    tuneGrid = lambda_grid
  )
  
  # Make predictions on the outer validation set
  outer_predictions <- predict(lasso_model, newdata = outer_x_valid)
  
  # Compute RMSE for the outer validation set
  outer_rmse[i] <- sqrt(mean((outer_predictions - outer_y_valid)^2))
}

# Calculate the average RMSE from the outer cross-validation
avg_outer_rmse <- mean(outer_rmse)
print(paste("Average Outer CV RMSE:", avg_outer_rmse))

# Train the final model on the full training data using the best lambda found
final_model <- caret::train(
  x_train, y_train,
  method = "glmnet",
  trControl = inner_cv,
  tuneGrid = lambda_grid
)

# Make predictions on the test set
final_predictions <- predict(final_model, newdata = x_test)

# Evaluate the model performance on the test set
test_rmse <- sqrt(mean((final_predictions - y_test)^2))
print(paste("Test RMSE:", test_rmse))
