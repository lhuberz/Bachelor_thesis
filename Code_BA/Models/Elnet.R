set.seed(123)

train_data_scaled <- train_data %>%
  mutate(across(-CPI, scale))

test_data_scaled <- test_data %>%
  mutate(across(-CPI, scale))



x_train <- model.matrix(CPI ~ . - 1, data = train_data_scaled)
y_train <- train_data$CPI


x_test <- model.matrix(CPI ~ . - 1, data = test_data_scaled)
y_test <- test_data$CPI



#############################################################

cv_fit <- cv.glmnet(x_train, y_train, alpha = 0.5)
best_lambda <- cv_fit$lambda.min
print(best_lambda)

final_model <- glmnet(x_train, y_train, alpha = 0.5, lambda = best_lambda)

# Make predictions on the test set
predictions <- predict(final_model, s = best_lambda, newx = x_test)


mse <- mean((predictions - y_test)^2)
print(mse)

rmse <- sqrt(mean((predictions - y_test)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

mae <- mean(abs(predictions - y_test))
print(paste("MAE:", mae))

#############################################################
#Training eval
#prediction on the train set
predictions_T <- predict(final_model, s = best_lambda, newx = x_train)


mse <- mean((predictions - y_test)^2)
print(mse)

rmse <- sqrt(mean((predictions_T - y_train)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

mae <- mean(abs(predictions_T - y_train))
print(paste("MAE:", mae))




#######################################################CV with traincontrol
#

outer_cv <- trainControl(method = "repeatedcv", number = 10, repeats =10)



# Define the grid of hyperparameters for LASSO
lambda_grid <- expand.grid(alpha = 0.5, lambda = seq(0.001, 1, length = 100))

# Perform nested cross-validation #glmnet performs 10fold CV
Elnet_model <- caret::train(
  x_train, y_train,
  method = "glmnet",
  trControl = outer_cv,
  tuneGrid = lambda_grid
)




best_lambda <- Elnet_model$bestTune[2]
best_lambda <- best_lambda %>%
  pull()
# Use best lambda to train final Ridge Regression model
Elnet_cv <- glmnet(x_train, y_train, alpha = 0, lambda = best_lambda)

# Make predictions on the test set
predictions <- predict(Elnet_cv, newx = x_test)

# Evaluate the model performance
test_rmse <- sqrt(mean((predictions - y_test)^2))
print(paste("Test RMSE:", test_rmse))


test_mae <- mean(abs(predictions - y_test))
print(paste("MAE:", test_mae))




###########################################################################
#feature importance

final_model$beta

variable_importance_Elnet <- broom::tidy(final_model) %>%
  slice(-1) %>%
  arrange(desc(abs(estimate)))
################################################################

ggplot(variable_importance_Elnet, aes(x = abs(estimate), y = reorder(term, abs(estimate)))) +
  geom_point() +
  labs(title = "Variable Importance in Elnet Regression without CPI Energy",
       x = "Absolute Coefficient Value",
       y = "Variable") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 8)) 


