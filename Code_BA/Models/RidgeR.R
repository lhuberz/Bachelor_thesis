
set.seed(123)
#non-scaled
#x_train <- model.matrix(CPI ~ . - 1, data = train_data)
#y_train <- train_data$CPI


#x_test <- model.matrix(CPI ~ . - 1, data = test_data)
#y_test <- test_data$CPI




###scaled
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
#GLMnet only swapped the formula putting 1-alpha in front instead of alpha (see more with help(glmnet)
##################################################################Hyperparameter
## optimization with CV

# Set up cross-validation
set.seed(123)  # for reproducibility
cv <- cv.glmnet(x_train, y_train, alpha = 0, type.measure = "mse", nfolds = 10)

# Plot cross-validation results


# Identify the best lambda
best_lambda <- cv$lambda.min
best_lambda
# Use best lambda to train final Ridge Regression model
ridge_model_cv <- glmnet(x_train, y_train, alpha = 0, lambda = best_lambda)

# Predict on test set
predictions_cv <- predict(ridge_model_cv, newx = x_test)

# Evaluate performance on test set
rmse_cv <- sqrt(mean((y_test - predictions_cv)^2))
rmse_cv

mae <- mean(abs(y_test - predictions_cv))
cat("Mean Absolute Error (MAE):", mae, "\n")




########################################################NESTED CV
set.seed(123)

control <- trainControl(method = "cv", number = 10) # 10-fold cross-validation outer

# Train the model using caret's train function
ridge_model_cv <- caret::train(
  x = x_train,
  y = y_train,
  method = "glmnet",
  trControl = control,
  tuneGrid = expand.grid(alpha = 0, lambda = seq(0.001, 1, length = 100))  # Grid search over lambda
)

# Print the cross-validation results
#print(ridge_model_cv)

best_lambda <- ridge_model_cv$bestTune[2]
best_lambda <- best_lambda %>%
  pull()
# Use best lambda to train final Ridge Regression model
ridge_model_cv <- glmnet(x_train, y_train, alpha = 0, lambda = best_lambda)

predictions_cv <- predict(ridge_model_cv, newx = x_test)

rmse <- sqrt(mean((predictions - y_test)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

mae <- mean(abs(y_test - predictions))
print(paste("MAE:", mae))

##########################################################################

predictions_cv_T <- predict(ridge_model_cv, newx = x_train)

# Evaluate performance on test set
rmse_cv <- sqrt(mean((y_train - predictions_cv_T)^2))
rmse_cv

mae <- mean(abs(y_train - predictions_cv_T))
cat("Mean Absolute Error (MAE):", mae, "\n")


######################################
###FEATURE IMPORTANCE


ridge_model_cv$beta

variable_importance <- broom::tidy(ridge_model_cv) %>%
  arrange(desc(abs(estimate)))
################################################################

ggplot(variable_importance, aes(x = abs(estimate), y = reorder(term, abs(estimate)))) +
  geom_point() +
  labs(title = "Variable Importance in Ridge Regression",
       x = "Absolute Coefficient Value",
       y = "Variable") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 8)) 
