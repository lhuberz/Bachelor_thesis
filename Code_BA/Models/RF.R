set.seed(123)

# Fit the Random Forest model
rf_model <- ranger(
  dependent.variable.name = "CPI",
  data = train_data,
  num.trees = 500,
  importance = "impurity" )

# Make predictions on the test set
predictions <- predict(rf_model, data = test_data)$predictions

print(rf_model)


# Calculate MSE
rmse <- sqrt(mean((predictions - y_test)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

mae <- mean(abs(predictions - y_test))
print(paste("MAE:", mae))




###############################################################With CV
x_train <- model.matrix(CPI ~ . - 1, data = train_data)
y_train <- train_data$CPI

x_test <- model.matrix(CPI ~ . - 1, data = test_data)
y_test <- test_data$CPI





control <- trainControl(method = "cv", number = 10)

# Define the grid of hyperparameters to search
tunegrid <- expand.grid(
  mtry = seq(2, ncol(x_train), by = 2),
  splitrule = "variance",
  min.node.size = c(5, 10, 15, 20, 25)
)

# Train the model with hyperparameter tuning using ranger
set.seed(123)
tuned_rf <- caret::train(
  x = x_train, 
  y = y_train,
  method = 'ranger',
  tuneGrid = tunegrid,
  trControl = control,
  num.trees = 500
)

# Print the results of the hyperparameter tuning
#print(tuned_rf)

# Best hyperparameters
best_mtry <- tuned_rf$bestTune$mtry
best_min_node_size <- tuned_rf$bestTune$min.node.size

# Train the final model with the best hyperparameters
set.seed(123)
final_rf_model <- ranger(
  dependent.variable.name = "CPI",
  data = train_data,
  mtry = best_mtry,
  num.trees = 500,
  min.node.size = best_min_node_size,
  importance = "permutation"
)

# Print the final model summary
print(final_rf_model)

# Evaluate the model on the test data
predictions <- predict(final_rf_model, data = x_test)$predictions


rmse <- sqrt(mean((predictions - y_test)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

mae <- mean(abs(predictions - y_test))
print(paste("MAE:", mae))

############################################################################
##TRAIN EVAL
predictions_T <- predict(final_rf_model, data = x_train)$predictions


rmse <- sqrt(mean((predictions_T - y_train)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

mae <- mean(abs(predictions_T - y_train))
print(paste("MAE:", mae))



#########################################################################

###################################################Residuals
residuals <- (y_test - predictions)

plot(residuals)




######################Ljung Box TEST
#######################
p_values_df <- data.frame(Lag = integer(), P_Value = numeric())

# Iterate over lags from 1 to 20
for (lag in 1:20) {
  # Perform Ljung-Box test
  ljung_box_test <- Box.test(residuals, lag = lag, type = "Ljung-Box")
  
  # Store the lag and p-value in the dataframe
  p_values_df <- rbind(p_values_df, data.frame(Lag = lag, P_Value = ljung_box_test$p.value))
}
p_values_df
plot(p_values_df)

ggplot(p_values_df, aes(x= Lag, y= P_Value)) +
  geom_point() +
  geom_hline(yintercept = 0.01,        
             linetype = "solid",      
             color = "red") +          
  geom_hline(yintercept = 0.05,        
             linetype = "solid",      
             color = "blue") +
  labs(title = "Ljung Box Test results for RF", x = "Lags", y = "P-values") +
  theme_minimal()







##########################VARIABLE IMPORTANCE
#vimportance_RF <- 1

importance_values <- importance(final_rf_model)
vimportance_RF <- as.data.frame(importance_values)
vimportance_RF <- vimportance_RF %>%
  rownames_to_column(var = "Variable") %>%
  gather(key = "Metric", value = "Value", -Variable) %>%
  select(-Metric) %>%
  arrange(desc(abs(Value)))



#plot
ggplot(vimportance_RF, aes(x = abs(Value), y = reorder(Variable, abs(Value)))) +
  geom_point() +
  labs(title = "Variable Importance in Random Forest",
       x = "Absolute Coefficient Value",
       y = "Variable") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 8)) 


##########################################


