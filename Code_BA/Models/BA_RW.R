set.seed(123)

sum(is.na(train_data$CPI)) # Should be 0
sum(is.na(test_data$CPI))  # Should be 0

y_train = train_data$CPI
y_test = test_data$CPI



######################################################################
#################FORECASTING FOR TEST SET
random_walk_model <- rwf(y_train, h=nrow(test_data), drift = FALSE)

#model <- arima(train_data$CPI, order = c(0, 1, 0))
plot(random_walk_model)


# Convert forecasts to a data frame
forecasts_df <- as.data.frame(random_walk_model)






forecast_mean <- forecasts_df$`Point Forecast`

# Extract actual values from the test data
actual_values <- y_test

# Calculate Mean Absolute Error (MAE)
mae <- mean(abs(actual_values - forecast_mean))
cat("Mean Absolute Error (MAE):", mae, "\n")

# Calculate Root Mean Squared Error (RMSE)
rmse <- sqrt(mean((actual_values - forecast_mean)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

# Calculate Mean Absolute Percentage Error (MAPE)
mape <- mean(abs((actual_values - forecast_mean) / actual_values)) * 100
cat("Mean Absolute Percentage Error (MAPE):", mape, "%\n")



######################################################################
##################FORECASTING ON TRAIN SET
random_walk_model1 <- rwf(y_train, h=156, drift = FALSE)
forecasts_df1 <- as.data.frame(random_walk_model1)


forecast_mean1 <- forecasts_df1$`Point Forecast`

# Extract actual values from the test data
actual_values1 <- y_train

# Calculate Mean Absolute Error (MAE)
mae <- mean(abs(actual_values1 - forecast_mean1))
cat("Mean Absolute Error (MAE):", mae, "\n")

# Calculate Root Mean Squared Error (RMSE)
rmse <- sqrt(mean((actual_values1 - forecast_mean1)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

#################################################

residuals <- actual_values - forecast_mean
plot(residuals)

ljung_box_test <- Box.test(residuals, lag = 20, type = "Ljung-Box")

# Print the test results
print(ljung_box_test)



p_values_df <- data.frame(Lag = integer(), P_Value = numeric())

# Iterate over lags from 1 to 20
for (lag in 1:20) {
  # Perform Ljung-Box test
  ljung_box_test <- Box.test(residuals, lag = lag, type = "Ljung-Box")
  
  # Store the lag and p-value in the dataframe
  p_values_df <- rbind(p_values_df, data.frame(Lag = lag, P_Value = ljung_box_test$p.value))
}

plot(p_values_df)
############################################################################
##########################CROSS VALIDATION:#################################


