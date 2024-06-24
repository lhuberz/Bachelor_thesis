
############TESTING DIFFERENT MODEL CONFIGURATIONS FROM c(0,1,0) to c(5,1,5) comparing the AIC
arima_model <- arima(train_data$CPI, order = c(2, 1, 3))
summary(arima_model)

forecast_values_arima <- forecast(arima_model, h = nrow(test_data))

forecast_arima <- as.data.frame(forecast_values_arima)


forecast_mean_arima <- forecast_arima$`Point Forecast`

# Extract actual values from the test data
actual_values <- test_data$CPI

# Calculate Mean Absolute Error (MAE)
mae <- mean(abs(forecast_mean_arima - actual_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# Calculate Root Mean Squared Error (RMSE)
rmse <- sqrt(mean((forecast_mean_arima - actual_values)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")

# Calculate Mean Absolute Percentage Error (MAPE)
mape <- mean(abs((actual_values - forecast_mean_arima) / actual_values)) * 100
cat("Mean Absolute Percentage Error (MAPE):", mape, "%\n")


##################################################################################
#TRAINING ERROR:
forecast_values_arima_T <- forecast(arima_model, h = nrow(train_data))

forecast_arima_T <- as.data.frame(forecast_values_arima_T)


forecast_mean_arima_T <- forecast_arima_T$`Point Forecast`


# Calculate Mean Absolute Error (MAE)
mae <- mean(abs(forecast_mean_arima_T - train_data$CPI))
cat("Mean Absolute Error (MAE):", mae, "\n")

# Calculate Root Mean Squared Error (RMSE)
rmse <- sqrt(mean((forecast_mean_arima_T - train_data$CPI)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")



###################################################################################


residuals_arima <- actual_values - forecast_mean_arima
plot(residuals_arima)

ljung_box_test <- Box.test(residuals_arima, lag = 1, type = "Ljung-Box")

# Print the test results
print(ljung_box_test)

plot(train_data$CPI)

