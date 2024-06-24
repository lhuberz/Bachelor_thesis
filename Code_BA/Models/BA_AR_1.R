AR_model5 <- arima(train_data$CPI, order = c(5, 0, 0))
summary(AR_model5)
forecasts_AR5 <- forecast(AR_model5, h=nrow(test_data))

plot(forecasts_AR5)

forecasts_AR_df5 <- as.data.frame(forecasts_AR5)

forecast_mean_AR5 <- forecasts_AR_df5$`Point Forecast`

# Extract actual values from the test data
actual_values <- test_data$CPI
#####################################################

AR_model4 <- arima(train_data$CPI, order = c(4, 0, 0))
summary(AR_model4)
forecasts_AR4 <- forecast(AR_model4, h=nrow(test_data))

plot(forecasts_AR4)

forecasts_AR_df4 <- as.data.frame(forecasts_AR4)

forecast_mean_AR4 <- forecasts_AR_df4$`Point Forecast`

########################################################

AR_model3 <- arima(train_data$CPI, order = c(3, 0, 0))
summary(AR_model3)
forecasts_AR3 <- forecast(AR_model3, h=nrow(test_data))

plot(forecasts_AR3)

forecasts_AR_df3 <- as.data.frame(forecasts_AR3)

forecast_mean_AR3 <- forecasts_AR_df3$`Point Forecast`

########################################################

AR_model2 <- arima(train_data$CPI, order = c(2, 0, 0))
summary(AR_model2)
forecasts_AR2 <- forecast(AR_model2, h=nrow(test_data))

plot(forecasts_AR2)

forecasts_AR_df2 <- as.data.frame(forecasts_AR2)

forecast_mean_AR2 <- forecasts_AR_df2$`Point Forecast`

######################################################## AR_model1 smallest AIC

AR_model1 <- arima(train_data$CPI, order = c(1, 0, 0))
summary(AR_model1)
forecasts_AR1 <- forecast(AR_model1, h=nrow(test_data))

plot(forecasts_AR1)

forecasts_AR_df1 <- as.data.frame(forecasts_AR1)

forecast_mean_AR1 <- forecasts_AR_df1$`Point Forecast`


actual_values <- y_test


mae <- mean(abs(forecast_mean_AR1 - actual_values))
cat("Mean Absolute Error (MAE):", mae, "\n")

# Calculate Root Mean Squared Error (RMSE)
rmse <- sqrt(mean((forecast_mean_AR1 - actual_values)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")
###############################################################
#Forecast training
forecasts_AR1_T <- forecast(AR_model1, h=nrow(train_data))


forecasts_AR_df1_T <- as.data.frame(forecasts_AR1_T)

forecast_mean_AR1_T <- forecasts_AR_df1_T$`Point Forecast`
##

mae <- mean(abs(forecast_mean_AR1_T - train_data$CPI))
cat("Mean Absolute Error (MAE):", mae, "\n")

# Calculate Root Mean Squared Error (RMSE)
rmse <- sqrt(mean((forecast_mean_AR1_T - train_data$CPI)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n")




############################################

residuals_AR <- actual_values - forecast_mean_AR1
plot(residuals_AR)

ljung_box_test <- Box.test(residuals_AR, lag = 6, type = "Ljung-Box")

# Print the test results
print(ljung_box_test)



p_values_df_AR <- data.frame(Lag = integer(), P_Value = numeric())

# Iterate over lags from 1 to 20
for (lag in 1:20) {
  # Perform Ljung-Box test
  ljung_box_test <- Box.test(residuals_AR, lag = lag, type = "Ljung-Box")
  
  # Store the lag and p-value in the dataframe
  p_values_df_AR <- rbind(p_values_df, data.frame(Lag = lag, P_Value = ljung_box_test$p.value))
}



############################################################################
##########################CROSS VALIDATION:#################################
set.seed(123)



