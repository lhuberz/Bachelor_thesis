


library('tidyverse')
library(dplyr)
library('lubridate')
library(mlr3)
library(mlr3learners)
library(skimr)
library(mlr3viz)
library(rpart.plot)
library(tseries)
library(forecast)
library(forecastHybrid)
library(Metrics)
library(caret)
library(glmnet)
library(randomForest)
library(ranger)
library(keras)
library(mlr3verse)
library(tensorflow)
library(rsample)
library(vip)


dat_test <- read_csv("bbk_monthly_en.csv") %>%
  tibble(dat_test)

dat1 <- read_csv("bbk_monthly_en.csv")
dat1 <- as_tibble(dat1)
dat3 <- read_csv("bbk_daily_en.csv")





############################################AVERAGING DAILY DATA'###############
dat3 <- as_tibble(dat3)

names(dat3)[1] <- "date"
names(dat3)[2] <- "bond10y_percent"
names(dat3)[4] <- "bond5y_percent"
names(dat3)[6] <- "bond2y_percent"

dat3_new <- dat3 %>%
  slice(5:8251) %>%
  select(date, bond10y_percent, bond5y_percent, bond2y_percent)

dat3_new <- dat3_new %>%
  mutate(bond10y_percent = as.numeric(bond10y_percent),
         bond5y_percent = as.numeric(bond5y_percent),
         bond2y_percent = as.numeric(bond2y_percent))

dat3_monthly <- dat3_new %>%
  mutate(date = as.Date(date)) %>%
  group_by(year = year(date), month = month(date)) %>%
  summarize(
    bond10y_percent = round(mean(bond10y_percent, na.rm = TRUE), 2), 
    bond5y_percent = round(mean(bond5y_percent, na.rm = TRUE), 2),
    bond2y_percent = round(mean(bond2y_percent, na.rm = TRUE), 2)
    )%>%
  ungroup()
  

dat3_monthly$date <- format(as.Date(paste(dat3_monthly$year, dat3_monthly$month, "01", sep = "-")), "%Y-%m")


dat3_monthly <- dat3_monthly %>%
  select(date, everything()) %>%
  select(-year, -month)

  

(head(dat3_monthly))
(head(dat1))
####################################RENAMING + ADDING THE TOTAL DATA############


new_names <- c("date", "ER_CHF", "ER_CNY", "ER_GB", "ER_JPY",
               "ER_USD", "ECB_DPFR", "ECB_MLFR", "ECB_IR", "RA",
               "BL_house", "M3", "M2", "M1", "Credit_NB", "Credit_HC",
               "Credit_HH", "CBS", "ER_real", "unemp", "order_ind", "order_con",
               "output_prod", "CPI", "CPI_E","DERIV","ER_CAD" )


rename_columns_by_position <- function(dat1, new_names) {
  dat1 %>%
    rename_with(~ new_names, everything())
}



dat1 <- tibble(dat1) %>%
  select(-contains("flag")) %>%
  slice(6:717)
dat1 <- rename_columns_by_position(dat1, new_names)





#####Joining both datasets

df1 <- left_join(dat1, dat3_monthly, by = "date")
skim(df1)


##########changing to numericals and filtering years##########




df1 <- df1 %>%
  mutate_at(vars(ER_CHF, ER_CNY, ER_GB, ER_JPY, ER_USD, ECB_DPFR, ECB_MLFR, ECB_IR, RA,
                 BL_house, M3, M2, M1, Credit_NB, Credit_HC, Credit_HH, CBS,
                 ER_real, unemp, order_ind, order_con, output_prod, CPI, CPI_E,
                 DERIV, ER_CAD), as.numeric) %>%
  mutate(date = ym(date)) %>%
  filter(year(date) >= 1999)



####LOGARITHMIC
vars_to_log <- c("ECB_MLFR", "M1", "Credit_NB")
df1 <- df1 %>%
  mutate_at(vars(vars_to_log), list(~ log(.)))



##################CPI to percentage change (last year)######Remove deriv because of NA's
df1 <- df1 %>%
  mutate(
    month = lubridate::month(date),  
    year = lubridate::year(date)    
  ) %>%
  group_by(month) %>%
  mutate(
    last_year_inflation = lag(CPI, 1), # Get last year's inflation for the same month
    CPI = ((CPI - last_year_inflation) / last_year_inflation) * 100 # Calculate percentage change
  ) %>%
  mutate(
    last_year_inflation_E = lag(CPI_E, 1), # Get last year's inflation for the same month
    CPI_E = ((CPI_E - last_year_inflation_E) / last_year_inflation_E) * 100 # Calculate percentage change
  ) %>%
  ungroup() %>%
  filter(year(date) >= 2000) %>%
  select(-month, -year, -DERIV, -last_year_inflation, -last_year_inflation_E)

############################
##########Testing for Stationarity#######



variables <- names(df1)[-1]
for (col in variables) {
  # Exclude NA values from the variable
  clean_data <- na.omit(df1[[col]])
  
  if (length(clean_data) > 0) {
    adf_result <- adf.test((clean_data), k=0)
    cat("ADF Test for variable", col, "\n")
    print(adf_result)
    cat("\n")
  } else {
    cat("Variable", col, "contains only missing values.\n\n")
  }
}



###############testing





df1_test <- df1 %>%
  mutate_at(vars(starts_with("ER")), list(~ . - lag(.))) %>%
  mutate_at(vars(starts_with("ECB")), list(~ . - lag(.))) %>%
  mutate_at(vars(starts_with("RA")), list(~ . - lag(.))) %>%
  mutate_at(vars(starts_with("BL")), list(~ . - lag(.))) %>%
  mutate_at(vars(starts_with("M")), list(~ . - lag(.))) %>%
  mutate_at(vars(starts_with("order")), list(~ . - lag(.))) %>%
  mutate_at(vars(starts_with("Credit")), list(~ . - lag(.))) %>%
  mutate_at(vars(starts_with("Credit_HH")), list(~ . - lag(.))) %>%
  mutate_at(vars(starts_with("unem")), list(~ . - lag(.))) %>%
  mutate_at(vars(starts_with("CPI")), list(~ . - lag(.))) %>%
  mutate_at(vars(starts_with("output")), list(~ . - lag(.))) %>%
  mutate_at(vars(starts_with("bond")), list(~ . - lag(.))) %>%
  mutate_at(vars(starts_with("CBS")), list(~ . - lag(.)))
  



###########################################





#####################################################TEST

perform_adf_test <- function(column) {
  column <- na.omit(column)
  adf_result <- adf.test(column)
  return(list(
    statistic = adf_result$statistic,
    p_value = adf_result$p.value,
    critical_values = adf_result$parameter
  ))
}

# Apply the function to each column except the 'time' column
results <- df1_test %>%
  select(-date) %>%
  map(perform_adf_test)

# Convert results to a tibble for easier viewing
results_df <- tibble(
  column <- names(results),
  statistic = map_dbl(results, "statistic"),
  p_value = map_dbl(results, "p_value"),
  critical_values = map(results, "critical_values")
)

# Print the results
print(n=28, (results_df))

###########################################
#Remove NA'S
sapply(df1_test, function(x) sum(is.na(x)))

df1_test <- df1_test %>%
  select(-bond5y_percent, -bond2y_percent)

df1_test <- na.omit(df1_test)



df1_test <-   df1_test %>%
  mutate(date = as.Date(date))



##################################################
#splitting data into test and train
set.seed(123)
split_index <- floor(0.7 * nrow(df1_test))

# Split the data into training and testing sets
train_data <- df1_test %>% slice(1:split_index) %>%
  select(-date)
test_data <- df1_test %>% slice((split_index + 1):nrow(df1_test)) %>%
  select(-date)


#WITHOUT CPI_E
#train_data <- df1_test %>% slice(1:split_index) %>%
#  select(-date, -CPI_E)
#test_data <- df1_test %>% slice((split_index + 1):nrow(df1_test)) %>%
 # select(-date, -CPI_E)

###dont forget to remove from table 


#################################################################
ggplot(df1, aes(x = date, y = CPI)) +
  geom_line(color = "blue") +
  labs(title = "CPI Over Time", x = "Date", y = "CPI") +
  theme_minimal()



ggplot(df1_test, aes(x = date, y = CPI)) +
  geom_line(color = "blue") +
  labs(title = "CPI Over Time Differenced", x = "Date", y = "CPI") +
  theme_minimal()






##############################CPI###################################################################################


cor_matrix <- cor(df1_test[-1], use = "complete.obs")

# Step 4: Reshape the correlation matrix for plotting
tidy_cor_matrix <- cor_matrix %>%
  as.data.frame() %>%
  rownames_to_column(var = "Variable1") %>%
  pivot_longer(cols = -Variable1, names_to = "Variable2", values_to = "Correlation")

# Create the heatmap
heatmap <- ggplot(tidy_cor_matrix, aes(x = Variable1, y = Variable2, fill = Correlation)) +
  geom_tile() +
  scale_fill_gradient2(low = "green", high = "red", mid = "white", midpoint = 0, limit = c(-1, 1), space = "Lab", name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  coord_fixed()

print(heatmap)








#################################################################################

ggplot(df1, aes(x = date, y = CPI)) +
  geom_line(color = 'blue') +
  labs(title = 'CPI % change from last year',
       x = 'Date',
       y = 'CPI') +
  theme_minimal()


ggplot(df1_test, aes(x = date, y = CPI)) +
  geom_line(color = 'blue') +
  labs(title = 'Stationary CPI',
       x = 'Date',
       y = 'CPI') +
  theme_minimal()


