# Supervised Part 2: ITP AERO by Miguel Diaz and Dante Schrantz 
# Using the class codes to try to solve the ITP Aero company problem. The database was matched by the company's representatives together with the other companies as one of the 7 challenges during the championship days. We asked them to share their challenge with us with the purpose of using the problem for our Machine Learning final project.

# Cargar bibliotecas
library(tidyverse)
library(readxl)
library(caret)
library(cluster)
library(factoextra)
library(randomForest)
library(glmnet)
library(mlbench)
library(tidymodels)
library(gbm)  
library(corrplot)
library(xgboost) 
library(keras)  
library(tensorflow)
library(zoo)

# Add at the start of the script, after loading libraries
start_time <- Sys.time()
cat("Starting supervised analysis at:", format(start_time), "\n\n")

# Add calculate_metrics function at the start of the script
calculate_metrics <- function(pred, actual) {
    list(
        rmse = sqrt(mean((pred - actual)^2)),
        mae = mean(abs(pred - actual)),
        max_error = max(abs(pred - actual)),
        r_squared = 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
    )
}

# Load the data from the csv file
data <- read.csv("/Users/danteschrantz/desktop/UNAV/2024-2025/Machine Learning/Trabajo Final/data/ITPaero.csv", header = TRUE)

# Verify data loading
if(is.null(data) || nrow(data) == 0) {
    stop("Error: Data not loaded properly")
}

# Enhanced preprocessing function for multiple targets and create artificial variables 
preprocess_data <- function(data, target_var) {
  # First, convert target variable to numeric
  data[[target_var]] <- as.numeric(as.character(data[[target_var]]))
  
  # Print diagnostic information
  cat("\nBefore preprocessing:", target_var)
  cat("\nNumber of rows:", nrow(data))
  cat("\nNumber of NA values in target:", sum(is.na(data[[target_var]])))
  cat("\nRange of target values:", range(data[[target_var]], na.rm = TRUE), "\n")
  
  processed_data <- data %>%
    mutate(
      FBrochado = as.Date(FBrochado, format = "%Y-%m-%d"),
      # Ensure numeric columns are properly converted
      NBrochasHSS = as.numeric(as.character(NBrochasHSS)),
      NDiscos = as.numeric(as.character(NDiscos)),
      NUsos = as.numeric(as.character(NUsos)),
      USDutchman = as.numeric(as.character(USDutchman)),
      NUso = as.numeric(as.character(NUso)),
      
      # Handle DUMMY and Dutchman as factors instead of numeric
      DUMMY = as.factor(DUMMY),
      Dutchman = as.factor(Dutchman),
      
      # Create features after ensuring numeric conversion
      Month = as.numeric(month(FBrochado)),
      DayOfWeek = as.numeric(wday(FBrochado)),
      NBrochas_NUsos = NBrochasHSS * NUsos,
      Efficiency = ifelse(NBrochasHSS == 0, 0, NUsos / NBrochasHSS),
      NBrochasHSS_2 = NBrochasHSS^2,
      NUsos_2 = NUsos^2
    )
  
  # Handle missing values more carefully
  processed_data <- processed_data %>%
    mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))
  
  # Handle outliers less aggressively
  if(is.numeric(processed_data[[target_var]])) {
    Q1 <- quantile(processed_data[[target_var]], 0.01, na.rm = TRUE)
    Q3 <- quantile(processed_data[[target_var]], 0.99, na.rm = TRUE)
    IQR <- Q3 - Q1
    processed_data <- processed_data %>%
      filter(
        !!sym(target_var) >= (Q1 - 3 * IQR),
        !!sym(target_var) <= (Q3 + 3 * IQR)
      )
  }
  
  # Print diagnostic information after preprocessing
  cat("\nAfter preprocessing:", target_var)
  cat("\nNumber of rows:", nrow(processed_data))
  cat("\nNumber of NA values in target:", sum(is.na(processed_data[[target_var]])))
  cat("\nRange of target values:", range(processed_data[[target_var]], na.rm = TRUE), "\n")
  
  # Convert factor variables to dummy variables for modeling
  if(any(sapply(processed_data, is.factor))) {
    dummy_vars <- model.matrix(~ . - 1, data = processed_data[, sapply(processed_data, is.factor)])
    processed_data <- cbind(processed_data[, !sapply(processed_data, is.factor)], as.data.frame(dummy_vars))
  }
  
  return(processed_data)
}

# Check the data types of the variables before preprocessing
check_data_types <- function(data) {
    cat("\nData types of columns:\n")
    print(sapply(data, class))
    
    cat("\nSample of target variables:\n")
    print(head(data[c("XCMM", "ZCMM", "BCMM", "CCMM")]))
    
    cat("\nSummary of target variables:\n")
    print(summary(data[c("XCMM", "ZCMM", "BCMM", "CCMM")]))
}

# Validate the data before preprocessing
validate_data <- function(data, target_variables) {
    cat("\nValidating data before processing:\n")
    
    # Check for target variables
    for(var in target_variables) {
        cat("\nChecking", var, ":")
        cat("\nUnique values:", length(unique(data[[var]])))
        cat("\nSample values:", head(data[[var]]), "\n")
    }
    
    # Check for required numeric columns
    numeric_cols <- c("NBrochasHSS", "NDiscos", "NUsos", "USDutchman", "NUso")
    for(col in numeric_cols) {
        cat("\nChecking", col, ":")
        cat("\nClass:", class(data[[col]]))
        cat("\nUnique values:", length(unique(data[[col]])))
        cat("\nRange:", range(as.numeric(data[[col]]), na.rm = TRUE), "\n")
    }
}

# Call the datatypes function defined aboves
cat("\nChecking data types and contents:\n")
check_data_types(data)

# Define target variables first
target_variables <- c("XCMM", "ZCMM", "BCMM", "CCMM")

# Verify that target variables exist in the data
missing_vars <- target_variables[!target_variables %in% names(data)]
if(length(missing_vars) > 0) {
    stop("Error: Missing target variables in data: ", paste(missing_vars, collapse = ", "))
}

# Call the diagnostic functions created above
cat("Data dimensions:", dim(data), "\n")
cat("Column names:", names(data), "\n")
cat("First few rows of target variables:\n")
print(head(data[target_variables]))

# Validate data
validate_data(data, target_variables)


#--------------------------------------------#
# Function to train models with grid search #
#--------------------------------------------#
train_target_models <- function(data, target_var) {
    # Process data
    processed_data <- preprocess_data(data, target_var)
    
    # Sort data by date for temporal features
    processed_data <- processed_data %>%
      arrange(FBrochado) %>%
      mutate(
        # Polynomial features for the variables
        NBrochasHSS_3 = NBrochasHSS^3,
        NBrochasHSS_4 = NBrochasHSS^4,
        NUsos_3 = NUsos^3,
        NUsos_4 = NUsos^4,
        
        # Create interactions between variables
        NBrochas_Discos = NBrochasHSS * NDiscos,
        NBrochas_Efficiency = NBrochasHSS * Efficiency,
        NUsos_Efficiency = NUsos * Efficiency,
        NBrochas_NUsos_Efficiency = NBrochasHSS * NUsos * Efficiency,
        
        # Exponential and log transformations
        log_NBrochas = log1p(abs(NBrochasHSS)),
        log_NUsos = log1p(abs(NUsos)),
        exp_NBrochas = exp(scale(NBrochasHSS)),
        exp_NUsos = exp(scale(NUsos)),
        
        # Sine and cosine to check for cyclical patterns
        Month_sin = sin(2 * pi * Month/12),
        Month_cos = cos(2 * pi * Month/12),
        DayOfWeek_sin = sin(2 * pi * DayOfWeek/7),
        DayOfWeek_cos = cos(2 * pi * DayOfWeek/7)
      )
    
    # Calculate rolling statistics separately to handle NAs properly
    roll_stats <- data.frame(
      rolling_mean_3 = rollapply(processed_data[[target_var]], width = 3, FUN = mean, fill = NA, align = "right"),
      rolling_mean_5 = rollapply(processed_data[[target_var]], width = 5, FUN = mean, fill = NA, align = "right"),
      rolling_sd_3 = rollapply(processed_data[[target_var]], width = 3, FUN = sd, fill = NA, align = "right"),
      rolling_sd_5 = rollapply(processed_data[[target_var]], width = 5, FUN = sd, fill = NA, align = "right"),
      rolling_max_3 = rollapply(processed_data[[target_var]], width = 3, FUN = max, fill = NA, align = "right"),
      rolling_min_3 = rollapply(processed_data[[target_var]], width = 3, FUN = min, fill = NA, align = "right")
    )
    
    # Fill NA values with appropriate values
    roll_stats <- roll_stats %>%
      mutate(across(everything(), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))
    
    # Add rolling statistics back to processed data
    processed_data <- bind_cols(processed_data, roll_stats)
    
    # Add lag features
    processed_data <- processed_data %>%
      mutate(
        !!paste0(target_var, "_lag1") := lag(!!sym(target_var), 1),
        !!paste0(target_var, "_lag2") := lag(!!sym(target_var), 2),
        !!paste0(target_var, "_lag3") := lag(!!sym(target_var), 3)
      )
    
    # Fill NA values in lag features with mean
    processed_data <- processed_data %>%
      mutate(across(starts_with(paste0(target_var, "_lag")), 
                   ~ifelse(is.na(.), mean(get(target_var), na.rm = TRUE), .)))
    
    # Extended feature set with our newly processed data
    features <- c(
      "NBrochasHSS", "NDiscos", "NUsos", "USDutchman", "NUso",
      "NBrochas_NUsos", "Efficiency", 
      "NBrochasHSS_2", "NBrochasHSS_3", "NBrochasHSS_4",
      "NUsos_2", "NUsos_3", "NUsos_4",
      "NBrochas_Discos", "NBrochas_Efficiency", "NUsos_Efficiency",
      "NBrochas_NUsos_Efficiency",
      "rolling_mean_3", "rolling_mean_5",
      "rolling_sd_3", "rolling_sd_5",
      "rolling_max_3", "rolling_min_3",
      "log_NBrochas", "log_NUsos",
      "exp_NBrochas", "exp_NUsos",
      "Month_sin", "Month_cos",
      "DayOfWeek_sin", "DayOfWeek_cos",
      paste0(target_var, "_lag1"), 
      paste0(target_var, "_lag2"), 
      paste0(target_var, "_lag3")
    )
    
    X <- processed_data %>% select(all_of(features))
    y <- processed_data[[target_var]]
    
    # Split data
    set.seed(42)
    train_index <- createDataPartition(y, p = 0.8, list = FALSE)
    X_train <- X[train_index, ]
    y_train <- y[train_index]
    X_test <- X[-train_index, ]
    y_test <- y[-train_index]
    
    # Scale features only in the x to conserve millimeters for max error
    preproc <- preProcess(X_train, method = c("center", "scale", "YeoJohnson"))
    X_train_scaled <- predict(preproc, X_train)
    X_test_scaled <- predict(preproc, X_test)
    
    # Convert to matrix
    X_train_matrix <- as.matrix(X_train_scaled)
    X_test_matrix <- as.matrix(X_test_scaled)
    
    # 1. Random Forest with hyperparameter tuning
    cat("\nTraining Random Forest...\n")
    rf_model <- randomForest(
      x = X_train_matrix,
      y = y_train,
      ntree = 1000,
      mtry = floor(sqrt(ncol(X_train_matrix))),
      importance = TRUE,
      do.trace = 100  # Show progress every 100 trees
    )
    
    # Grid search for Random Forest (after extensive hyperparameter tuning we were able to narrow it down to 2 values per parameter)
    rf_grid <- expand.grid(
      ntree = c(1500, 2000),
      mtry = c(8, 12),
      min.node.size = c(3, 5),
      sample.fraction = c(0.7, 0.9)
    )
    
    cat("\nTuning Random Forest...\n")
    cat("Total configurations to try:", nrow(rf_grid), "\n")
    
    best_rf <- NULL
    best_rf_score <- Inf
    
    for(i in 1:nrow(rf_grid)) {
        cat(sprintf("\nTrying configuration %d/%d\n", i, nrow(rf_grid)))
        rf_model <- randomForest(
            x = X_train_matrix,
            y = y_train,
            ntree = rf_grid$ntree[i],
            mtry = rf_grid$mtry[i],
            min.node.size = rf_grid$min.node.size[i],
            sample.fraction = rf_grid$sample.fraction[i],
            importance = TRUE
        )
        
        rf_pred <- predict(rf_model, X_test_matrix)
        rf_score <- max(abs(rf_pred - y_test))
        
        cat(sprintf("Current score: %.6f\n", rf_score))
        
        if(rf_score < best_rf_score) {
            best_rf_score <- rf_score
            best_rf <- rf_model
            cat("New best RF score:", rf_score, "\n")
            cat("Parameters: ntree =", rf_grid$ntree[i], 
                "mtry =", rf_grid$mtry[i],
                "min.node.size =", rf_grid$min.node.size[i],
                "sample.fraction =", rf_grid$sample.fraction[i], "\n")
        }
    }
    
    # 2. XGBoost with optimized parameters
    cat("\nTraining XGBoost...\n")
    xgb_params <- list(
      objective = "reg:squarederror",
      eta = 0.01,
      max_depth = 6,
      subsample = 0.8,
      colsample_bytree = 0.8,
      min_child_weight = 1,
      gamma = 0
    )
    
    # Grid search for XGBoost (same thing as randomForest. Was narrowed down after extensive searches)
    xgb_grid <- expand.grid(
        eta = 0.05,
        max_depth = 8,
        min_child_weight = c(3, 5),
        subsample = c(0.7, 0.9),
        colsample_bytree = c(0.7, 0.9),
        gamma = 0,
        lambda = 0,
        alpha = 0,
        nrounds = c(1500, 2000)
    )
    
    cat("\nTuning XGBoost...\n")
    cat("Total configurations to try:", nrow(xgb_grid), "\n")
    
    dtrain <- xgb.DMatrix(data = X_train_matrix, label = y_train)
    dtest <- xgb.DMatrix(data = X_test_matrix, label = y_test)
    
    best_xgb <- NULL
    best_xgb_score <- Inf
    
    #Get each parameter for each grid search and for every target variable
    for(i in 1:nrow(xgb_grid)) {
        cat(sprintf("\nTrying XGBoost configuration %d/%d\n", i, nrow(xgb_grid)))
        params <- list(
            objective = "reg:squarederror",
            eta = xgb_grid$eta[i],
            max_depth = xgb_grid$max_depth[i],
            min_child_weight = xgb_grid$min_child_weight[i],
            subsample = xgb_grid$subsample[i],
            colsample_bytree = xgb_grid$colsample_bytree[i],
            gamma = xgb_grid$gamma[i],
            lambda = xgb_grid$lambda[i],
            alpha = xgb_grid$alpha[i]
        )
        
        # Train the model on the target variable
        xgb_model <- xgb.train(
            params = params,
            data = dtrain,
            nrounds = xgb_grid$nrounds[i],
            early_stopping_rounds = 50,
            watchlist = list(train = dtrain, test = dtest),
            verbose = 0
        )
        
        #Predict and score the model
        xgb_pred <- predict(xgb_model, dtest)
        xgb_score <- max(abs(xgb_pred - y_test))
        
        cat(sprintf("Current XGB score: %.6f\n", xgb_score))
        
        # Keep the highest score in the best_xgb_score variable
        if(xgb_score < best_xgb_score) {
            best_xgb_score <- xgb_score
            best_xgb <- xgb_model
            cat("New best XGB score:", xgb_score, "\n")
            cat("Parameters:", paste(names(params), params, sep="=", collapse=", "), "\n")
        }
    }
    
    # Calculate final metrics
    rf_pred <- predict(best_rf, X_test_matrix)
    xgb_pred <- predict(best_xgb, dtest)
    ensemble_pred <- (rf_pred * 0.5 + xgb_pred * 0.5)
    
    metrics <- list(
        rf = calculate_metrics(rf_pred, y_test),
        xgb = calculate_metrics(xgb_pred, y_test),
        ensemble = calculate_metrics(ensemble_pred, y_test)
    )
    
    return(list(
        models = list(rf = best_rf, xgb = best_xgb),
        metrics = metrics,
        preprocessing = preproc,
        features = features
    ))
}

# Train all models
cat("\nStarting model training for all target variables...\n")

target_variables <- c("XCMM", "ZCMM", "BCMM", "CCMM")
all_models <- list()

for(target in target_variables) {
    cat(sprintf("\n\nTraining models for %s (%d/%d)\n", 
                target, 
                which(target_variables == target), 
                length(target_variables)))
    cat("=======================================\n")
    
    tryCatch({
        all_models[[target]] <- train_target_models(data, target)
        
        # Print summary metrics for this target
        cat(sprintf("\nResults for %s:\n", target))
        print(all_models[[target]]$metrics)
        
    }, error = function(e) {
        cat(sprintf("\nError training models for %s: %s\n", target, e$message))
    })
}

# Print final summary
cat("\n\nFinal Summary Report")
cat("\n==================\n")
for(target in target_variables) {
    if(!is.null(all_models[[target]])) {
        cat(sprintf("\n%s Performance:\n", target))
        cat("RMSE:", all_models[[target]]$metrics$ensemble$rmse, "\n")
        cat("MAE:", all_models[[target]]$metrics$ensemble$mae, "\n")
        cat("R-squared:", all_models[[target]]$metrics$ensemble$r_squared, "\n")
        cat("Max Error:", all_models[[target]]$metrics$ensemble$max_error, "\n")
        
        # Determine best model based on max error
        rf_error <- all_models[[target]]$metrics$rf$max_error
        xgb_error <- all_models[[target]]$metrics$xgb$max_error
        
        # Find the if the XGB or randomForest is better and print the parameters to be able to narrow down parameters
        if(rf_error <= xgb_error) {
            cat("\nBest Model: Random Forest")
            rf_model <- all_models[[target]]$models$rf
            cat("\nParameters:\n")
            cat("ntree:", rf_model$ntree, "\n")
            cat("mtry:", rf_model$mtry, "\n")
            cat("nodesize:", rf_model$nodesize, "\n")
            cat("sample.fraction:", rf_model$sampsize/length(rf_model$y), "\n")
        } else {
            cat("\nBest Model: XGBoost")
            xgb_model <- all_models[[target]]$models$xgb
            params <- xgb_model$params
            cat("\nParameters:\n")
            cat("eta:", params$eta, "\n")
            cat("max_depth:", params$max_depth, "\n")
            cat("min_child_weight:", params$min_child_weight, "\n")
            cat("subsample:", params$subsample, "\n")
            cat("colsample_bytree:", params$colsample_bytree, "\n")
            cat("gamma:", params$gamma, "\n")
            cat("lambda:", params$lambda, "\n")
            cat("alpha:", params$alpha, "\n")
            cat("nrounds:", xgb_model$niter, "\n")
        }
        cat("\n----------------------------------------\n")
    }
}

# Add at the end of the script, before saving results
end_time <- Sys.time()
time_taken <- end_time - start_time

# Give the total time 
cat("\nAnalysis Complete")
cat("\n========================")
cat("\nStart time:", format(start_time))
cat("\nEnd time:", format(end_time))
cat("\nTotal time taken:", round(time_taken, 2), units(time_taken), "\n")

# Save models
saveRDS(all_models, "trained_models2.rds")
