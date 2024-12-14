# ITP AERO by Miguel Diaz and Dante Schrantz
rm(list=ls()) 


carga <- function(x){
  for( i in x ){
    if( ! require( i , character.only = TRUE ) ){
      install.packages( i , dependencies = TRUE )
      require( i , character.only = TRUE )
    }
  }
}
carga("caret")
library(caret)
library(e1071)


# DATASET EXPLANATION: Regression Problem
ITP <- read.csv("/Users/danteschrantz/desktop/UNAV/2024-2025/Machine Learning/Trabajo Final/data/ITPaero.csv", header = TRUE)
View(ITP)
str(ITP)


#----------------------------------------#
# PREPROCESSING:
#----------------------------------------#
# we have to correctly change columns formats
library(dplyr)
ITP <- ITP %>%
  mutate(
    # Variables categóricas convertidas a factores
    Brocha = as.character(Brocha),  # Esto puede seguir siendo un string si no es categórica
    BrochaSN = as.character(BrochaSN),
    OrdenFabricacion = factor(OrdenFabricacion),  # Convertir a factor
    PartNumber = factor(PartNumber),  # Convertir a factor
    Maquina = as.character(Maquina),
    TpoIndexador = as.character(TpoIndexador),
    Utillaje = as.character(Utillaje),
    
    # Variables enteras
    NBrochasHSS = as.integer(NBrochasHSS),
    NDiscos = as.integer(NDiscos),
    NUsos = as.integer(NUsos),
    USDutchman = as.integer(USDutchman),
    NUso = as.integer(NUso),
    NDisco = as.integer(NDisco),
    
    # Variables lógicas
    DUMMY = as.logical(DUMMY),
    Dutchman = as.logical(Dutchman),
    
    # Variables datetime
    FBrochado = as.POSIXct(FBrochado, format = "%Y-%m-%d %H:%M:%S", tz = "CET"),
    
    # Variables numéricas
    XC = as.numeric(XC),
    ZC = as.numeric(ZC),
    BC = as.numeric(BC),
    CC = as.numeric(CC),
    XCMM = as.numeric(XCMM),
    ZCMM = as.numeric(ZCMM),
    BCMM = as.numeric(BCMM),
    CCMM = as.numeric(CCMM)
  )


str(ITP)

# Imputar valores faltantes con la mediana
preProcValues <- preProcess(ITP, method = "medianImpute")

# Aplicar la imputación al conjunto de datos
ITP <- predict(preProcValues, ITP)

# Confirmar que no hay valores NA
colSums(is.na(ITP)) 



#----------------------------------------#
# SPLITING the data in train and test set:
#----------------------------------------#
set.seed(333)

# (75% train, 25% test)
spl <- createDataPartition(ITP$XC, p = 0.75, list = FALSE)  # For deciding which target variable are we going to use for the partition we will review their distributions
#---------------------#
library(ggplot2)
library(gridExtra)


plot_xc <- ggplot(ITP, aes(x = XCMM)) + 
  geom_histogram(bins = 30, fill = "blue", color = "black") + 
  ggtitle("Distribución de XCMM")

plot_zc <- ggplot(ITP, aes(x = ZCMM)) + 
  geom_histogram(bins = 30, fill = "green", color = "black") + 
  ggtitle("Distribución de ZCMM")

plot_bc <- ggplot(ITP, aes(x = BCMM)) + 
  geom_histogram(bins = 30, fill = "red", color = "black") + 
  ggtitle("Distribución de BCMM")

plot_cc <- ggplot(ITP, aes(x = CCMM)) + 
  geom_histogram(bins = 30, fill = "purple", color = "black") + 
  ggtitle("Distribución de CCMM")

grid.arrange(plot_xc, plot_zc, plot_bc, plot_cc, nrow = 2, ncol = 2)
#---------------------#

train_set <- ITP[spl, ]
test_set <- ITP[-spl, ]

# ...
#----------------------------------------#
# FEATURE ENGINEERING
#----------------------------------------#
# Correlation Matrix
carga("corrplot")
carga("Hmisc")
library(corrplot)
library(Hmisc)
View(ITP)

numeric_features <- sapply(ITP, is.numeric)
ITP_numeric <- ITP[, numeric_features]

# Corr ,atrix
cor <- rcorr(as.matrix(ITP_numeric)) 

M <- cor$r
p_mat <- cor$P

# Ploting
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(M, method = "color", col = col(200),  
         type = "upper", order = "hclust", 
         addCoef.col = "black", # Añadir coeficiente de correlación
         tl.col = "darkblue", tl.srt = 45, # Color y rotación de las etiquetas
         p.mat = p_mat, sig.level = 0.05,  # Combinar con nivel de significancia
         diag = FALSE # Ocultar la diagonal principal
)

# High correlation values
w <- which(abs(M) > 0.8 & row(M) < t(row(M)), arr.ind = TRUE)
high_cor_var <- matrix(colnames(M)[w], ncol = 2)
print("Variables with high correlation (greater than 0.8):")
print(high_cor_var)


# 4 LASSO Regression for 4 target axis (XC, ZC, BC, CC)
library(glmnet)

target_variables <- c('XCMM', 'ZCMM', 'BCMM', 'CCMM')

# Loop through each target variable to perform LASSO regression
for (target in target_variables) {
  cat("\nPerforming LASSO Regression for target variable:", target, "\n")
  
  # Define target variable (y) and predictors (x)
  y <- as.matrix(ITP[[target]])  # Select current target variable
  x <- as.matrix(scale(ITP[, c('NBrochasHSS', 'NDiscos', 'NUsos', 'USDutchman', 'XC', 'ZC', 'BC', 'CC')]))  # Select and scale predictors
  
  # Set seed for reproducibility
  set.seed(123)
  
  # Fit LASSO model with cross-validation
  cv_model <- cv.glmnet(x, y, alpha = 1)
  
  # Find optimal lambda value that minimizes MSE
  best_lambda <- cv_model$lambda.min
  cat("The best value for lambda for", target, "is:", best_lambda, "\n")
  cat("MSE associated with the best lambda:", cv_model$cvm[which(cv_model$lambda == best_lambda)], "\n")
  
  # Plot MSE vs. Lambda values
  plot(cv_model, main = paste("LASSO Cross-validation for", target))
  
  # Train the final model using the best lambda
  best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
  
  # Get the coefficients of the best model
  cat("Coefficients of the best model for", target, ":\n")
  print(coef(best_model))
}


# RANDOM FOREST VARIABLE IMPORTANCE (again) for each target variable 
library(randomForest)
set.seed(76)

ITP <- ITP %>%
  mutate(
    OrdenFabricacion = as.numeric(OrdenFabricacion),  # Convertir a factor
    PartNumber = as.numeric(PartNumber),  # Convertir a factor
  )

# List of target variables
target_variables <- c("XCMM", "ZCMM", "BCMM", "CCMM")

# Loop through each target variable and fit Random Forest model
for (target in target_variables) {
  
  # Create formula for the target variable
  formula <- as.formula(paste(target, "~ ."))
  
  # Fit Random Forest model
  rf_model <- randomForest(formula, data = ITP, importance = TRUE, ntree = 100)
  
  # Print model summary
  print(paste("Random Forest Model for:", target))
  print(rf_model)
  
  # Evaluate variable importance
  print(paste("Variable Importance for:", target))
  importance_values <- rf_model$importance[!rownames(rf_model$importance) %in% c("XCMM", "BCMM", "CCMM","ZCMM"), ]
  print(importance_values)
  
  
  # Plot variable importance
  varImpPlot(rf_model, main = paste("Variable Importance for:", target))
}


# We have chosen the variables that showed high importance across multiple axes to ensure they contribute to the model's performance.
# ...
# So here are our variables based on the previous RF analysis
features <- features <- c('Utillaje', 'FBrochado', 'XC', 'CC','BC', 'Maquina', 'Brocha', 'TpoIndexador', 'OrdenFabricacion', 'ZC', 'NUso', 'PartNumber')
labels <- c('XCMM', 'ZCMM', 'BCMM', 'CCMM')

train_set$PartNumber <- factor(train_set$PartNumber)
head(train_set$PartNumber)
train_set$OrdenFabricacion <- factor(train_set$OrdenFabricacion)
head(train_set$OrdenFabricacion)

# Conjunto de entrenamiento
X_train <- train_set[, features]
y_train <- train_set[, labels]
X_test <- train_set[, features]
y_test <- train_set[, labels]

# Ajustar niveles de PartNumber y OrdenFabricacion en el conjunto de prueba
X_test$PartNumber <- factor(X_test$PartNumber, levels = levels(X_train$PartNumber))
X_test$OrdenFabricacion <- factor(X_test$OrdenFabricacion, levels = levels(X_train$OrdenFabricacion))

# Verificar consistencia de niveles (opcional)
cat("Niveles no coincidentes en PartNumber:", 
    setdiff(levels(X_test$PartNumber), levels(X_train$PartNumber)), "\n")
cat("Niveles no coincidentes en OrdenFabricacion:", 
    setdiff(levels(X_test$OrdenFabricacion), levels(X_train$OrdenFabricacion)), "\n")

# Verificar las dimensiones de los conjuntos de datos
cat("Dimensiones de X_train:", dim(X_train), "\n")
cat("Dimensiones de y_train:", dim(y_train), "\n")
cat("Dimensiones de X_test:", dim(X_test), "\n")
cat("Dimensiones de y_test:", dim(y_test), "\n")

# Verificar si hay valores NA en los conjuntos de datos
cat("Número de NA en X_train:", sum(is.na(X_train)), "\n")
cat("Número de NA en y_train:", sum(is.na(y_train)), "\n")
cat("Número de NA en X_test:", sum(is.na(X_test)), "\n")
cat("Número de NA en y_test:", sum(is.na(y_test)), "\n")
#----------------------------------------#
# TUNING+RESAMPLING
#----------------------------------------#


#----------------------------------------#
# LINEAR REGRESSION
#----------------------------------------#

library(doParallel)

# Configuración de paralelización
num_cores <- detectCores() - 1  # Usa todos los núcleos menos 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Crear fórmulas para cada variable objetivo
formulas <- lapply(labels, function(label) {
  as.formula(paste(label, "~", paste(features, collapse = "+")))
})

# Configuración del resampling
control <- trainControl(method = "repeatedcv", number = 5, repeats = 3, verboseIter = TRUE, allowParallel = TRUE)

# Lista para almacenar resultados
results_linear <- list()

# Entrenar modelos en paralelo
set.seed(123)
for (label in labels) {
  # Crear un dataframe para cada variable objetivo
  train_data <- cbind(X_train, target = y_train[[label]])
  
  # Crear fórmula
  formula <- as.formula(paste("target ~", paste(features, collapse = "+")))
  
  # Entrenar el modelo
  model <- train(
    formula,
    data = train_data,
    method = "lm",
    trControl = control,
    tuneLength = 10
  )
  
  # Guardar el modelo entrenado
  results_linear[[label]] <- model
}

# Evaluar el rendimiento de los modelos
cat("\n--- Resultados del Modelo de Regresión Lineal ---\n")
for (label in labels) {
  # Crear conjunto de prueba para cada variable objetivo
  test_data <- cbind(X_test, target = y_test[[label]])
  
  # Predicción
  predictions <- predict(results_linear[[label]], newdata = test_data)
  
  # Calcular métricas
  resample_results <- postResample(predictions, test_data$target)
  
  max_error <- max(abs(predictions - test_data$target), na.rm = TRUE)
  
  # Imprimir resultados
  cat("\nResultados de predicción para la variable objetivo:", label, "\n")
  cat("RMSE:", resample_results[1], "\n")
  cat("R-squared:", resample_results[2], "\n")
  cat("MAE:", resample_results[3], "\n")
  cat("Error Máximo:", max_error, "\n")
}

# Detener el clúster
stopCluster(cl)
registerDoSEQ()  # Volver al modo secuencial


#----------------------------------------#
# KNN
#----------------------------------------#

# Configuración del resampling
control <- trainControl(method = "repeatedcv", number = 5, repeats = 3, verboseIter = TRUE, allowParallel = TRUE)

# Crear una lista para almacenar los resultados de los modelos KNN
results_knn <- list()

# Entrenar un modelo KNN para cada una de las variables objetivo
set.seed(123)
cat("\n--- Entrenando Modelos KNN ---\n")
for (label in labels) {
  # Crear un nuevo dataframe específico para cada variable objetivo con predictores + variable objetivo
  train_data <- cbind(X_train, target = y_train[[label]])
  
  # Crear la fórmula usando los predictores y la variable objetivo actual
  formula <- as.formula(paste("target ~", paste(features, collapse = "+")))
  
  # Entrenar el modelo KNN usando cross-validation
  model <- train(
    formula,
    data = train_data,
    method = "knn",
    trControl = control,
    tuneLength = 5
  )
  
  # Guardar el modelo en la lista de resultados
  results_knn[[label]] <- model
  
  # Imprimir progreso
  cat("\nModelo KNN entrenado para la variable objetivo:", label, "\n")
}

# Evaluar el rendimiento de los modelos KNN
cat("\n--- Resultados del Modelo KNN ---\n")
for (label in labels) {
  # Combinar las características del conjunto de prueba
  test_data <- cbind(X_test, target = y_test[[label]])
  
  # Predecir los valores para el conjunto de prueba
  predictions <- predict(results_knn[[label]], newdata = test_data)
  
  # Calcular métricas de evaluación utilizando el conjunto de prueba
  resample_results <- postResample(predictions, test_data$target)
  
  # Calcular el error máximo
  max_error <- max(abs(predictions - test_data$target), na.rm = TRUE)
  
  # Imprimir los resultados
  cat("\nResultados de predicción para la variable objetivo:", label, "\n")
  cat("RMSE:", resample_results[1], "\n")
  cat("R-squared:", resample_results[2], "\n")
  cat("MAE:", resample_results[3], "\n")
  cat("Error Máximo:", max_error, "\n")
}

# Detener el clúster
#stopCluster(cl)
#registerDoSEQ()  # Volver al modo secuencial

#----------------------------------------#
#             RANDOM FOREST
#----------------------------------------#

# Ajustar el grid de tuneo para Random Forest
mtry <- sqrt(ncol(X_train))  # Proponemos usar la raíz cuadrada del número de variables de entrada

tunegrid <- expand.grid(.mtry = c(1:15))  # Expandimos el grid para probar diferentes valores de mtry

# Lista para almacenar los resultados de los modelos de Random Forest
results_rf <- list()

# Entrenar un modelo Random Forest para cada variable objetivo
for (label in labels) {
  # Crear un nuevo dataframe específico para cada variable objetivo con predictores + variable objetivo
  train_data <- cbind(X_train, target = y_train[[label]])
  
  # Crear la fórmula usando los predictores y la variable objetivo actual
  formula <- as.formula(paste("target ~", paste(features, collapse = "+")))
  
  # Entrenar el modelo Random Forest usando cross-validation y tuneando el valor de mtry
  set.seed(76)
  model <- train(formula, data = train_data, method = "rf", trControl = control, tuneGrid = tunegrid, ntree = 1000)
  
  # Guardar el modelo en la lista de resultados
  results_rf[[label]] <- model
}

# Evaluar el rendimiento de los modelos Random Forest
cat("\n--- Resultados del Modelo Random Forest ---\n")
for (label in labels) {
  # Combinar las características del conjunto de prueba
  test_data <- cbind(X_test, target = y_test[[label]])
  
  # Predecir los valores para el conjunto de prueba
  predictions <- predict(results_rf[[label]], newdata = test_data)
  
  # Calcular métricas de evaluación utilizando el conjunto de prueba
  resample_results <- postResample(predictions, test_data$target)
  
  # Imprimir los resultados
  cat("\nResultados de predicción para la variable objetivo:", label, "\n")
  cat("RMSE:", resample_results[1], "\n")
  cat("R-squared:", resample_results[2], "\n")
  cat("MAE:", resample_results[3], "\n")
}




# https://rpubs.com/joaquin_ar/383283, completar Script con esto
