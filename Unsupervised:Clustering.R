# Clustering and Unsupervised Part 2: ITP AERO by Miguel Diaz and Dante Schrantz
# Using the class codes to try to solve the ITP Aero company problem. The database was matched by the company's representatives together with the other companies as one of the 7 challenges during the championship days. We asked them to share their challenge with us with the purpose of using the problem for our Machine Learning final project.

# Load required libraries
library(tidyverse)
library(cluster)    # For PAM clustering
library(fpc)        # For cluster validation
library(factoextra) # For visualization
library(e1071)      # For fuzzy clustering
library(NbClust)    # For determining optimal number of clusters
library(gridExtra)  # For arranging plots
library(caret)
library(dplyr)


# Add at the start of the script, after loading libraries
start_time <- Sys.time()
cat("Starting clustering analysis at:", format(start_time), "\n\n")

# Load and preprocess the data
data <- read.csv("/Users/danteschrantz/desktop/UNAV/2024-2025/Machine Learning/Trabajo Final/data/ITPaero.csv", header = TRUE)

preprocess_for_clustering <- function(data) {
  processed_data <- data %>%
    mutate(
      # Convert columns to appropriate types, handling NAs and invalid values
      Brocha = as.character(Brocha),
      BrochaSN = as.character(BrochaSN),
      OrdenFabricacion = factor(OrdenFabricacion),
      PartNumber = factor(PartNumber),
      Maquina = as.character(Maquina),
      TpoIndexador = as.character(TpoIndexador),
      Utillaje = as.character(Utillaje),
      
      # Handle numeric columns with potential issues
      NBrochasHSS = suppressWarnings(as.integer(NBrochasHSS)),
      NDiscos = suppressWarnings(as.integer(NDiscos)),
      NUsos = suppressWarnings(as.integer(NUsos)),
      USDutchman = suppressWarnings(as.integer(USDutchman)),
      NUso = suppressWarnings(as.integer(NUso)),
      NDisco = suppressWarnings(as.integer(NDisco)),
      
      # Handle logical columns
      DUMMY = as.logical(DUMMY),
      Dutchman = as.logical(Dutchman),
      
      # Handle datetime columns
      FBrochado = suppressWarnings(as.POSIXct(FBrochado, format = "%Y-%m-%d %H:%M:%S", tz = "CET")),
      
      # Convert numeric columns
      XC = suppressWarnings(as.numeric(XC)),
      ZC = suppressWarnings(as.numeric(ZC)),
      BC = suppressWarnings(as.numeric(BC)),
      CC = suppressWarnings(as.numeric(CC)),
      XCMM = suppressWarnings(as.numeric(XCMM)),
      ZCMM = suppressWarnings(as.numeric(ZCMM)),
      BCMM = suppressWarnings(as.numeric(BCMM)),
      CCMM = suppressWarnings(as.numeric(CCMM))
    )
  
  # Replace NAs with median values for numeric columns
  preProcValues <- preProcess(processed_data, method = "medianImpute")
  processed_data <- predict(preProcValues, processed_data)
  
  # Select relevant columns for clustering
  clustering_vars <- c("NUso", "NDisco", "XC", "ZC", "BC", "CC", "XCMM", "ZCMM", "BCMM", "CCMM")
  
  # Ensure selected columns are numeric
  clustering_data <- processed_data %>%
    select(all_of(clustering_vars)) %>%
    mutate(across(everything(), as.numeric))
  
  # Remove rows with remaining NA values
  clustering_data <- na.omit(clustering_data)
  
  # Scale the data
  scaled_data <- scale(clustering_data)
  
  return(list(
    original = clustering_data,
    scaled = scaled_data
  ))
}


# Prepare data for clustering
prepared_data <- preprocess_for_clustering(data)
scaled_data <- prepared_data$scaled

# Function to determine optimal number of clusters
find_optimal_clusters <- function(data, max_k = 10) {
    # Elbow method
    wss <- sapply(1:max_k, function(k) {
        kmeans(data, centers = k, nstart = 25)$tot.withinss
    })
    
    # Silhouette method
    sil_width <- sapply(2:max_k, function(k) {
        km <- kmeans(data, centers = k, nstart = 25)
        mean(silhouette(km$cluster, dist(data))[,3])
    })
    
    # Calculate elbow point
    elbow_point <- which(diff(diff(wss)) > mean(diff(diff(wss)))) + 1
    
    # Calculate optimal k from silhouette
    sil_optimal <- which.max(sil_width) + 1
    
    # Calculate gap statistic
    set.seed(123)
    gap_stat <- clusGap(data, FUN = kmeans, nstart = 25, K.max = max_k, B = 50)
    gap_optimal <- maxSE(gap_stat$Tab[, "gap"], gap_stat$Tab[, "SE.sim"], method = "firstmax")
    
    # Plot results
    par(mfrow = c(2,2))
    
    # Elbow plot
    plot(1:max_k, wss, type = "b", pch = 19,
         xlab = "Number of clusters K",
         ylab = "Total within-clusters sum of squares",
         main = "Elbow Method")
    points(elbow_point, wss[elbow_point], col = "red", cex = 2, pch = 19)
    
    # Silhouette plot
    plot(2:max_k, sil_width, type = "b", pch = 19,
         xlab = "Number of clusters K",
         ylab = "Average silhouette width",
         main = "Silhouette Method")
    points(sil_optimal, sil_width[sil_optimal-1], col = "red", cex = 2, pch = 19)
    
    # Gap statistic plot
    plot(gap_stat, main = "Gap Statistic Method")
    
    par(mfrow = c(1,1))
    
    # Print recommendations
    cat("\nCluster number recommendations:")
    cat("\nElbow method suggests:", elbow_point)
    cat("\nSilhouette method suggests:", sil_optimal)
    cat("\nGap statistic suggests:", gap_optimal)
    
    # Use consensus between methods
    suggested_k <- round(median(c(elbow_point, sil_optimal, gap_optimal)))
    cat("\n\nRecommended number of clusters (consensus):", suggested_k, "\n")
    
    return(suggested_k)
}

# Find optimal number of clusters
optimal_k <- min(find_optimal_clusters(scaled_data), 10)  # Limit to maximum of 10 clusters

# Perform K-means clustering
kmeans_result <- kmeans(scaled_data, centers = optimal_k, nstart = 25)

# Perform PAM clustering
pam_result <- pam(scaled_data, k = optimal_k)

# Perform Fuzzy C-means clustering
fuzzy_result <- cmeans(scaled_data, centers = optimal_k, iter.max = 100, verbose = FALSE)

# Function to evaluate clustering results
evaluate_clustering <- function(data, kmeans_res, pam_res, fuzzy_res) {
    # Calculate silhouette scores
    kmeans_sil <- mean(silhouette(kmeans_res$cluster, dist(data))[,3])
    pam_sil <- mean(silhouette(pam_res$clustering, dist(data))[,3])
    fuzzy_sil <- mean(silhouette(max.col(fuzzy_res$membership), dist(data))[,3])
    
    # Calculate cluster separation using alternative to Dunn index
    # Using average silhouette width as a measure of cluster separation
    kmeans_sep <- kmeans_sil
    pam_sep <- pam_sil
    fuzzy_sep <- fuzzy_sil
    
    # Print results
    cat("\nClustering Evaluation Results:")
    cat("\n============================")
    cat("\nK-means:")
    cat("\n  Silhouette Score:", kmeans_sil)
    cat("\n  Cluster Separation:", kmeans_sep)
    cat("\n  Within-cluster SS:", kmeans_res$tot.withinss)
    
    cat("\n\nPAM:")
    cat("\n  Silhouette Score:", pam_sil)
    cat("\n  Cluster Separation:", pam_sep)
    
    cat("\n\nFuzzy C-means:")
    cat("\n  Silhouette Score:", fuzzy_sil)
    cat("\n  Cluster Separation:", fuzzy_sep)
    cat("\n  Objective Function:", fuzzy_res$withinerror)
    
    # Return evaluation metrics
    return(list(
        kmeans = list(silhouette = kmeans_sil, separation = kmeans_sep),
        pam = list(silhouette = pam_sil, separation = pam_sep),
        fuzzy = list(silhouette = fuzzy_sil, separation = fuzzy_sep)
    ))
}

# Evaluate all clustering methods
evaluation_results <- evaluate_clustering(scaled_data, kmeans_result, pam_result, fuzzy_result)

# Visualize clusters
plot_clusters <- function(data, kmeans_res, pam_res, fuzzy_res) {
    # Create PCA for visualization
    pca_result <- prcomp(data)
    pc_data <- as.data.frame(pca_result$x[,1:2])
    
    # K-means plot
    kmeans_plot <- fviz_cluster(kmeans_res, data = pc_data,
                               geom = "point",
                               main = "K-means Clustering")
    
    # PAM plot
    pam_plot <- fviz_cluster(list(data = pc_data, cluster = pam_res$clustering),
                            geom = "point",
                            main = "PAM Clustering")
    
    # Fuzzy plot (using hardened clusters)
    fuzzy_clusters <- max.col(fuzzy_res$membership)
    fuzzy_plot <- fviz_cluster(list(data = pc_data, cluster = fuzzy_clusters),
                              geom = "point",
                              main = "Fuzzy C-means Clustering")
    
    # Arrange plots
    gridExtra::grid.arrange(kmeans_plot, pam_plot, fuzzy_plot, ncol = 2)
}

# Plot clustering results
plot_clusters(scaled_data, kmeans_result, pam_result, fuzzy_result)

# Analyze cluster characteristics
analyze_clusters <- function(original_data, kmeans_res, pam_res, fuzzy_res) {
    # Add cluster assignments to original data
    clustered_data <- cbind(
        original_data,
        kmeans_cluster = kmeans_res$cluster,
        pam_cluster = pam_res$clustering,
        fuzzy_cluster = max.col(fuzzy_res$membership)
    )
    
    # Calculate cluster summaries
    cat("\nCluster Characteristics:")
    cat("\n=======================")
    
    # K-means summaries
    cat("\n\nK-means Cluster Summaries:")
    print(clustered_data %>%
              group_by(kmeans_cluster) %>%
              summarise(across(everything(), list(mean = mean, sd = sd))))
    
    # PAM summaries
    cat("\n\nPAM Cluster Summaries:")
    print(clustered_data %>%
              group_by(pam_cluster) %>%
              summarise(across(everything(), list(mean = mean, sd = sd))))
    
    # Fuzzy summaries
    cat("\n\nFuzzy C-means Cluster Summaries:")
    print(clustered_data %>%
              group_by(fuzzy_cluster) %>%
              summarise(across(everything(), list(mean = mean, sd = sd))))
    
    return(clustered_data)
}

# Analyze cluster characteristics
cluster_analysis <- analyze_clusters(prepared_data$original, 
                                   kmeans_result, 
                                   pam_result, 
                                   fuzzy_result)

# Calculate time taken
end_time <- Sys.time()
time_taken <- end_time - start_time

cat("\nClustering Analysis Complete")
cat("\n========================")
cat("\nStart time:", format(start_time))
cat("\nEnd time:", format(end_time))
cat("\nTotal time taken:", round(time_taken, 2), units(time_taken), "\n")

# Save results with timing information
clustering_results <- list(
    kmeans = kmeans_result,
    pam = pam_result,
    fuzzy = fuzzy_result,
    evaluation = evaluation_results,
    cluster_analysis = cluster_analysis,
    optimal_k = optimal_k,
    timing = list(
        start_time = start_time,
        end_time = end_time,
        duration = time_taken
    )
)

saveRDS(clustering_results, "clustering_results.rds") 