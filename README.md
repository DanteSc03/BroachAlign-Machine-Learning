# ✨ Machine Learning Project for Predicting Broach Maintenance in Airplane Engine Manufacturing 🎨

## Overview
This project was developed as part of a focus on machine learning techniques for clustering and regression analysis. It explores real-world datasets to solve challenges and extract meaningful insights. Specifically, it addresses the critical task of predicting when to replace broaches used in manufacturing airplane engines.

The work was a collaborative effort by:
- **Dante Schrantz** ([GitHub: dantesc03](https://github.com/dantesc03))
- **Miguel Diaz Perez de Juan** ([GitHub: migueldiazpdj](https://github.com/migueldiazpdj))

## Goals
The primary objectives of this project are:
1. **Clustering Analysis**: Identifying natural groupings within the data to understand broach wear patterns.
2. **Regression Modeling**: Developing models to predict when broaches should be replaced to minimize downtime and costs.

## Key Features
- **🔀 Clustering Techniques**: Analysis using various clustering algorithms, including K-Means and hierarchical clustering.
- **⚛️ Regression Models**: Implementation of multiple regression techniques tailored for industrial datasets.
- **📊 Dataset Handling**: Preprocessing, cleaning, and validating the data for robust model performance.

## Files
- `clustering_analysis.R`: Script implementing clustering techniques.
- `ITPaero.R`: Regression modeling script using ITP Aero dataset.
- `TrabajoFinal.R`: Final integration script combining insights from all analyses.
- `ITPaero.csv`: Dataset provided for regression modeling.

## Dataset
The dataset used in this project focuses on broach calibration and wear patterns. Key features of the dataset include:
- **Training Dataset**: Includes parameters such as broach usage time, wear indicators, and operating conditions.
- **Validation Dataset**: Contains the target variables to evaluate model accuracy.
- **Metrics Evaluated**:
  - Absolute Maximum Error: Less than 1 for the corrector on the X-axis and 0.15 for others.
  - RMSE: Less than 0.25 for the corrector on the X-axis and 0.025 for others.
- **Size**: The dataset is structured to represent both operational variability and wear progression, ensuring robust model training.

## Compliance with Requirements
The models developed in this project meet all requirements specified:
1. **Error Metrics**:
   - The absolute maximum error and RMSE values for all correctors remain within the defined thresholds.
   - Models were extensively validated to ensure compliance with these constraints.
2. **Computational Efficiency**:
   - Optimization techniques were applied to ensure models run efficiently, reducing computational time while maintaining accuracy.
3. **Originality and Applicability**:
   - Novel preprocessing techniques were implemented to handle noisy and missing data.
   - The methodology is adaptable to other predictive maintenance scenarios.
4. **Presentation and Clarity**:
   - The process, results, and insights are clearly documented and visualized for ease of interpretation.

## Results
1. **🎨 Clustering**:
   - Successfully identified meaningful clusters representing different broach wear stages.
   - Visualizations and insights extracted to enhance interpretability.

2. **⚛️ Regression**:
   - Achieved RMSE below the threshold of 0.25 on specific axes.
   - Models demonstrated generalizability to unseen validation data.

## Instructions
### 🔧 Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/dantesc03/BroachAlign-Machine-Learning.git
   ```
2. Install required dependencies in R.
   ```R
   install.packages(c("dplyr", "ggplot2", "caret", "tidyr", "readr", "cluster", "factoextra"))
   ```
3. Load the scripts and dataset into your R environment.

### 💡 Execution
- Run `clustering_analysis.R` for clustering insights.
- Use `ITPaero.R` for regression analysis.
- Execute `TrabajoFinal.R` for a comprehensive summary.

## Acknowledgments
- **ITP Aero** for providing the dataset and the challenge.
- Our professor and classmates for guidance and support.

---
### 📢 Contact
For any inquiries or feedback, reach out to us:
- **Dante Schrantz**: [GitHub](https://github.com/dantesc03)
- **Miguel Diaz Perez de Juan**: [GitHub](https://github.com/migueldiazpdj)

