##HEALTH INSURANCE COST PREDICTION

##PRPJECT OVERVIEW

This project develops and deploys a machine learning model to predict individual medical insurance charges based on demographic and health realted factors. The objective is to support premium pricing, financial risk assessment and healthcare cost forecasting data-driven modelling.
The solution includes an end-to-end machine learning pipeline and a
deployed Streamlit web application for real-time cost estimation.

------------------------------------------------------------------------

##  Business Problem

Accurately estimating medical insurance premiums is critical for:

-   Risk management
-   Financial forecasting
-   Underwriting optimization
-   Cost transparency

Traditional pricing methods may fail to capture complex nonlinear
relationships between health indicators and insurance costs. This
project leverages machine learning to improve prediction accuracy and
decision support.

------------------------------------------------------------------------

##  Dataset Description

Features include:

-   Age
-   BMI
-   Smoking status
-   Region
-   Number of dependents
-   Gender

Target Variable: - Medical insurance charges

------------------------------------------------------------------------

##  Methodology

### Data Preprocessing

-   Exploratory Data Analysis (EDA)
-   Categorical variable encoding
-   Feature engineering
-   Train-test split

###  Model Selection

Model Used:

**Gradient Boosting Regressor**

Chosen for: - Strong performance on structured tabular data - Ability to
model nonlinear relationships - Balanced bias-variance tradeoff

###  Model Evaluation

Performance Metrics:

-   **R² Score:** 0.86
-   **RMSE:** 4,404
-   **MAE:** 1,965

### Model Optimization

-   Cross-validation
-   Hyperparameter tuning
-   Error analysis

------------------------------------------------------------------------

## Deployment

The model was deployed using **Streamlit**, enabling:

-   Real-time premium prediction
-   Interactive user input
-   Scenario simulation for pricing strategy

------------------------------------------------------------------------

## Tech Stack

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Gradient Boosting Regressor
-   Matplotlib / Seaborn
-   Streamlit

------------------------------------------------------------------------

##  Key Insights

-   Smoking status is the strongest predictor of insurance cost.
-   BMI significantly impacts premium variability.
-   Age shows nonlinear influence on medical charges.

------------------------------------------------------------------------

## Relevance to Finance & Healthcare

### Finance Applications

-   Premium pricing strategy
-   Risk modeling
-   Revenue forecasting

### Healthcare Applications

-   Medical cost prediction
-   Health risk assessment
-   Insurance analytics

------------------------------------------------------------------------

##  Future Improvements

-   Compare with XGBoost / LightGBM
-   Add SHAP explainability
-   Deploy via cloud infrastructure
-   Build REST API for scalability

------------------------------------------------------------------------

##  Author

Lesson Shepherd Karidza
Data Scientist | Financial & Healthcare Analytics
