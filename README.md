# 🌍 Air Quality Analytics & AQI Prediction

A machine learning project that analyzes air quality data and predicts the Air Quality Index (AQI) using pollutant measurements.

## 📌 Project Overview

This project performs Exploratory Data Analysis (EDA) on air quality data and builds machine learning models to predict AQI.

A Random Forest Regression model was selected as the final model because it performed better than Linear Regression on the test dataset.

The trained model is integrated with a Streamlit web application where users can enter pollutant measurements and get an AQI prediction along with the corresponding AQI category and health recommendation.

## 🚀 Features

- Exploratory Data Analysis of air quality data
- City-wise AQI analysis
- Monthly and yearly AQI trends
- Seasonal AQI analysis
- Pollutant correlation analysis
- AQI distribution analysis
- AQI category analysis
- Linear Regression model
- Random Forest Regression model
- Model performance comparison
- Feature importance analysis
- Residual analysis
- AQI prediction using new pollutant values
- Interactive Streamlit dashboard
- AQI category and health recommendations
- Deployed as a live Streamlit application

## 🧪 Pollutants Used

The machine learning model uses the following pollutant features:

- PM2.5
- PM10
- NO
- NO2
- NOx
- NH3
- CO
- SO2
- O3
- Benzene
- Toluene
- Xylene

## 🤖 Machine Learning Models

### Linear Regression

- MAE: 31.17
- RMSE: 59.45
- R² Score: 0.807

### Random Forest Regression

- MAE: 20.83
- RMSE: 40.84
- R² Score: 0.909

Random Forest Regression was selected as the final model because it achieved lower MAE and RMSE and a higher R² score.

## 📊 Model Performance

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 31.17 | 59.45 | 0.807 |
| Random Forest | 20.83 | 40.84 | 0.909 |

## 🔍 Important Features

According to the Random Forest model, the most important features were:

1. PM2.5
2. CO
3. NO
4. PM10
5. O3

Feature importance represents the model's contribution from each feature and should not be interpreted as direct causation.

## 🖥️ Streamlit Dashboard

The Streamlit application allows users to:

1. Enter pollutant measurements.
2. Predict AQI using the trained Random Forest model.
3. View the predicted AQI.
4. View the AQI category.
5. See a health recommendation.
6. View an AQI level indicator.
7. Review the entered pollutant values.

## 🔄 Project Workflow

```text
Air Quality Dataset
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Selection
        ↓
Missing Value Handling
        ↓
Train-Test Split
        ↓
Linear Regression
        ↓
Random Forest Regression
        ↓
Model Evaluation
        ↓
Final Model Selection
        ↓
Model Saving
        ↓
Streamlit Dashboard
        ↓
AQI Prediction
## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Air_Quality_Analytics