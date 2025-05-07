# HackLab-Data-Dominatorsrs


## Telco Churn Prediction – Hacklab Data Dominators 2025

This project predicts customer churn for a telecom company using both structured and unstructured data. It was built for the Hacklab 2025 competition.

---

## What the Project Does

* Cleans and processes structured data (e.g., customer plans, payments)
* Analyzes customer complaints using Google’s Gemini AI (sentiment + categories)
* Combines both types of data into a single dataset
* Trains an XGBoost model to predict churn
* Explains predictions with top 3 reasons per customer
* Generates a final report using an LLM for support agents

---

## Main Steps

1. **Data Cleaning** – Handles missing values and outliers
2. **Feature Engineering** – Adds useful columns (e.g., avg spend, total services)
3. **Text Analysis** – Uses an LLM to score complaint sentiment and categorize topics
4. **Merging Data** – Combines structured and unstructured info
5. **Model Training** – Uses XGBoost + hyperparameter tuning
6. **Prediction** – Returns churn prediction + top 3 churn drivers
7. **Reporting** – Generates a summary for agents using Gemini

---

## Requirements

* Google Colab
* Python (pandas, sklearn, xgboost)
* Google API key for Gemini (`genai`)

---

## Output

* Cleaned dataset
* Trained model (`churn_pipeline.pkl`)
* CSV with churn predictions and explanations
* Summary report for support agents

---

Let me know if you want this exported as a file or formatted for uploading directly to GitHub.
