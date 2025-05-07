# HackLab-Data-Dominatorsrs


## Telco Churn Prediction – Hacklab Data Dominators 2025

This project predicts customer churn for a telecom company using both structured and unstructured data. It was built for the Hacklab 2025 competition.

---

## What the Project Does

* Cleans and processes structured data (e.g., customer plans, payments)
* Analyzes customer complaints using Google’s Gemini AI (sentiment + categories)
* Combines structured and unstructured data
* Trains an XGBoost model to predict churn
* Explains predictions with top 3 reasons per customer
* Generates a report for support agents using an LLM

---

## Main Steps

1. **Data Cleaning** – Fixes missing values and outliers
2. **Feature Engineering** – Adds useful metrics (e.g., avg spend, services used)
3. **Text Analysis** – Uses Gemini to analyze complaints
4. **Data Merge** – Combines all data into one dataset
5. **Model Training** – Uses XGBoost with hyperparameter tuning
6. **Predictions** – Includes churn probability and top 3 drivers
7. **Final Report** – Gemini writes a summary for call center agents

---

## Requirements

* Google Colab
* Python libraries: `pandas`, `scikit-learn`, `xgboost`, `joblib`
* Google API key for Gemini (`genai`)

---

## Notes

* ✅ All code was written and executed in **Google Colab**
* 📂 The code on GitHub is the **original source** used during the project

---

## Output

* Cleaned and processed datasets
* Trained model: `churn_pipeline.pkl`
* CSV with churn predictions and driver explanations
* LLM-generated summary report
