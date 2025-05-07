# 📊 HackLab-Data-Dominators

## Telco Churn Prediction – Hacklab Data Dominators 2025

This project predicts customer churn for a telecom company using both structured and unstructured data. It was built as part of the Hacklab 2025 competition.

---

## 🚀 What the Project Does

* Cleans and processes structured customer data
* Analyzes complaint text using Gemini AI (sentiment & topic classification)
* Trains an XGBoost model to predict churn
* Explains predictions with the top 3 churn drivers per customer
* Generates LLM-powered summaries tailored for customer support agents on live calls

---

## 💼 Ideal Use Case

This tool is built for customer support teams. It functions as a real-time co-pilot: 📲 While speaking with customers, agents can quickly view:

* Churn risk
* Top 3 drivers of dissatisfaction
* Actionable suggestions to retain them

---

## 📘️ USER GUIDE: Churn Prediction Co-Pilot (Google Colab)

### ▶️ Run the Notebook in Colab

This notebook predicts churn and generates a human-friendly summary using Gemini 2.5.
**[Try it here →](https://colab.research.google.com/drive/1PonU3l5-CeEh3dUHJHlJ-LWFWFKF28oz?usp=sharing)**

---

## 📁 Files Required

| File Name           | Description                                     |
| ------------------- | ----------------------------------------------- |
| Model execution.py  | Main script to execute churn prediction         |
| churn\_pipeline.pkl | Trained machine learning pipeline               |
| Data sample.csv     | Sample input file with customer data (no Churn) |

---

## ⚙️ How to Use

1. Open the Colab notebook using the link above
2. Upload:

   * Your .pkl pipeline file
   * Your customer .csv file (excluding the Churn column)
3. The notebook:

   * Predicts churn and churn probability
   * Identifies top 3 churn drivers per customer
   * Outputs a downloadable CSV
   * Displays a Gemini-powered summary for support agents

---

## 🧠 Output

* `churn_predictions_with_drivers.csv` — detailed predictions per customer
* Gemini summary — easy-to-read insights and suggested responses for agents
* No setup needed — the Colab includes a valid Gemini API key

---

## ✅ Hacklab 2025 Deliverables

### 1. Model Documentation

**Model Used:** XGBoost (with SHAP-style explanations via `predict_contribs=True`)

We chose XGBoost for its strong performance on structured data, built-in support for explainability, and efficient training. Unlike deep learning or reinforcement learning, it works well with small to medium-sized tabular datasets and allows real-time predictions with low latency—ideal for live support use cases.

Why not deep learning or GenAI for churn classification?
While we used Gemini (LLM) for interpreting unstructured complaints, classical ML outperforms deep learning on structured churn data due to better interpretability and lower data/compute needs.

**Feature Engineering:**

* Tenure buckets
* Average monthly spend, auto-pay detection, service counts
* Sentiment score, complaint volume, dominant complaint category

**Validation Metrics:**

* Accuracy: 0.8133
* ROC-AUC: 0.8638
* F1-score: macro 0.74, weighted 0.80

**Segmentation Insight:**
Highest churn risk among customers on month-to-month contracts with fiber optic internet and no support services

### 2. Actionable Business Recommendations

* Target month-to-month contract customers with loyalty incentives
* Proactively contact users with high complaint frequency or poor sentiment
* Offer support bundles to reduce churn risk linked to lack of TechSupport or OnlineSecurity
* Estimated cost of targeting the top 20% at-risk customers is recoverable within 6–8 months from retained revenue

### 3. Business Strategy

* Deploy support co-pilot tool to agents via web interface or internal dashboard
* Integrate LLM-generated insights with CRM systems to prep agents pre-call
* Long term: Fine-tune sentiment model on internal complaint history for cost efficiency

---

## 🧪 Methodology

**[See the full training pipeline →](https://colab.research.google.com/drive/1SULtN6LD8MvFjLFeF5ATwcwq5iWjM4GO?usp=sharing)**

### 1. Data Preparation

* Structured CSV cleaned for missing values, outliers capped using IQR, binary fields coerced to categorical.
* Features engineered including average monthly spend, tenure buckets, number of services, auto-pay flag, and more.
* Unstructured Excel file of customer complaints enriched with Gemini-based sentiment scoring (1–5) and topic classification.

### 2. Data Fusion

* Aggregated unstructured insights by customer: average sentiment, complaint count, and most frequent complaint topic.
* Merged structured and enriched unstructured features into one training dataset.

### 3. Modeling

* Train/test split (80/20), pipeline preprocessing with scaling and one-hot encoding.
* XGBoost classifier with randomized hyperparameter search (30 iterations, 3-fold CV).
* Evaluation using accuracy, ROC-AUC, F1-score, and per-class performance.

### 4. Explainability

* Top-3 churn drivers extracted per customer using SHAP-like feature contributions (`pred_contribs=True`).

### 5. LLM Summary Generation

* For all customers flagged at high churn risk, a Gemini-powered summary generates:

  * Risk overview
  * Top reasons for churn
  * Agent-ready retention strategies

---

## 🛠️ Model Training Results

* ✅ Rows processed: 7,043
* ✔️ Train/Test Split: 80/20

### 🏅 Best XGBoost Params:

```python
{subsample: 0.85,
 n_estimators: 300,
 max_depth: 5,
 learning_rate: 0.05,
 gamma: 5,
 colsample_bytree: 0.85}
```

### ✅ Performance Metrics

* Accuracy: 0.8133
* ROC-AUC: 0.8638

**Classification Report:**

```
               precision    recall  f1-score   support
           0       0.84      0.92      0.88      1035
           1       0.70      0.53      0.60       374
    accuracy                           0.81      1409
   macro avg       0.77      0.72      0.74      1409
weighted avg       0.80      0.81      0.80      1409
```

**Confusion Matrix:**

```
[[949  86]
 [177 197]]
```

---

## 🌐 Feature Importance (Top 20 - XGBoost)

Most impactful features driving churn prediction:

* Contract\_Month-to-month
* InternetService\_Fiber optic
* TechSupport\_No
* OnlineSecurity\_No
* InternetService\_DSL
* Contract\_Two year
* tenure\_bucket\_0-12 months
* avg\_sentiment
* tenure\_bucket\_49+ months
* PaymentMethod\_Electronic check
* ...and more

---

## 📊 ROC Curve

The model achieves an AUC of 0.86, indicating strong predictive performance in distinguishing churn vs. non-churn.

---

## 📂 Project Output Summary

* Cleaned dataset
* Trained model: `churn_pipeline.pkl`
* Churn predictions with driver attribution
* LLM-generated report for customer support
* Built with Google Colab, XGBoost, and Gemini 2.5
