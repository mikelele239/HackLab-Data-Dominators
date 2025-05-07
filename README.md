# 📊 HackLab-Data-Dominators

## Telco Churn Prediction – Hacklab Data Dominators 2025

This project predicts customer churn for a telecom company using both structured and unstructured data. It was built as part of the Hacklab 2025 competition.

---

## 🚀 What the Project Does

* Cleans and processes structured customer data
* Analyzes complaint text using **Gemini AI** (sentiment & topic classification)
* Trains an XGBoost model to predict churn
* Explains predictions with the top 3 churn drivers per customer
* Generates **LLM-powered summaries** tailored for customer support agents on live calls

---

## 💼 Ideal Use Case

This tool is built for **customer support teams**. It functions as a **real-time co-pilot**:
📞 While speaking with customers, agents can quickly view:

* Churn risk
* Top 3 drivers of dissatisfaction
* Actionable suggestions to retain them

---

## 📘️ USER GUIDE: Churn Prediction Co-Pilot (Google Colab)

### ▶️ [Run the Notebook in Colab](https://colab.research.google.com/drive/1PonU3l5-CeEh3dUHJHlJ-LWFWFKF28oz?usp=sharing)

This notebook predicts churn and generates a human-friendly summary using Gemini 2.0.

### 📁 Files Required

| File Name            | Description                                       |
| -------------------- | ------------------------------------------------- |
| `Model execution.py` | Main script to execute churn prediction           |
| `churn_pipeline.pkl` | Trained machine learning pipeline                 |
| `Data sample.csv`    | Sample input file with customer data (no `Churn`) |

### ⚙️ How to Use

1. Open the Colab notebook using the link above
2. Upload:

   * Your `.pkl` pipeline file
   * Your customer `.csv` file (excluding the `Churn` column)
3. The notebook:

   * Predicts churn and churn probability
   * Identifies top 3 churn drivers per customer
   * Outputs a downloadable CSV
   * Displays a Gemini-powered summary for support agents

### 🧠 Output

* `churn_predictions_with_drivers.csv` — detailed predictions per customer
* Gemini summary — easy-to-read insights and suggested responses for agents

> No setup needed — the Colab includes a valid Gemini API key.

---

## ✅ Hacklab 2025 Deliverables

### 1. Model Documentation

* **Model Used:** XGBoost (with SHAP-style explanations via `predict_contribs=True`)
* **Feature Engineering:**

  * Tenure buckets
  * Average monthly spend, auto-pay detection, service counts
  * Sentiment score, complaint volume, dominant complaint category
* **Validation Metrics:**

  * Accuracy: **0.8133**
  * ROC-AUC: **0.8638**
  * F1-score: macro **0.74**, weighted **0.80**
* **Segmentation Insight:**

  * Highest churn risk among customers on **month-to-month contracts** with **fiber optic internet** and **no support services**

### 2. Actionable Business Recommendations

Most of these recommendations will be **custom created via Gemini** after customer analysis, enabling agents to respond with **personalized retention strategies** in real time. However, here are some general trends based on our model insights:

* Target **month-to-month contract** customers with loyalty incentives
* Proactively contact users with **high complaint frequency or poor sentiment**
* Offer **support bundles** to reduce churn risk linked to lack of TechSupport or OnlineSecurity
* Estimated cost of targeting the **top 20% at-risk customers** is recoverable within **6–8 months** from retained revenue

### 3. Business Strategy

* Deploy support co-pilot tool to agents via web interface or internal dashboard
* Integrate LLM-generated insights with CRM systems to prep agents pre-call
* Long term: Fine-tune sentiment model on internal complaint history for cost efficiency

---

## 🛠️ Model Training Results

* ✅ **Rows processed**: 7,043
* ✔️ Train/Test Split: 80/20
* 🏅 **Best XGBoost Params**:

```python
{subsample: 0.85,
 n_estimators: 300,
 max_depth: 5,
 learning_rate: 0.05,
 gamma: 5,
 colsample_bytree: 0.85}
```

### ✅ Performance Metrics

* **Accuracy**: 0.8133
* **ROC-AUC**: 0.8638

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

* `Contract_Month-to-month`
* `InternetService_Fiber optic`
* `TechSupport_No`
* `OnlineSecurity_No`
* `InternetService_DSL`
* `Contract_Two year`
* `tenure_bucket_0-12 months`
* `avg_sentiment`
* `tenure_bucket_49+ months`
* `PaymentMethod_Electronic check`

...and more

---

## 📊 ROC Curve

The model achieves an **AUC of 0.86**, indicating strong predictive performance in distinguishing churn vs. non-churn.

---

## 📂 Project Output Summary

* Cleaned dataset
* Trained model: `churn_pipeline.pkl`
* Churn predictions with driver attribution
* LLM-generated report for customer support

> Built with Google Colab, XGBoost, and Gemini 2.0
