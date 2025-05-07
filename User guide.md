# 📘 USER GUIDE: Churn Prediction Pipeline (Google Colab)

This guide helps you run the **churn prediction + AI insights pipeline** using Google Colab and Gemini 2.0. The script takes in customer data, predicts churn, and generates a summary with actionable suggestions.

---

## 🧩 Key Files Used

| File Name            | Description                                           |
| -------------------- | ----------------------------------------------------- |
| `Model execution.py` | 🔁 Main script to execute the churn prediction flow   |
| `churn_pipeline.pkl` | 🧠 Trained machine learning pipeline                  |
| `Data sample.csv`    | 📂 Example input file with customer data (no `Churn`) |

---

## ⚙️ Requirements

* Google account to use [Google Colab](https://colab.research.google.com)
* A CSV file with new customer data (excluding the `Churn` column)
* A valid Gemini 2.0 API key from Google

---

## 🧑‍💻 How to Run in Google Colab

1. **Open `Model execution.py` in Google Colab**

   * Either upload the file or paste its contents into a new notebook

2. **Upload Required Files**

   * Upload:

     * `churn_pipeline.pkl`
     * Your CSV file (e.g., `Data sample.csv`) when prompted

3. **Execute All Cells**

   * Use `Runtime > Run all` to execute the pipeline

4. **Results**

   * 📄 Output file: `churn_predictions_with_drivers.csv`
   * 🤖 AI summary printed in the notebook
   * 🔽 Automatic download of predictions CSV

---

## 📂 Input CSV Format

* Must include **all features** used during model training
* Must **exclude the `Churn`** column
* Should include `MonthlyCharges` for Lifetime Value estimation (optional but helpful)

---

## 🔐 Gemini API Setup

In `Model execution.py`, replace the placeholder API key:

```python
os.environ["API_KEY"] = "your-api-key-here"
```

> Never expose your API key publicly.

---

## 🧠 Output Summary

The script uses Gemini 2.0 to generate:

* Churn rate and value risk overview
* Detailed breakdown of each customer at risk
* Top 3 churn drivers per customer + suggested interventions

---

## 🛠 Troubleshooting

| Problem                      | Fix                                             |
| ---------------------------- | ----------------------------------------------- |
| No file uploaded             | Re-run the upload cell and choose your CSV file |
| Gemini errors                | Double-check your API key and internet status   |
| Mismatched columns           | Ensure input CSV matches training features      |
| Missing `churn_pipeline.pkl` | Upload the model file manually in Colab         |
