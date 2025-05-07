# 1. Imports
import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from google import genai
from google.genai import types
from google.colab import files
import time

# 2. Configuration
PIPELINE_PATH = "churn_pipeline.pkl"
os.environ["API_KEY"] = "AIzaSyAPrTyFXKJbs_wLLZcaC1JIkkZOCpT9Ieo"
client = genai.Client(api_key=os.environ["API_KEY"])
MODEL_NAME = "gemini-2.0-flash"

# 3. Upload your new data CSV (no 'Churn' column)
print("🔼 Please upload your new data CSV (no 'Churn' column)…")
uploaded = files.upload()
if not uploaded:
    raise RuntimeError("No file uploaded.")
data_path = next(iter(uploaded))

# 4. Load pipeline and data
pipeline = joblib.load(PIPELINE_PATH)
df_new = pd.read_csv(data_path)
print(f"✅ Loaded {df_new.shape[0]} rows, {df_new.shape[1]} columns from {data_path}")

# 5. Make predictions
preds = pipeline.predict(df_new)
probs = pipeline.predict_proba(df_new)[:, 1]

# 6. Compute top-3 churn drivers
def get_top_drivers(raw_df, pipeline, cat_cols, num_cols, top_n=3):
    pre     = pipeline.named_steps["pre"]
    booster = pipeline.named_steps["model"].best_estimator_
    enc     = pre.named_transformers_["cat"].named_steps["encoder"]
    encoded = enc.get_feature_names_out(cat_cols)
    features = list(num_cols) + list(encoded)
    Xt = pre.transform(raw_df)
    dmat = xgb.DMatrix(Xt, feature_names=features)
    contribs = booster.get_booster().predict(dmat, pred_contribs=True)
    names, imps = [], []
    for row in contribs:
        vals = row[:-1]
        idx  = np.argsort(np.abs(vals))[::-1][:top_n]
        names.append([features[i] for i in idx])
        imps.append([float(vals[i]) for i in idx])
    return names, imps

pre      = pipeline.named_steps["pre"]
num_cols = pre.transformers_[0][2]
cat_cols = pre.transformers_[1][2]
dnames, dimps = get_top_drivers(df_new, pipeline, cat_cols, num_cols, top_n=3)

# 7. Assemble output DataFrame
out = df_new.copy()
out["PredictedChurn"]    = preds
out["ChurnProbability"]  = probs
if "MonthlyCharges" in out.columns:
    out["LifetimeValueRisk"] = np.where(preds==1, out["MonthlyCharges"]*12, np.nan)
for i in range(3):
    out[f"Driver{i+1}"] = [names[i] if len(names)>i else None for names in dnames]
    out[f"Impact{i+1}"] = [imps[i]  if len(imps)>i  else None for imps in dimps]

# 8. Save & download CSV
output_csv = "churn_predictions_with_drivers.csv"
out.to_csv(output_csv, index=False)
print(f"💾 Saved predictions to {output_csv}")
files.download(output_csv)

# 9. Build prompt and call Gemini 2.0 Flash
with open(output_csv, "r") as f:
    csv_data = f.read()

# Improved prompt with structural guidance and token constraints
prompt = """
You are a customer-support co-pilot helping a new agent on a Telco call. 
Respond as if speaking directly to the agent in clear, conversational language.

1. FIRST SECTION - OVERVIEW:
   - Total number of customers in the dataset, churning and not
   - Current churn rate (percentage of customers predicted to churn)
   - Average annual Lifetime Value at risk across all churning customers

2. SECOND SECTION - CUSTOMER DETAILS:
   For each customer predicted to churn (PredictedChurn=1), provide:
   - CustomerID
   - Churn probability as a percentage
   - Annual Lifetime Value at risk (dollar amount)
   - The top three drivers of churn in order of impact
   - For each driver, provide ONE specific, actionable suggestion the agent can say

FORMAT YOUR RESPONSE WITH CLEAR HEADINGS AND BULLET POINTS.
BE CONCISE AND FOCUS ON ACTIONABLE INSIGHTS, cover all the churning customers.

Here's the customer data:
""" + csv_data

print("🤖 Generating summary via Gemini 2.0 Flash…")

# Configure for more complete responses with chunking to handle token limits
def generate_with_retry(prompt, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=4096,  # Increased token limit
                    top_p=0.95,
                    top_k=40
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)  # Brief pause before retry
    return "Error: Failed to generate complete response after multiple attempts."

# Generate and print the summary
summary = generate_with_retry(prompt)
print("\n" + "="*50)
print("CUSTOMER CHURN ANALYSIS")
print("="*50 + "\n")
print(summary)
