import pandas as pd
import numpy as np
from google.colab import files
# ┌────────────────────────────────────────────────────┐
# │ 1 ) Upload raw file                               │
# └────────────────────────────────────────────────────┘
uploaded = files.upload()  # UI prompt (drag‑and‑drop or file picker)
if not uploaded:
    raise ValueError("No file selected – please upload a CSV or Excel document.")
raw_name = next(iter(uploaded))  # first/only uploaded filename
# ┌────────────────────────────────────────────────────┐
# │ 2 ) Load into DataFrame                            │
# └────────────────────────────────────────────────────┘
if raw_name.lower().endswith(".csv"):
    df = pd.read_csv(raw_name)
elif raw_name.lower().endswith((".xls", ".xlsx")):
    df = pd.read_excel(raw_name)
else:
    raise ValueError("Unsupported file type – please upload .csv, .xls, or .xlsx")
# Ensure critical numeric columns are numeric (TotalCharges sometimes ships as object)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# ┌────────────────────────────────────────────────────┐
# │ 3 ) Feature engineering                            │
# └────────────────────────────────────────────────────┘
# 3.1 Average monthly spend
#     (protect from divide‑by‑zero when tenure==0)
df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
df["avg_monthly_spend"].fillna(0, inplace=True)
# 3.2 Spend variance (trend)
df["spend_diff"] = df["MonthlyCharges"] - df["avg_monthly_spend"]
# 3.3 Tenure bucket
bins = [0, 12, 24, 48, np.inf]
labels = ["0-12 months", "13-24 months", "25-48 months", "49+ months"]
df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels, right=True, include_lowest=True)
# 3.4 Auto‑payment flag
df["auto_pay"] = df["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)
# 3.5 Total services subscribed
service_cols = [
    "PhoneService",
    "InternetService",      # counts as 1 if NOT "No"
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
def summed_services(row):
    total = 0
    for col in service_cols:
        val = str(row[col]).strip().lower()
        if col == "InternetService":
            total += 0 if val == "no" else 1
        else:
            total += 1 if val == "yes" else 0
    return total
df["total_services_subscribed"] = df.apply(summed_services, axis=1)
# 3.6 High‑spender flag
threshold = df["MonthlyCharges"].mean() + df["MonthlyCharges"].std()
df["high_spender"] = (df["MonthlyCharges"] > threshold).astype(int)
# ┌────────────────────────────────────────────────────┐
# │ 4 ) Re‑order so churn sits last                    │
# └────────────────────────────────────────────────────┘
if "Churn" in df.columns:
    ordered_cols = [c for c in df.columns if c != "Churn"] + ["Churn"]
    df = df[ordered_cols]
# ┌────────────────────────────────────────────────────┐
# │ 5 ) Save & trigger download                        │
# └────────────────────────────────────────────────────┘
output_name = f"processed_{raw_name.rsplit('.', 1)[0]}.csv"
df.to_csv(output_name, index=False)
files.download(output_name)
print(f"✅ Done! Your processed file is downloading as ➜ {output_name}")
