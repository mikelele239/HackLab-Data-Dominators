import io
import pandas as pd
from google.colab import files

# ─────────────────────────────────────────
# 1️⃣  Upload the structured dataset (Dataset 3)
# ─────────────────────────────────────────
print("⬆️  Please upload Dataset 3 (structured CSV, e.g. 'Feature engineered data.csv'):")
up_struct = files.upload()
struct_name = next(iter(up_struct))               # first uploaded filename
df_struct = pd.read_csv(io.BytesIO(up_struct[struct_name]))

# ─────────────────────────────────────────
# 2️⃣  Upload the unstructured dataset (Dataset 4)
# ─────────────────────────────────────────
print("\n⬆️  Please upload Dataset 4 (unstructured XLSX, e.g. 'with_categories_and_sentiment.xlsx'):")
up_unstr = files.upload()
unstr_name = next(iter(up_unstr))
df_unstr  = pd.read_excel(io.BytesIO(up_unstr[unstr_name]))

# ─────────────────────────────────────────
# 3️⃣  Normalise the key column name
#     (works whether it is 'customerID' or 'customer_id')
# ─────────────────────────────────────────
for df in (df_struct, df_unstr):
    if "customer_id" in df.columns:
        df.rename(columns={"customer_id": "customerID"}, inplace=True)

key = "customerID"   # unified primary key

# ─────────────────────────────────────────
# 4️⃣  Feature‑engineer the unstructured data
# ─────────────────────────────────────────
#   • Average sentiment score
#   • Number of complaints
agg = (df_unstr
       .groupby(key, as_index=False)
       .agg(avg_sentiment      = ("Sentiment_Score", "mean"),
            num_complaints     = ("complaint",       "count")))

#   • Most prevalent complaint category
cat_stats = (df_unstr
             .groupby([key, "category"])
             .agg(freq          = ("category",        "size"),
                  avg_sent_cat  = ("Sentiment_Score", "mean"))
             .reset_index())

# Select the category with
#   1) highest frequency
#   2) if tied → higher avg_sent_cat (5 = more negative)
cat_stats.sort_values(["freq", "avg_sent_cat"], ascending=[False, False], inplace=True)
best_cat = (cat_stats
            .drop_duplicates(subset=key, keep="first")
            .loc[:, [key, "category"]]
            .rename(columns={"category": "prev_complaint_category"}))

# Combine the three engineered features
features = agg.merge(best_cat, on=key, how="outer")

# ─────────────────────────────────────────
# 5️⃣  Merge with the structured data
# ─────────────────────────────────────────
df_merged = df_struct.merge(features, on=key, how="left")

# Fill NaNs (no unstructured data) with None
df_merged[["avg_sentiment", "num_complaints", "prev_complaint_category"]] = (
    df_merged[["avg_sentiment", "num_complaints", "prev_complaint_category"]].where(
        df_merged[["avg_sentiment", "num_complaints", "prev_complaint_category"]].notna(), None)
)

# ─────────────────────────────────────────
# 6️⃣  Ensure the Churn column is last
# ─────────────────────────────────────────
churn_col = next((c for c in df_merged.columns if c.lower() == "churn"), None)
if churn_col:
    reordered = [c for c in df_merged.columns if c != churn_col] + [churn_col]
    df_merged = df_merged[reordered]

# ─────────────────────────────────────────
# 7️⃣  Download the result
# ─────────────────────────────────────────
out_file = "dataset3_with_unstructured_features.csv"
df_merged.to_csv(out_file, index=False)
files.download(out_file)          # triggers browser download
print(f"\n✅  Finished!  '{out_file}' is downloading…")
