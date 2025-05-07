# ================================================================
# 1.  Basic setup – run this cell first in Google Colab
# ================================================================
import pandas as pd
import numpy as np

# ⬇️  Adjust path if needed
file_path = '/Structured data.csv'

# ------------------------------------------------
# 2.  Load data
# ------------------------------------------------
df = pd.read_csv(file_path)
print(f"Loaded shape: {df.shape}")
display(df.head())

# ------------------------------------------------
# 3.  Quick audit – dtypes & obvious issues
# ------------------------------------------------
print("\n🔎  Raw dtypes")
print(df.dtypes)

# ------------------------------------------------
# 4.  ✨  MISSING-VALUE CLEANUP
# ------------------------------------------------
# • Common hidden missings → NaN
df = df.applymap(lambda x: np.nan if (isinstance(x, str) and str(x).strip() in 
                                      ['', 'NA', 'N/A', 'na', 'n/a', '?', '-', '–']) else x)

# • Coerce “object” columns to numeric when possible
for col in df.columns:
    if df[col].dtype == object:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass

print("\n🧹  Null counts after coercion:")
display(df.isna().sum())

# ------------------------------------------------
# 4b.  SPECIAL-CASE: SeniorCitizen AS CATEGORICAL
# ------------------------------------------------
# Treat the binary flag correctly:
if 'SeniorCitizen' in df.columns:
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('category')

# ------------------------------------------------
# 4c.  IMPUTATION: Drop/Fill
# ------------------------------------------------
thresh = len(df.columns) - 2
df = df.dropna(thresh=thresh)

# Now pick numeric vs categorical (SeniorCitizen will be in cat_cols)
num_cols = [c for c in df.select_dtypes(include=['number']).columns
            if c != 'SeniorCitizen']
cat_cols = df.select_dtypes(exclude=['number']).columns

# Fill missing
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# ------------------------------------------------
# 5.  🩹  OUTLIER HANDLING (IQR “fence” method)
# ------------------------------------------------
def cap_iqr(series, factor=1.5):
    """Clip values outside [Q1 – factor·IQR, Q3 + factor·IQR]."""
    q1, q3 = series.quantile([.25, .75])
    iqr = q3 - q1
    return series.clip(q1 - factor*iqr, q3 + factor*iqr)

# Only apply to true numeric columns (SeniorCitizen is skipped)
for col in num_cols:
    df[col] = cap_iqr(df[col])

# ------------------------------------------------
# 6.  ✅  Tidy output check
# ------------------------------------------------
print("\n🎉  Cleaned shape:", df.shape)
print(df.dtypes)
display(df.describe(include='all').T)

# ------------------------------------------------
# 7.  (Optional) Export the cleaned file
# ------------------------------------------------
clean_fname = 'Structured data_cleaned.csv'
df.to_csv(clean_fname, index=False)

from google.colab import files
files.download(clean_fname)
