# 2. Imports
import os
import pandas as pd
import time
from google import genai
from google.genai import types
from google.colab import files

# 3. Upload your workbook
print("📁 Upload your 'Unstructured data.xlsx':")
uploaded = files.upload()
file_name = next(iter(uploaded))

# 4. Read it
df = pd.read_excel(file_name)
print("✅ Columns found:", df.columns.tolist())

# 5. 🔑 Put your key here and init client
os.environ["API_KEY"] = "AIzaSyAkO2ZZmmJCAwmUsyrBn11vpk1zFVXQADA"
client = genai.Client(api_key=os.environ["API_KEY"])

# 6. Complaint column name
COMPLAINT_COL = "complaint"

# 7. Helper: split into chunks of size N
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

# 8. STEP 1 – Generate up to 10 one-word categories
all_texts = df[COMPLAINT_COL].astype(str).tolist()
prompt = (
    "You are given a list of customer complaints. "
    "Please identify up to 10 broad one-word categories that cover these "
    "complaints (e.g., 'Billing', 'Service', 'Quality', etc.). "
    "Answer with only the category words, separated by commas, in lowercase."
    "\n\n"
    + "\n".join(f"- {t}" for t in all_texts)
)
resp = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=prompt,
    config=types.GenerateContentConfig(temperature=0)
)
cats_text = resp.text.strip()
categories = [c.strip() for c in cats_text.split(",")]
print("🏷️ Categories:", categories)

# 9. STEP 2 – Batched categorization using same 20×/15 rpm rate limits
def categorize_batched(texts, categories, batch_size=20, rpm=15):
    delay = 60.0 / rpm
    all_labels = []
    for batch in chunk_list(texts, batch_size):
        numbered = "\n".join(f"{i+1}. {txt}" for i, txt in enumerate(batch))
        prompt = (
            f"Given these categories: {', '.join(categories)}\n\n"
            "Assign each customer complaint to the single best matching category. "
            "Answer with only the category word for each complaint, in order, "
            "separated by commas.\n\n"
            f"{numbered}"
        )
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0)
        )
        text = resp.text.strip()
        # parse and fallback if needed
        labels = [lbl.strip().lower() for lbl in text.split(",")]
        if len(labels) != len(batch):
            # fallback to one-by-one if parse fails
            labels = []
            for txt in batch:
                r = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=(
                        f"Categories: {', '.join(categories)}\n\n"
                        f"Complaint: {txt}\n\n"
                        "Assign the single best category. Answer with only the word."
                    ),
                    config=types.GenerateContentConfig(temperature=0)
                )
                labels.append(r.text.strip().lower())
        all_labels.extend(labels)
        time.sleep(delay)
    return all_labels

print("🔍 Categorizing complaints in batches…")
labels = categorize_batched(all_texts, categories)
df["category"] = labels

# 10. Move 'category' to be the leftmost column
cols = ["category"] + [c for c in df.columns if c != "category"]
df = df[cols]

# 11. Save & download
out = "with_categories.xlsx"
df.to_excel(out, index=False)
print("✅ Done:", out)
files.download(out)
