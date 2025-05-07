# 2. Imports
import os
import pandas as pd
from google import genai
from google.genai import types
from google.colab import files
import time
import math


# 3. Upload your workbook
print("📁 Upload your 'Unstructured data.xlsx':")
uploaded = files.upload()
file_name = next(iter(uploaded))

# 4. Read it
df = pd.read_excel(file_name)
print("✅ Columns found:", df.columns.tolist())

# 5. 🔑 Put your key here and init client
os.environ["API_KEY"] = "AIzaSyAkO2ZZmmJCAwmUsyrBn11vpk1zFVXQADA"   # ← REPLACE with your Google API key
client = genai.Client(api_key=os.environ["API_KEY"]) 

# 6. Complaint column
complaint_column = "complaint"

# 7a. Helper: split into chunks of size N
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

# 7b. Batched sentiment rating
def rate_sentiments_batched(texts, batch_size=20, rpm=15):
    all_scores = []
    delay = 60.0 / rpm
    for batch in chunk_list(texts, batch_size):
        numbered = "\n".join(f"{i+1}. {txt}" for i, txt in enumerate(batch))
        prompt = (
            "Rate the anger level of each of the following customer complaints on a scale "
            "from 1 (neutral/mild) to 5 (very angry). Answer with only the numbers separated "
            "by commas, in order.\n\n"
            f"{numbered}"
        )
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0)
        )
        text = resp.text.strip()
        try:
            scores = [int(x) for x in text.split(",")]
            if len(scores) != len(batch):
                raise ValueError(f"{len(scores)}≠{len(batch)}")
        except Exception:
            # Fallback to single calls if parse fails
            scores = [rate_sentiment(t) for t in batch]
        all_scores.extend(scores)
        time.sleep(delay)   # <-- now works
    return all_scores

# 8. Apply batched rating
print(f"🔍 Rating '{complaint_column}' in batches…")
texts = df[complaint_column].astype(str).tolist()
df["Sentiment_Score"] = rate_sentiments_batched(texts)

# 9. Save & download
out = "with_sentiment.xlsx"
df.to_excel(out, index=False)
print("✅ Done:", out)
files.download(out)


# 3. Upload your workbook
print("📁 Upload your 'Unstructured data.xlsx':")
uploaded = files.upload()
file_name = next(iter(uploaded))

# 4. Read it
df = pd.read_excel(file_name)
print("✅ Columns found:", df.columns.tolist())

# 5. 🔑 Put your key here and init client
os.environ["API_KEY"] = "AIzaSyAkO2ZZmmJCAwmUsyrBn11vpk1zFVXQADA"   # ← REPLACE with your Google API key
client = genai.Client(api_key=os.environ["API_KEY"])  # ← instantiate the Gemini client :contentReference[oaicite:1]{index=1}

# 6. Complaint column
complaint_column = "complaint"

# 7a. Helper: split into chunks of size N
def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

# 7b. Batched sentiment rating
def rate_sentiments_batched(texts, batch_size=20, rpm=15):
    """
    Returns a list of ints, one per text in 'texts', rating anger 1–5.
    Respects at most 'rpm' requests per minute.
    """
    all_scores = []
    delay = 60.0 / rpm
    for batch in chunk_list(texts, batch_size):
        # build numbered prompt
        numbered = "\n".join(f"{i+1}. {txt}" for i, txt in enumerate(batch))
        prompt = (
            "Rate the anger level of each of the following customer complaints on a scale "
            "from 1 (neutral/mild) to 5 (very angry). "
            "Answer with only the numbers separated by commas, in order.\n\n"
            f"{numbered}"
        )
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0)
        )
        # parse “1,2,3,1,5” → [1,2,3,1,5]
        text = resp.text.strip()
        try:
            scores = [int(x) for x in text.split(",")]
            if len(scores) != len(batch):
                raise ValueError(f"{len(scores)}≠{len(batch)}")
        except Exception:
            # fallback: one-by-one (slower) if parsing fails
            scores = [rate_sentiment(t) for t in batch]
        all_scores.extend(scores)
        # throttle
        time.sleep(delay)
    return all_scores

# 8. Apply batched rating
print(f"🔍 Rating '{complaint_column}' in batches…")
texts = df[complaint_column].astype(str).tolist()
df["Sentiment_Score"] = rate_sentiments_batched(texts)

# 9. Save & download
out = "with_sentiment.xlsx"
df.to_excel(out, index=False)
print("✅ Done:", out)
files.download(out)
