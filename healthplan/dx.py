import pandas as pd
import requests
import io
import os

# -------------------- CONFIG --------------------
OUTPUT_PATH = "icd10_master.csv"
SOURCE_URL = "https://raw.githubusercontent.com/k4m1113/ICD-10-CSV/master/codes.csv"

# -------------------- DOWNLOAD & PARSE --------------------

print("📥 Downloading ICD-10 diagnosis codes from:", SOURCE_URL)
resp = requests.get(SOURCE_URL)
resp.raise_for_status()

df_icd = pd.read_csv(io.StringIO(resp.text), header=None, names=[
    "category", "diagnosis_code", "full_code", "short_desc", "long_desc", "category_title"
], dtype=str)

# -------------------- CLEANING & RENAME --------------------

print("Available columns:", df_icd.columns.tolist())

df_icd = df_icd[["category", "diagnosis_code", "full_code", "short_desc", "long_desc"]]
df_icd = df_icd.drop_duplicates(subset=["full_code"])

# Optional sort
df_icd = df_icd.sort_values(by="full_code").reset_index(drop=True)

# -------------------- SAVE --------------------

df_icd.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Saved {len(df_icd):,} diagnosis codes to {OUTPUT_PATH}")