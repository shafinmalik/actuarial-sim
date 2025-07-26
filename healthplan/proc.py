import pandas as pd
import requests
import io

# ---------------------- CONFIG ----------------------
SOURCE_URL = (
    "https://gist.githubusercontent.com/"
    "lieldulev/439793dc3c5a6613b661c33d71fdd185/raw/cpt4.csv"
)
OUTPUT_FILE = "procedure_codes.csv"

# ---------------------- DOWNLOAD & PARSE ----------------------
print("📥 Downloading procedure codes from:", SOURCE_URL)
resp = requests.get(SOURCE_URL)
resp.raise_for_status()

# This CSV has no header row: assume first column code, second description
df_raw = pd.read_csv(io.StringIO(resp.text), header=None, names=["procedure_code", "description"], dtype=str)

# Drop rows missing key data or where procedure_code isn't 5 digits
df = df_raw.dropna(subset=["procedure_code", "description"])
df = df[df["procedure_code"].str.match(r"^\d{5}$")].reset_index(drop=True)

# ---------------------- OPTIONAL METADATA ----------------------
# Categorize CPT ranges into broader groups
def categorize_proc(code):
    c = int(code)
    if 99201 <= c <= 99499:
        return "E/M"
    elif 10000 <= c <= 69990:
        return "Surgery"
    elif 70000 <= c <= 79999:
        return "Radiology"
    elif 80000 <= c <= 89398:
        return "Pathology"
    else:
        return "Other"

df["procedure_group"] = df["procedure_code"].apply(categorize_proc)

# ---------------------- OUTPUT ----------------------
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(df):,} procedure codes to '{OUTPUT_FILE}'")