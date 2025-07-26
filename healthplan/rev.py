import pandas as pd
import requests
import io

# ---------------------- CONFIG ----------------------
SOURCE_URL = "https://gist.githubusercontent.com/Arithmos-Health/970705a1fc96a5223e0dc9e6921e2f7f/raw/Revenue Codes.csv"
OUTPUT_FILE = "revenue_codes.csv"

# ---------------------- DOWNLOAD & PARSE ----------------------
print("📥 Downloading revenue codes from:", SOURCE_URL)
resp = requests.get(SOURCE_URL)
resp.raise_for_status()

# Read the tab-delimited file
df_raw = pd.read_csv(
    io.StringIO(resp.text),
    sep="\t",
    dtype=str
)

# Normalize column names (lowercase, strip, underscore)
df_raw.columns = [col.strip().lower().replace(" ", "_") for col in df_raw.columns]
print("🧪 Normalized columns:", df_raw.columns)

# Rename columns to match expected names
df_raw = df_raw.rename(columns={
    "revenue_codes": "revenue_code",
    "revenue_code_description": "description"
})

# Drop rows with missing values
df = df_raw.dropna(subset=["revenue_code", "description"])
df = df[df["revenue_code"].str.match(r'^\d{3,4}$')].reset_index(drop=True)

# ---------------------- ADDITIONAL METADATA ----------------------
def categorize_rev(code):
    c = int(code)
    if 100 <= c < 200:
        return "Room-and-board"
    elif 200 <= c < 300:
        return "Ancillary/Equipment"
    elif 300 <= c < 400:
        return "Diagnostic"
    elif 400 <= c < 500:
        return "Emergency/Imaging"
    elif 500 <= c < 600:
        return "Therapies"
    elif 600 <= c < 700:
        return "Supplies"
    elif 700 <= c < 800:
        return "Other Ancillary"
    elif 800 <= c < 900:
        return "Dialysis/Blood"
    elif 900 <= c < 1000:
        return "Clinic"
    else:
        return "Misc"

df["category_group"] = df["revenue_code"].apply(categorize_rev)

# ---------------------- SAVE OUTPUT ----------------------
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(df):,} revenue codes to '{OUTPUT_FILE}'")