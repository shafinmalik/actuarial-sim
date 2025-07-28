import pandas as pd
import numpy as np
import random
import uuid
from datetime import timedelta
from tqdm.auto import tqdm
import time

# ---------------------- CONFIG ----------------------
SHOW_PROGRESS = True
USE_FIXED_SEED = False  # set True for reproducibility
if USE_FIXED_SEED:
    random.seed(42)
    np.random.seed(42)

MAX_LINES_PER_CLAIM = 5  # hard cap; 5 should be rare
CATASTROPHIC_PROB = 0.0015  # rare catastrophic multiplier trigger
CATASTROPHIC_MULT = 15.0

# Rx co-occurs with a same-month medical claim probability boost
RX_CO_OCCUR_P = 0.35

# ----------- TREND CONFIG -----------
TREND_ANNUAL = 0.025     # +2.5% per year; can be negative
TREND_NOISE_STD = 0.01   # per-claim noise around the trend
TREND_BASE_YEAR = None   # auto: earliest year in member_months

# ---------------------- LOAD SOURCE TABLES ----------------------

def load_tables():
    mm = pd.read_csv("member_months.csv", parse_dates=["birth_date"])
    mm["COV_Month"] = pd.to_datetime(mm["COV_Month"], format="%Y-%m")

    bt = pd.read_csv("bill_type_codes.csv", dtype=str)
    if "category_of_service" not in bt.columns:
        bt["category_of_service"] = "Other"

    pos = pd.read_csv("pos_codes.csv", dtype=str)
    if "category_of_service" not in pos.columns:
        pos["category_of_service"] = "Other"

    rev = pd.read_csv("revenue_codes.csv", dtype=str)
    # optional helper grouping
    if "category_group" not in rev.columns:
        rev["category_group"] = "Misc"

    proc = pd.read_csv("procedure_codes.csv", dtype=str)
    if "procedure_group" not in proc.columns:
        proc["procedure_group"] = "Other"

    icd = pd.read_csv("icd10_master.csv", dtype=str)
    # normalize helper columns
    for df in (bt, pos, rev, proc, icd):
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip()

    return mm, bt, pos, rev, proc, icd

# ---------------------- CATEGORY MAPPINGS ----------------------

CATEGORIES = ["Professional", "Outpatient", "Inpatient", "ER", "SNF", "Rx", "Other"]

REV_GROUPS_BY_CAT = {
    "Professional": {"Diagnostic", "Therapies", "Other Ancillary", "Misc", "Clinic"},
    "Outpatient": {"Diagnostic", "Therapies", "Supplies", "Emergency/Imaging", "Other Ancillary", "Clinic"},
    "ER": {"Emergency/Imaging", "Diagnostic", "Supplies"},
    "Inpatient": {"Room-and-board", "Therapies", "Diagnostic", "Other Ancillary", "Dialysis/Blood"},
    "SNF": {"Room-and-board", "Therapies"},
    "Rx": {"Ancillary/Equipment", "Supplies", "Other Ancillary"},
    "Other": {"Misc", "Other Ancillary"},
}

PROC_GROUPS_BY_CAT = {
    "Professional": ["E/M", "Other"],
    "Outpatient": ["Radiology", "Pathology", "Surgery", "Other"],
    "ER": ["E/M", "Radiology", "Pathology", "Other"],
    "Inpatient": ["Surgery", "Other"],
    "SNF": ["Other"],
    "Rx": ["Other"],
    "Other": ["Other"],
}

# ---------------------- FREQUENCY MODEL ----------------------

def monthly_lambda(row: pd.Series) -> float:
    """Expected total number of claims in a month for a member-month."""
    r = float(row.get("risk_score", 0.5))
    age = int(row.get("age_at_month", 45))
    esrd = int(row.get("ESRD_Flag", 0))
    
    base = 0.10 + 0.45 * r  # strong risk driver
    # age effects
    if age >= 85:
        base *= 1.35
    elif age >= 75:
        base *= 1.25
    elif age >= 65:
        base *= 1.15
    
    # comorbidity bumps
    bumps = 1.0
    for flag, mult in (
        ("Diabetes", 1.08), ("Hypertension", 1.05), ("COPD", 1.10), ("Asthma", 1.05),
        ("Cancer_Skin", 1.05), ("Cancer_Blood", 1.20), ("Cancer_Pancreatic", 1.35), ("Cancer_Other", 1.15),
    ):
        if int(row.get(flag, 0)) == 1:
            bumps *= mult
    if esrd:
        bumps *= 1.50
    
    lam = base * bumps
    # keep within a reasonable band
    return float(np.clip(lam, 0.02, 6.0))

CATEGORY_WEIGHTS_BASE = {
    "Professional": 0.34,
    "Outpatient": 0.22,
    "ER": 0.12,
    "Inpatient": 0.06,
    "SNF": 0.04,
    "Rx": 0.20,
    "Other": 0.02,
}

def category_weights(row: pd.Series) -> np.ndarray:
    w = CATEGORY_WEIGHTS_BASE.copy()
    # ESRD drives dialysis (Rx + Outpatient + Inpatient dialysis related)
    if int(row.get("ESRD_Flag", 0)) == 1:
        w["Rx"] += 0.06
        w["Outpatient"] += 0.05
        w["Inpatient"] += 0.02
    # cancers drive outpatient / inpatient / Rx
    cancer_flags = [int(row.get(c, 0)) for c in ("Cancer_Blood", "Cancer_Pancreatic", "Cancer_Other", "Cancer_Skin")]
    if any(cancer_flags):
        w["Outpatient"] += 0.05
        w["Inpatient"] += 0.03
        w["Rx"] += 0.04
    # older folks more SNF
    if int(row.get("age_at_month", 45)) >= 75:
        w["SNF"] += 0.03
    # normalize
    arr = np.array([w[c] for c in CATEGORIES], dtype=float)
    arr = np.maximum(arr, 0.001)
    arr /= arr.sum()
    return arr

# ---------------------- SEVERITY MODEL ----------------------

SEVERITY_PARAMS = {
    # (shape k, scale theta) for Gamma; multiplied by risk & condition factors
    "Professional": (2.0, 120.0),
    "Outpatient": (2.2, 380.0),
    "ER": (2.5, 520.0),
    "Inpatient": (3.0, 3200.0),
    "SNF": (2.8, 2200.0),  # LOS factor added
    "Rx": (1.6, 40.0),
    "Other": (1.8, 150.0),
}

def condition_multiplier(row: pd.Series, category: str) -> float:
    m = 1.0
    if int(row.get("ESRD_Flag", 0)) == 1:
        m *= 1.6 if category in ("Outpatient", "Inpatient") else 1.3
    # cancers
    if int(row.get("Cancer_Pancreatic", 0)) == 1:
        m *= 2.2
    if int(row.get("Cancer_Blood", 0)) == 1:
        m *= 1.8
    if int(row.get("Cancer_Other", 0)) == 1:
        m *= 1.4
    if int(row.get("Cancer_Skin", 0)) == 1:
        m *= 1.15
    # respiratory
    if int(row.get("COPD", 0)) == 1:
        m *= 1.25 if category in ("ER", "Inpatient") else 1.10
    if int(row.get("Asthma", 0)) == 1:
        m *= 1.15 if category in ("ER", "Outpatient") else 1.05
    # metabolic / HTN small effect
    if int(row.get("Diabetes", 0)) == 1:
        m *= 1.10
    if int(row.get("Hypertension", 0)) == 1:
        m *= 1.05
    return m

def sample_claim_amount(row: pd.Series, category: str, los_days: int) -> float:
    k, theta = SEVERITY_PARAMS[category]
    risk = float(row.get("risk_score", 0.5))
    mult = (1.0 + 0.55 * risk) * condition_multiplier(row, category)
    base = np.random.gamma(k, theta) * mult
    if category == "SNF":
        # per-diem style uplift
        per_diem = 600.0 * (1.0 + 0.4 * risk)
        base += per_diem * max(los_days, 1)
    if category == "Inpatient":
        base *= (1.0 + 0.08 * max(los_days - 3, 0))
    if np.random.rand() < CATASTROPHIC_PROB:
        base *= CATASTROPHIC_MULT
    return float(max(base, 10.0))

# ---------------------- DATES ----------------------

def sample_service_dates(cov_month: pd.Timestamp, category: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = cov_month + pd.to_timedelta(random.randint(0, 27), unit="D")
    if category == "Inpatient":
        los = random.randint(3, 15)  # 2-midnight rule -> LOS >= 3 calendar days
    elif category == "SNF":
        los = int(np.clip(int(np.random.gamma(2.2, 4.0)), 3, 45))
    elif category == "Rx":
        los = 0
    else:
        los = random.randint(0, 2)
    end = start + pd.to_timedelta(los, unit="D")
    return start, end

def sample_paid_lag_days(category: str) -> int:
    if category == "Rx":
        return 0
    if category == "Professional":
        return int(np.clip(np.random.gamma(2.0, 8.0), 5, 60))
    if category == "Outpatient":
        return int(np.clip(np.random.gamma(2.2, 10.0), 10, 75))
    if category == "ER":
        return int(np.clip(np.random.gamma(2.0, 9.0), 7, 60))
    if category == "Inpatient":
        return int(np.clip(np.random.gamma(2.5, 18.0), 20, 120))
    if category == "SNF":
        return int(np.clip(np.random.gamma(2.5, 20.0), 25, 120))
    return int(np.clip(np.random.gamma(2.0, 12.0), 7, 90))

# ---------------------- TREND HELPERS ----------------------

def set_trend_base_year_from_mm(mm: pd.DataFrame):
    """Set TREND_BASE_YEAR to the earliest COV_Month year in member_months."""
    global TREND_BASE_YEAR
    TREND_BASE_YEAR = int(mm["COV_Month"].dt.year.min())

def trend_factor_for_date(dt: pd.Timestamp) -> float:
    """
    Compound trend from TREND_BASE_YEAR to the date 'dt' with small zero-mean noise.
    Works with negative TREND_ANNUAL too.
    """
    assert TREND_BASE_YEAR is not None, "TREND_BASE_YEAR must be set before calling trend_factor_for_date()."
    # fractional years since base
    years = (dt.year - TREND_BASE_YEAR) + ((dt.dayofyear - 1) / 365.0)
    base_factor = (1.0 + TREND_ANNUAL) ** years

    # noise centered at 1.0; magnitude scales gently with |years|
    sigma = TREND_NOISE_STD * max(1.0, abs(years))
    noise = 1.0 + np.random.normal(0.0, sigma)
    noise = max(0.7, noise)  # conservative floor
    return base_factor * noise

# ---------------------- CODE SELECTION ----------------------

def sample_pos(pos_df: pd.DataFrame, category: str) -> str:
    cand = pos_df[pos_df["category_of_service"].str.lower() == category.lower()]
    if cand.empty:
        cand = pos_df
    return str(cand.sample(1).iloc[0]["pos_code"])

def sample_billtype(bt_df: pd.DataFrame, category: str) -> str:
    cand = bt_df[bt_df["category_of_service"].str.lower() == category.lower()]
    if cand.empty:
        cand = bt_df
    return str(cand.sample(1).iloc[0]["bill_type_code"])

def sample_rev(rev_df: pd.DataFrame, category: str) -> str:
    groups = REV_GROUPS_BY_CAT.get(category, {"Misc"})
    cand = rev_df[rev_df["category_group"].isin(groups)]
    if category == "ER":
        # prefer explicit ER codes if present
        er_like = rev_df[rev_df["description"].str.contains("Emergency", case=False, na=False)]
        if not er_like.empty:
            cand = er_like
    if category == "Rx":
        pharm = rev_df[rev_df["description"].str.contains("Pharmacy|Drug", case=False, na=False)]
        if not pharm.empty:
            cand = pharm
    if category in ("Inpatient", "Outpatient", "SNF"):
        dial = rev_df[rev_df["description"].str.contains("Dialysis", case=False, na=False)]
        # mix a bit of dialysis into eligible categories; choice happens later in diag logic
        if not dial.empty and np.random.rand() < 0.15:
            cand = dial
    if cand.empty:
        cand = rev_df
    return str(cand.sample(1).iloc[0]["revenue_code"])

def sample_proc(proc_df: pd.DataFrame, category: str) -> str:
    groups = PROC_GROUPS_BY_CAT.get(category, ["Other"])
    cand = proc_df[proc_df["procedure_group"].isin(groups)]
    if category == "ER":
        # prefer E/M when available
        em = proc_df[proc_df["procedure_group"] == "E/M"]
        if not em.empty and np.random.rand() < 0.6:
            cand = em
    if cand.empty:
        cand = proc_df
    return str(cand.sample(1).iloc[0]["procedure_code"])

# ---------------------- DIAGNOSIS SELECTION ----------------------

# Helper keyword filters

def _pick_icd(icd: pd.DataFrame, pattern: str) -> pd.DataFrame:
    return icd[icd["long_desc"].str.contains(pattern, case=False, na=False) | icd["short_desc"].str.contains(pattern, case=False, na=False)]

def sample_diagnosis(icd: pd.DataFrame, row: pd.Series, category: str) -> str:
    # strong signals first
    if int(row.get("ESRD_Flag", 0)) == 1:
        esrd = icd[icd["full_code"].str.upper() == "N18.6"]
        if not esrd.empty and np.random.rand() < 0.85:
            return str(esrd.sample(1).iloc[0]["full_code"])
        dial = _pick_icd(icd, "dialysis|end stage renal|ESRD|N18")
        if not dial.empty:
            return str(dial.sample(1).iloc[0]["full_code"])
    # CKD (non-ESRD)
    if "CKD_Status" in row and "ESRD" not in str(row["CKD_Status"]).upper():
        ckd = icd[icd["full_code"].str.startswith("N18")]
        if not ckd.empty and np.random.rand() < 0.7:
            return str(ckd.sample(1).iloc[0]["full_code"])
    # cancers priorities
    if int(row.get("Cancer_Pancreatic", 0)) == 1:
        pan = icd[icd["full_code"].str.startswith("C25")]
        if not pan.empty and np.random.rand() < 0.8:
            return str(pan.sample(1).iloc[0]["full_code"])
    if int(row.get("Cancer_Blood", 0)) == 1:
        blood = icd[icd["full_code"].str.match(r"^C9[0-6].*")]
        if not blood.empty and np.random.rand() < 0.75:
            return str(blood.sample(1).iloc[0]["full_code"])
    if int(row.get("Cancer_Skin", 0)) == 1:
        skin = icd[icd["full_code"].str.startswith("C44")]
        if not skin.empty and np.random.rand() < 0.7:
            return str(skin.sample(1).iloc[0]["full_code"])
    if int(row.get("Cancer_Other", 0)) == 1:
        neoplasm = _pick_icd(icd, "malignant neoplasm|cancer")
        if not neoplasm.empty and np.random.rand() < 0.7:
            return str(neoplasm.sample(1).iloc[0]["full_code"])
    # respiratory
    if int(row.get("COPD", 0)) == 1 and np.random.rand() < 0.6:
        copd = icd[icd["full_code"].str.startswith("J44")]
        if not copd.empty:
            return str(copd.sample(1).iloc[0]["full_code"])
    if int(row.get("Asthma", 0)) == 1 and np.random.rand() < 0.6:
        asth = icd[icd["full_code"].str.startswith("J45")]
        if not asth.empty:
            return str(asth.sample(1).iloc[0]["full_code"])
    # metabolic / HTN
    if int(row.get("Diabetes", 0)) == 1 and np.random.rand() < 0.6:
        dia = icd[icd["full_code"].str.startswith("E11")]
        if not dia.empty:
            return str(dia.sample(1).iloc[0]["full_code"])
    if int(row.get("Hypertension", 0)) == 1 and np.random.rand() < 0.6:
        htn = icd[icd["full_code"].str.upper() == "I10"]
        if not htn.empty:
            return str(htn.sample(1).iloc[0]["full_code"])
    # category-driven fallback (e.g., Z00 for preventive)
    if category == "Professional" and np.random.rand() < 0.25:
        z00 = icd[icd["full_code"].str.startswith("Z00")]
        if not z00.empty:
            return str(z00.sample(1).iloc[0]["full_code"])
    # pure fallback
    return str(icd.sample(1).iloc[0]["full_code"])

# ---------------------- CLAIM LINE BUILDER ----------------------

def build_claim_lines(row: pd.Series, category: str, bt_df: pd.DataFrame, pos_df: pd.DataFrame,
                       rev_df: pd.DataFrame, proc_df: pd.DataFrame, icd_df: pd.DataFrame) -> list[dict]:
    start_dt, end_dt = sample_service_dates(row["COV_Month"], category)
    # LOS used for severity
    los_days = (end_dt - start_dt).days
    total_allowed = sample_claim_amount(row, category, los_days)

    total_allowed *= trend_factor_for_date(start_dt)

    # split across 1..MAX_LINES_PER_CLAIM (geometric-like)
    # P(n lines) ~ (0.65, 0.22, 0.09, 0.03, 0.01)
    choices = [1, 2, 3, 4, 5]
    probs = [0.65, 0.22, 0.09, 0.03, 0.01]
    n_lines = int(np.random.choice(choices, p=probs))

    # create base splits
    parts = np.random.dirichlet(np.ones(n_lines)) * total_allowed

    claim_id = str(uuid.uuid4())
    base_paid_lag = sample_paid_lag_days(category)

    # choose diagnosis once per claim
    diag_code = sample_diagnosis(icd_df, row, category)

    lines = []
    earliest_paid = None
    for i in range(n_lines):
        allowed = float(np.round(parts[i] * np.random.uniform(0.9, 1.1), 2))
        # small per-line paid variation; allow small adjustments including negative on a line
        paid_core = allowed * np.random.uniform(0.82, 0.98)
        adj = np.random.normal(0, allowed * 0.02)  # small adj
        paid = float(np.round(paid_core + adj, 2))

        pos_code = sample_pos(pos_df, category)
        bill_type = sample_billtype(bt_df, category)
        revenue_code = sample_rev(rev_df, category)
        proc_code = sample_proc(proc_df, category)

        paid_date = start_dt + timedelta(days=int(max(0, base_paid_lag + np.random.normal(0, 4))))
        if earliest_paid is None or paid_date < earliest_paid:
            earliest_paid = paid_date

        line = {
            "claim_id": claim_id,
            "line_number": i + 1,
            "member_id": row["member_id"],
            "category_of_service": category,
            "service_start_date": start_dt.date(),
            "service_end_date": end_dt.date(),
            "paid_date": paid_date.date(),
            "bill_type_code": bill_type,
            "pos_code": pos_code,
            "revenue_code": revenue_code,
            "procedure_code": proc_code,
            "diagnosis_code": diag_code,
            "allowed_amount": round(allowed, 2),
            "paid_amount": round(paid, 2),
            "los_days": los_days,
        }
        if category == "Rx":
            line["rx_flag"] = 1
        lines.append(line)

    # ensure total across lines is non-negative
    tot_paid = sum(l["paid_amount"] for l in lines)
    if tot_paid < 0:
        diff = -tot_paid
        # bump the first line to cover negatives
        lines[0]["paid_amount"] = round(lines[0]["paid_amount"] + diff + 0.01, 2)

    # sort by paid_date so first line is earliest
    lines.sort(key=lambda x: x["paid_date"])
    return lines

# ---------------------- PER-MONTH GENERATION ----------------------

def generate_claims_for_month(row: pd.Series, bt_df, pos_df, rev_df, proc_df, icd_df) -> list[dict]:
    lam = monthly_lambda(row)
    total_claims = int(np.random.poisson(lam))
    if total_claims == 0:
        # occasionally force a preventive professional visit for very low risk
        if np.random.rand() < 0.03:
            total_claims = 1
        else:
            return []

    weights = category_weights(row)
    categories = np.random.choice(CATEGORIES, size=total_claims, p=weights)

    # Optionally add an Rx claim co-occurring with a medical claim
    if ("Rx" not in categories) and any(c in ("Professional", "Outpatient", "ER", "Inpatient") for c in categories):
        if np.random.rand() < RX_CO_OCCUR_P:
            categories = np.append(categories, "Rx")

    lines = []
    for cat in categories:
        lines.extend(build_claim_lines(row, cat, bt_df, pos_df, rev_df, proc_df, icd_df))
    return lines

# ---------------------- MAIN ----------------------

def main():
    mm, bt, pos, rev, proc, icd = load_tables()
    set_trend_base_year_from_mm(mm)
    print(f"ℹ️ Trend base year set to {TREND_BASE_YEAR} (annual={TREND_ANNUAL:.3%}, noise_std={TREND_NOISE_STD:.2%})")

    # We will only emit claims for member_ids present in member_months
    all_lines: list[dict] = []

    # Convert to dicts for faster iteration and clean tqdm display
    rows = mm.to_dict("records")

    start_time = time.time()
    iterable = tqdm(rows, total=len(rows), desc="Generating claims", unit="mm") if SHOW_PROGRESS else rows

    for row in iterable:
        all_lines.extend(generate_claims_for_month(row, bt, pos, rev, proc, icd))

    elapsed = time.time() - start_time
    print(f"⏱️ Generation finished in {elapsed:,.1f}s")

    df_claims = pd.DataFrame(all_lines)
    # Order columns
    cols = [
        "claim_id", "line_number", "member_id", "category_of_service",
        "service_start_date", "service_end_date", "paid_date",
        "bill_type_code", "pos_code", "revenue_code", "procedure_code", "diagnosis_code",
        "allowed_amount", "paid_amount", "los_days"
    ]
    extra = [c for c in df_claims.columns if c not in cols]
    df_claims = df_claims[cols + extra]

    df_claims.to_csv("claims.csv", index=False)
    print(f"✅ Wrote {len(df_claims):,} claim lines to claims.csv")

if __name__ == "__main__":
    main()
