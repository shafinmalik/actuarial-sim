import pandas as pd
import numpy as np
import uuid
import time
from datetime import timedelta
from tqdm.auto import tqdm

# ============================ CONFIG ============================

USE_FIXED_SEED = False            # Set True for reproducible runs
SHOW_PROGRESS  = True             # Progress bar

MAX_LINES_PER_CLAIM = 5
CATASTROPHIC_PROB   = 0.0015
CATASTROPHIC_MULT   = 15.0
RX_CO_OCCUR_P       = 0.35        # chance to add Rx when a medical claim exists

# ---- Trend (inflation/deflation) ----
TREND_ANNUAL    = 0.025           # +2.5% per year; can be negative
TREND_NOISE_STD = 0.01            # per-claim noise around the trend
TREND_BASE_YEAR = None            # will be set from member_months earliest year

# ----- Line distribution & adjustments -----
PRIMARY_LINE_WEIGHT = 4.0    # >1 biases line 1 to carry more allowed than others
ALLOW_NEGATIVE_SUBSEQUENT = True
NEG_SUBSEQ_MIN_FACTOR = -0.25  # allow line i>0 paid as low as -25% of its allowed
FIRST_LINE_MIN_PAID = 0.0      # keep line 1 paid non-negative

# Categories
CATEGORIES = ["Professional", "Outpatient", "Inpatient", "ER", "SNF", "Rx", "Other"]

# Revenue code grouping by category (broad)
REV_GROUPS_BY_CAT = {
    "Professional": {"Diagnostic", "Therapies", "Other Ancillary", "Misc", "Clinic"},
    "Outpatient":   {"Diagnostic", "Therapies", "Supplies", "Emergency/Imaging", "Other Ancillary", "Clinic"},
    "ER":           {"Emergency/Imaging", "Diagnostic", "Supplies"},
    "Inpatient":    {"Room-and-board", "Therapies", "Diagnostic", "Other Ancillary", "Dialysis/Blood"},
    "SNF":          {"Room-and-board", "Therapies"},
    "Rx":           {"Ancillary/Equipment", "Supplies", "Other Ancillary"},
    "Other":        {"Misc", "Other Ancillary"},
}

# Procedure group preferences by category
PROC_GROUPS_BY_CAT = {
    "Professional": ["E/M", "Other"],
    "Outpatient":   ["Radiology", "Pathology", "Surgery", "Other"],
    "ER":           ["E/M", "Radiology", "Pathology", "Other"],
    "Inpatient":    ["Surgery", "Other"],
    "SNF":          ["Other"],
    "Rx":           ["Other"],
    "Other":        ["Other"],
}

# Frequency base mix; adjusted per member
CATEGORY_WEIGHTS_BASE = {
    "Professional": 0.34,
    "Outpatient":   0.22,
    "ER":           0.12,
    "Inpatient":    0.06,
    "SNF":          0.04,
    "Rx":           0.20,
    "Other":        0.02,
}

# Severity params: Gamma(k, theta) per category
SEVERITY_PARAMS = {
    "Professional": (2.0,  120.0),
    "Outpatient":   (2.2,  380.0),
    "ER":           (2.5,  520.0),
    "Inpatient":    (3.0, 3200.0),
    "SNF":          (2.8, 2200.0),
    "Rx":           (1.6,   40.0),
    "Other":        (1.8,  150.0),
}

# ============================ RNG ============================

rng = np.random.default_rng(42 if USE_FIXED_SEED else None)

# ============================ LOAD TABLES ============================

def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df

def load_tables():
    mm = pd.read_csv("member_months.csv", parse_dates=["birth_date"])
    mm["COV_Month"] = pd.to_datetime(mm["COV_Month"], format="%Y-%m")
    _strip_strings(mm)

    bt = pd.read_csv("bill_type_codes.csv", dtype=str)
    _strip_strings(bt)
    if "category_of_service" not in bt.columns:
        bt["category_of_service"] = "Other"

    pos = pd.read_csv("pos_codes.csv", dtype=str)
    _strip_strings(pos)
    if "category_of_service" not in pos.columns:
        pos["category_of_service"] = "Other"

    rev = pd.read_csv("revenue_codes.csv", dtype=str)
    _strip_strings(rev)
    if "category_group" not in rev.columns:
        rev["category_group"] = "Misc"
    # ensure 'description' is present
    if "description" not in rev.columns:
        rev["description"] = ""

    proc = pd.read_csv("procedure_codes.csv", dtype=str)
    _strip_strings(proc)
    if "procedure_group" not in proc.columns:
        proc["procedure_group"] = "Other"

    icd = pd.read_csv("icd10_master.csv", dtype=str)
    _strip_strings(icd)
    # normalize expected columns
    if "full_code" not in icd.columns and "diagnosis_code" in icd.columns:
        icd["full_code"] = icd["diagnosis_code"]
    if "short_desc" not in icd.columns:
        icd["short_desc"] = ""
    if "long_desc" not in icd.columns:
        icd["long_desc"] = ""

    return mm, bt, pos, rev, proc, icd

# ============================ POOLS (FAST PICKERS) ============================

def _pool(series: pd.Series) -> np.ndarray:
    return series.astype(str).to_numpy()

def build_pools(bt, pos, rev, proc, icd):
    # POS
    pos_by_cat = {}
    for c in CATEGORIES:
        arr = _pool(pos.loc[pos["category_of_service"].str.lower() == c.lower(), "pos_code"])
        if arr.size == 0:
            arr = _pool(pos["pos_code"])
        pos_by_cat[c] = arr
    pos_all = _pool(pos["pos_code"])

    # Bill Type
    bt_by_cat = {}
    if "bill_type_code" in bt.columns:
        for c in CATEGORIES:
            arr = _pool(bt.loc[bt["category_of_service"].str.lower() == c.lower(), "bill_type_code"])
            if arr.size == 0:
                arr = _pool(bt["bill_type_code"])
            bt_by_cat[c] = arr
        bt_all = _pool(bt["bill_type_code"])
    else:
        # Safe empty pools if file lacks bill_type_code column
        for c in CATEGORIES:
            bt_by_cat[c] = np.array([], dtype=str)
        bt_all = np.array([], dtype=str)

    # Revenue code pools by group and quick special arrays
    rev_by_group = {}
    for grp in rev["category_group"].unique():
        rev_by_group[grp] = _pool(rev.loc[rev["category_group"] == grp, "revenue_code"])
    rev_all = _pool(rev["revenue_code"])

    rev_er = _pool(rev.loc[rev["description"].str.contains("Emergency", case=False, na=False), "revenue_code"])
    rev_pharm = _pool(rev.loc[rev["description"].str.contains("Pharmacy|Drug", case=False, na=False), "revenue_code"])
    rev_dialysis = _pool(rev.loc[rev["description"].str.contains("Dialysis", case=False, na=False), "revenue_code"])

    # Procedure by group
    proc_by_group = {}
    for grp in proc["procedure_group"].unique():
        proc_by_group[grp] = _pool(proc.loc[proc["procedure_group"] == grp, "procedure_code"])
    proc_all = _pool(proc["procedure_code"])

    # ICD pools
    icd_all  = _pool(icd["full_code"])
    icd_esrd = _pool(icd.loc[icd["full_code"].str.upper() == "N18.6", "full_code"])
    icd_ckd  = _pool(icd.loc[icd["full_code"].str.startswith("N18"), "full_code"])
    icd_copd = _pool(icd.loc[icd["full_code"].str.startswith("J44"), "full_code"])
    icd_asth = _pool(icd.loc[icd["full_code"].str.startswith("J45"), "full_code"])
    icd_dm2  = _pool(icd.loc[icd["full_code"].str.startswith("E11"), "full_code"])
    icd_htn  = _pool(icd.loc[icd["full_code"].str.upper() == "I10", "full_code"])
    icd_z00  = _pool(icd.loc[icd["full_code"].str.startswith("Z00"), "full_code"])
    icd_c25  = _pool(icd.loc[icd["full_code"].str.startswith("C25"), "full_code"])
    icd_c9x  = _pool(icd.loc[icd["full_code"].str.match(r"^C9[0-6]"), "full_code"])
    icd_c44  = _pool(icd.loc[icd["full_code"].str.startswith("C44"), "full_code"])
    icd_neop = _pool(icd.loc[
        icd["long_desc"].str.contains("malignant neoplasm|cancer", case=False, na=False)
        | icd["short_desc"].str.contains("malignant neoplasm|cancer", case=False, na=False),
        "full_code"
    ])

    return {
        "pos_by_cat": pos_by_cat, "pos_all": pos_all,
        "bt_by_cat": bt_by_cat, "bt_all": bt_all,
        "rev_by_group": rev_by_group, "rev_all": rev_all,
        "rev_er": rev_er, "rev_pharm": rev_pharm, "rev_dialysis": rev_dialysis,
        "proc_by_group": proc_by_group, "proc_all": proc_all,
        "icd": {
            "all": icd_all, "esrd": icd_esrd, "ckd": icd_ckd,
            "copd": icd_copd, "asth": icd_asth, "dm2": icd_dm2, "htn": icd_htn, "z00": icd_z00,
            "c25": icd_c25, "c9x": icd_c9x, "c44": icd_c44, "neop": icd_neop
        }
    }

# ============================ FREQUENCY ============================

def monthly_lambda(row) -> float:
    r   = float(row.get("risk_score", 0.5))
    age = int(row.get("age_at_month", 45))
    esrd = int(row.get("ESRD_Flag", 0))

    base = 0.10 + 0.45 * r
    if age >= 85: base *= 1.35
    elif age >= 75: base *= 1.25
    elif age >= 65: base *= 1.15

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
    return float(np.clip(lam, 0.02, 6.0))

def category_weights(row) -> np.ndarray:
    w = CATEGORY_WEIGHTS_BASE.copy()
    if int(row.get("ESRD_Flag", 0)) == 1:
        w["Rx"]        += 0.06
        w["Outpatient"]+= 0.05
        w["Inpatient"] += 0.02
    cancer_flags = [int(row.get(c, 0)) for c in ("Cancer_Blood", "Cancer_Pancreatic", "Cancer_Other", "Cancer_Skin")]
    if any(cancer_flags):
        w["Outpatient"] += 0.05
        w["Inpatient"]  += 0.03
        w["Rx"]         += 0.04
    if int(row.get("age_at_month", 45)) >= 75:
        w["SNF"] += 0.03

    arr = np.array([w[c] for c in CATEGORIES], dtype=float)
    arr = np.maximum(arr, 0.001)
    arr /= arr.sum()
    return arr

# ============================ SEVERITY ============================

def condition_multiplier(row, category: str) -> float:
    m = 1.0
    if int(row.get("ESRD_Flag", 0)) == 1:
        m *= 1.6 if category in ("Outpatient", "Inpatient") else 1.3
    if int(row.get("Cancer_Pancreatic", 0)) == 1: m *= 2.2
    if int(row.get("Cancer_Blood", 0)) == 1:      m *= 1.8
    if int(row.get("Cancer_Other", 0)) == 1:      m *= 1.4
    if int(row.get("Cancer_Skin", 0)) == 1:       m *= 1.15
    if int(row.get("COPD", 0)) == 1:              m *= 1.25 if category in ("ER", "Inpatient") else 1.10
    if int(row.get("Asthma", 0)) == 1:            m *= 1.15 if category in ("ER", "Outpatient") else 1.05
    if int(row.get("Diabetes", 0)) == 1:          m *= 1.10
    if int(row.get("Hypertension", 0)) == 1:      m *= 1.05
    return m

def sample_claim_amount(row, category: str, los_days: int) -> float:
    k, theta = SEVERITY_PARAMS[category]
    risk = float(row.get("risk_score", 0.5))
    mult = (1.0 + 0.55 * risk) * condition_multiplier(row, category)
    base = rng.gamma(k, theta) * mult

    if category == "SNF":
        per_diem = 600.0 * (1.0 + 0.4 * risk)
        base += per_diem * max(los_days, 1)
    if category == "Inpatient":
        base *= (1.0 + 0.08 * max(los_days - 3, 0))
    if rng.random() < CATASTROPHIC_PROB:
        base *= CATASTROPHIC_MULT
    return float(max(base, 10.0))

# ============================ DATES ============================

def sample_service_dates(cov_month: pd.Timestamp, category: str):
    start = cov_month + pd.to_timedelta(rng.integers(0, 28), unit="D")
    if category == "Inpatient":
        los = rng.integers(3, 16)  # inclusive high is exclusive in numpy
    elif category == "SNF":
        los = int(np.clip(int(rng.gamma(2.2, 4.0)), 3, 45))
    elif category == "Rx":
        los = 0
    else:
        los = rng.integers(0, 3)
    end = start + pd.to_timedelta(los, unit="D")
    return start, end

def sample_paid_lag_days(category: str) -> int:
    if category == "Rx":
        return 0
    if category == "Professional":
        return int(np.clip(rng.gamma(2.0, 8.0), 5, 60))
    if category == "Outpatient":
        return int(np.clip(rng.gamma(2.2, 10.0), 10, 75))
    if category == "ER":
        return int(np.clip(rng.gamma(2.0, 9.0), 7, 60))
    if category == "Inpatient":
        return int(np.clip(rng.gamma(2.5, 18.0), 20, 120))
    if category == "SNF":
        return int(np.clip(rng.gamma(2.5, 20.0), 25, 120))
    return int(np.clip(rng.gamma(2.0, 12.0), 7, 90))

# ============================ TREND HELPERS ============================

def set_trend_base_year_from_mm(mm: pd.DataFrame):
    global TREND_BASE_YEAR
    TREND_BASE_YEAR = int(mm["COV_Month"].dt.year.min())

def trend_factor_for_date(dt: pd.Timestamp) -> float:
    assert TREND_BASE_YEAR is not None, "TREND_BASE_YEAR must be set first."
    years = (dt.year - TREND_BASE_YEAR) + ((dt.dayofyear - 1) / 365.0)
    base_factor = (1.0 + TREND_ANNUAL) ** years
    sigma = TREND_NOISE_STD * max(1.0, abs(years))
    noise = 1.0 + rng.normal(0.0, sigma)
    noise = max(0.7, noise)
    return base_factor * noise

# ============================ FAST PICKERS ============================

def pick_pos(category, pools):
    arr = pools["pos_by_cat"].get(category)
    if arr is None or arr.size == 0:
        arr = pools["pos_all"]
    return arr[rng.integers(arr.size)]

def pick_billtype(category, pools):
    arr = pools["bt_by_cat"].get(category)
    if arr is None or arr.size == 0:
        arr = pools["bt_all"]
    if arr.size == 0:
        return ""  # safe fallback
    return arr[rng.integers(arr.size)]

def pick_rev(category, pools):
    # base groups
    groups = REV_GROUPS_BY_CAT.get(category, {"Misc"})
    candidates = []
    for g in groups:
        if g in pools["rev_by_group"]:
            candidates.append(pools["rev_by_group"][g])
    if candidates:
        cand = np.concatenate(candidates)
    else:
        cand = pools["rev_all"]

    # category-specific overrides
    if category == "ER" and pools["rev_er"].size and rng.random() < 0.6:
        cand = pools["rev_er"]
    if category == "Rx" and pools["rev_pharm"].size and rng.random() < 0.8:
        cand = pools["rev_pharm"]
    if category in ("Inpatient", "Outpatient", "SNF") and pools["rev_dialysis"].size and rng.random() < 0.15:
        cand = pools["rev_dialysis"]

    if cand.size == 0:
        cand = pools["rev_all"]
    return cand[rng.integers(cand.size)]

def pick_proc(category, pools):
    groups = PROC_GROUPS_BY_CAT.get(category, ["Other"])
    cands = []
    for g in groups:
        arr = pools["proc_by_group"].get(g)
        if arr is not None and arr.size:
            cands.append(arr)
    if cands:
        cand = np.concatenate(cands)
    else:
        cand = pools["proc_all"]

    if category == "ER":
        em = pools["proc_by_group"].get("E/M", np.array([], dtype=str))
        if em.size and rng.random() < 0.6:
            cand = em
    return cand[rng.integers(cand.size)]

def pick_icd(row, category, pools):
    icd = pools["icd"]
    if int(row.get("ESRD_Flag", 0)) == 1 and icd["esrd"].size and rng.random() < 0.85:
        return icd["esrd"][rng.integers(icd["esrd"].size)]
    if "CKD_Status" in row and "ESRD" not in str(row["CKD_Status"]).upper() and icd["ckd"].size and rng.random() < 0.7:
        return icd["ckd"][rng.integers(icd["ckd"].size)]
    if int(row.get("Cancer_Pancreatic", 0)) == 1 and icd["c25"].size and rng.random() < 0.8:
        return icd["c25"][rng.integers(icd["c25"].size)]
    if int(row.get("Cancer_Blood", 0)) == 1 and icd["c9x"].size and rng.random() < 0.75:
        return icd["c9x"][rng.integers(icd["c9x"].size)]
    if int(row.get("Cancer_Skin", 0)) == 1 and icd["c44"].size and rng.random() < 0.7:
        return icd["c44"][rng.integers(icd["c44"].size)]
    if int(row.get("Cancer_Other", 0)) == 1 and icd["neop"].size and rng.random() < 0.7:
        return icd["neop"][rng.integers(icd["neop"].size)]
    if int(row.get("COPD", 0)) == 1 and icd["copd"].size and rng.random() < 0.6:
        return icd["copd"][rng.integers(icd["copd"].size)]
    if int(row.get("Asthma", 0)) == 1 and icd["asth"].size and rng.random() < 0.6:
        return icd["asth"][rng.integers(icd["asth"].size)]
    if int(row.get("Diabetes", 0)) == 1 and icd["dm2"].size and rng.random() < 0.6:
        return icd["dm2"][rng.integers(icd["dm2"].size)]
    if int(row.get("Hypertension", 0)) == 1 and icd["htn"].size and rng.random() < 0.6:
        return icd["htn"][rng.integers(icd["htn"].size)]
    if category == "Professional" and icd["z00"].size and rng.random() < 0.25:
        return icd["z00"][rng.integers(icd["z00"].size)]
    return icd["all"][rng.integers(icd["all"].size)]

# ============================ CLAIM BUILD (COLUMNAR) ============================

def build_claim_lines(row, category, pools, cols):
    start_dt, end_dt = sample_service_dates(row["COV_Month"], category)
    los_days = (end_dt - start_dt).days
    total_allowed = sample_claim_amount(row, category, los_days)
    # Apply trend
    total_allowed *= trend_factor_for_date(start_dt)

    # number of lines
    choices = np.array([1, 2, 3, 4, 5])
    probs   = np.array([0.65, 0.22, 0.09, 0.03, 0.01])
    n_lines = int(rng.choice(choices, p=probs))

    alpha = np.ones(n_lines, dtype=float)
    alpha[0] = PRIMARY_LINE_WEIGHT
    parts = rng.dirichlet(alpha) * total_allowed
    claim_id = str(uuid.uuid4())
    base_lag = sample_paid_lag_days(category)

    # one diagnosis per claim
    diag_code = pick_icd(row, category, pools)

    paid_dates = []
    allowed_lines = []
    paid_lines = []

    for i in range(n_lines):
        allowed = float(np.round(parts[i] * rng.uniform(0.9, 1.1), 2))
        paid_core = allowed * rng.uniform(0.82, 0.98)
        adj = rng.normal(0.0, allowed * 0.02)
        paid = float(np.round(paid_core + adj, 2))

        # Clamp paid by line index rules:
        if i == 0:
            # first line should not be negative
            if paid < FIRST_LINE_MIN_PAID:
                paid = FIRST_LINE_MIN_PAID
        else:
            # subsequent lines may go modestly negative if allowed
            if ALLOW_NEGATIVE_SUBSEQUENT:
                min_paid = NEG_SUBSEQ_MIN_FACTOR * allowed
                if paid < min_paid:
                    paid = float(np.round(min_paid, 2))
            else:
                if paid < 0.0:
                    paid = 0.0

        pos_code = pick_pos(category, pools)
        billtype = pick_billtype(category, pools)
        rev_code = pick_rev(category, pools)
        proc_code= pick_proc(category, pools)

        lag_days = int(max(0, base_lag + rng.normal(0, 4)))
        paid_date = start_dt + timedelta(days=lag_days)

        # Append columns
        cols["claim_id"].append(claim_id)
        cols["line_number"].append(i + 1)
        cols["member_id"].append(row["member_id"])
        cols["category_of_service"].append(category)
        cols["service_start_date"].append(start_dt.date())
        cols["service_end_date"].append(end_dt.date())
        cols["paid_date"].append(paid_date.date())
        cols["bill_type_code"].append(billtype)
        cols["pos_code"].append(pos_code)
        cols["revenue_code"].append(rev_code)
        cols["procedure_code"].append(proc_code)
        cols["diagnosis_code"].append(diag_code)
        cols["allowed_amount"].append(round(allowed, 2))
        cols["paid_amount"].append(round(paid, 2))
        cols["los_days"].append(los_days)

        paid_dates.append(paid_date)
        allowed_lines.append(allowed)
        paid_lines.append(paid)

    # ensure claim total non-negative
    if sum(paid_lines) < 0:
        diff = -sum(paid_lines) + 0.01
        cols["paid_amount"][-n_lines] = round(cols["paid_amount"][-n_lines] + diff, 2)

def generate_claims_for_month(row, pools, cols):
    lam = monthly_lambda(row)
    total_claims = int(rng.poisson(lam))
    if total_claims == 0:
        if rng.random() < 0.03:
            total_claims = 1
        else:
            return

    weights = category_weights(row)
    categories = rng.choice(CATEGORIES, size=total_claims, p=weights)

    if ("Rx" not in categories) and any(c in ("Professional", "Outpatient", "ER", "Inpatient") for c in categories):
        if rng.random() < RX_CO_OCCUR_P:
            categories = np.append(categories, "Rx")

    for cat in categories:
        build_claim_lines(row, cat, pools, cols)

# ============================ MAIN ============================

def main():
    mm, bt, pos, rev, proc, icd = load_tables()
    set_trend_base_year_from_mm(mm)
    print(f"ℹ️ Trend base year set to {TREND_BASE_YEAR} (annual={TREND_ANNUAL:.3%}, noise_std={TREND_NOISE_STD:.2%})")

    pools = build_pools(bt, pos, rev, proc, icd)

    # Columnar storage
    cols = {
        "claim_id": [], "line_number": [], "member_id": [], "category_of_service": [],
        "service_start_date": [], "service_end_date": [], "paid_date": [],
        "bill_type_code": [], "pos_code": [], "revenue_code": [], "procedure_code": [], "diagnosis_code": [],
        "allowed_amount": [], "paid_amount": [], "los_days": []
    }

    rows = mm.to_dict("records")
    iterable = tqdm(rows, total=len(rows), desc="Generating claims", unit="mm") if SHOW_PROGRESS else rows
    t0 = time.time()
    for row in iterable:
        generate_claims_for_month(row, pools, cols)
    print(f"⏱️ Generation finished in {time.time()-t0:,.1f}s")

    df_claims = pd.DataFrame(cols)

    # Column order (others will be appended if any)
    desired = [
        "claim_id","line_number","member_id","category_of_service",
        "service_start_date","service_end_date","paid_date",
        "bill_type_code","pos_code","revenue_code","procedure_code","diagnosis_code",
        "allowed_amount","paid_amount","los_days"
    ]
    extra = [c for c in df_claims.columns if c not in desired]
    df_claims = df_claims[desired + extra]

    print("💾 Writing claims.csv ...")
    df_claims.to_csv("claims.csv", index=False)
    print(f"✅ Wrote {len(df_claims):,} claim lines to claims.csv")

if __name__ == "__main__":
    main()