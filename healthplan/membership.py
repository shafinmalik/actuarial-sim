import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime
from dateutil.relativedelta import relativedelta


# ---------------------- RANDOMNESS TOGGLE ----------------------
USE_FIXED_SEED = False  # Set to True for reproducible results

# ---------------------- INITIAL SETUP ----------------------
fake = Faker()
if USE_FIXED_SEED:
    np.random.seed(42)
    random.seed(42)
# If False, system RNG is used (non-deterministic)

# ---------------------- CONFIGURATION ----------------------

NUM_MEMBERS = 500
HP_IDS = ['HP001', 'HP002', 'HP003']
PLAN_TYPES = ['HMO', 'PPO', 'SNP']

COV_START = pd.to_datetime("2022-01-01")
COV_END = pd.to_datetime("2025-12-01")

MIN_MONTHS = 1
MAX_MONTHS = 36
AGE_DISTRIBUTION = 'skewed_old'  # Options: 'uniform', 'skewed_young', 'skewed_old'
NOISE_LEVEL = 0.2                # Controls randomness of premium
RISK_NOISE_STD = 0.15            # Noise for risk score
PREMIUM_MULTIPLIER = 0.4         # Premium increases per unit risk

# Prevalence of conditions
condition_probs = {
    'CKD': [0.60, 0.05, 0.08, 0.10, 0.05, 0.02, 0.10],  # No CKD to ESRD
    'COPD': 0.15,
    'Asthma': 0.10,
    'Cancer_Skin': 0.03,
    'Cancer_Blood': 0.01,
    'Cancer_Pancreatic': 0.005,
    'Cancer_Other': 0.02,
    'Diabetes': 0.25,
    'Hypertension': 0.40
}

# Remission probabilities per month
remission_probs = {
    'Asthma': 0.01,
    'COPD': 0.005,
    'Cancer_Skin': 0.10,
    'Cancer_Blood': 0.03,
    'Cancer_Pancreatic': 0.005,
    'Cancer_Other': 0.07
}

ckd_progress_prob = 0.04
ckd_regress_prob = 0.002

# ---------------------- CONFIG GUARD ----------------------

total_months_available = (COV_END.to_period("M") - COV_START.to_period("M")).n + 1
if MAX_MONTHS > total_months_available:
    print(f"⚠️ MAX_MONTHS ({MAX_MONTHS}) exceeds available coverage window ({total_months_available}). Adjusting MAX_MONTHS = {total_months_available}")
    MAX_MONTHS = total_months_available

# ---------------------- UTILITY FUNCTIONS ----------------------

def get_random_coverage_months(cov_start, cov_end, min_months, max_months):
    total_months_available = (cov_end.to_period("M") - cov_start.to_period("M")).n + 1
    if min_months > total_months_available:
        raise ValueError(
            f"❌ Not enough available months ({total_months_available}) to meet minimum coverage of {min_months} months."
        )

    # Choose start month uniformly over all starts that can still fit min_months
    max_start_index = total_months_available - min_months  # inclusive
    start_idx = random.randint(0, max_start_index)

    # Now choose a duration that fits after the chosen start
    remaining = total_months_available - start_idx
    adjusted_max = min(max_months, remaining)
    months_total = random.randint(min_months, adjusted_max)

    start_date = cov_start + relativedelta(months=start_idx)
    return start_date, months_total

def generate_birth_year(dist):
    if dist == 'uniform':
        return fake.date_of_birth(minimum_age=18, maximum_age=90)
    elif dist == 'skewed_young':
        age = int(np.random.beta(2, 5) * 72 + 18)
    elif dist == 'skewed_old':
        age = int(np.random.beta(5, 2) * 72 + 18)
    else:
        age = random.randint(18, 90)
    return datetime.today() - relativedelta(years=age)

def update_ckd_stage(stage):
    if stage == 6:
        return stage
    if np.random.rand() < ckd_progress_prob and stage < 6:
        return stage + 1
    elif np.random.rand() < ckd_regress_prob and stage > 0:
        return stage - 1
    return stage

def calculate_risk_score(age, ckd_stage, esrd, flags):
    score = 0
    score += [0.0, 0.1, 0.15, 0.25, 0.35, 0.50, 0.80][ckd_stage]
    if esrd: score += 1.2
    if age >= 85: score += 0.8
    elif age >= 75: score += 0.6
    elif age >= 65: score += 0.4
    elif age >= 45: score += 0.2

    score += 0.30 * flags.get('Diabetes', 0)
    score += 0.35 * flags.get('Hypertension', 0)
    score += 0.40 * flags.get('COPD', 0)
    score += 0.25 * flags.get('Asthma', 0)
    score += 0.45 * flags.get('Cancer_Skin', 0)
    score += 0.75 * flags.get('Cancer_Blood', 0)
    score += 0.95 * flags.get('Cancer_Pancreatic', 0)
    score += 0.55 * flags.get('Cancer_Other', 0)

    noise = np.random.normal(0, RISK_NOISE_STD)
    return max(0.1, round(score + noise, 2))

# ---------------------- MAIN GENERATOR ----------------------

def generate_member_months(num_members):
    all_records = []

    for member_id in range(1, num_members + 1):
        birth_date = generate_birth_year(AGE_DISTRIBUTION)
        gender = random.choice(['M', 'F'])
        state = fake.state_abbr()
        zip_code = fake.zipcode()[:5]
        hp_id = random.choice(HP_IDS)
        plan_type = random.choice(PLAN_TYPES)

        cov_start_actual, months_total = get_random_coverage_months(COV_START, COV_END, MIN_MONTHS, MAX_MONTHS)
        months = pd.date_range(cov_start_actual, periods=months_total, freq='MS')

        ckd_stage = np.random.choice(range(7), p=condition_probs['CKD'])
        has_flags = {
            'Diabetes': np.random.rand() < condition_probs['Diabetes'],
            'Hypertension': np.random.rand() < condition_probs['Hypertension'],
            'COPD': np.random.rand() < condition_probs['COPD'],
            'Asthma': np.random.rand() < condition_probs['Asthma'],
            'Cancer_Skin': np.random.rand() < condition_probs['Cancer_Skin'],
            'Cancer_Blood': np.random.rand() < condition_probs['Cancer_Blood'],
            'Cancer_Pancreatic': np.random.rand() < condition_probs['Cancer_Pancreatic'],
            'Cancer_Other': np.random.rand() < condition_probs['Cancer_Other']
        }

        months_by_year = {}
        for m in months:
            y = m.year
            months_by_year.setdefault(y, []).append(m)

        for year, month_list in months_by_year.items():
            first_month = month_list[0]
            age = relativedelta(first_month, birth_date).years
            esrd_flag = 1 if ckd_stage == 6 else 0
            flags_snapshot = has_flags.copy()
            risk_score = calculate_risk_score(age, ckd_stage, esrd_flag, flags_snapshot)
            base = {'HMO': 600, 'PPO': 750, 'SNP': 850}[plan_type]
            premium = round(base * (1 + risk_score * PREMIUM_MULTIPLIER) + np.random.normal(0, NOISE_LEVEL * 50), 2)

            for m in month_list:
                age_m = relativedelta(m, birth_date).years
                esrd_flag = 1 if ckd_stage == 6 else 0

                row = {
                    'member_id': f"M{member_id:05}",
                    'HP_ID': hp_id,
                    'COV_Month': m.strftime('%Y-%m'),
                    'premium': premium,
                    'risk_score': risk_score,
                    'age_at_month': age_m,
                    'birth_date': birth_date.strftime('%Y-%m-%d'),
                    'gender': gender,
                    'state': state,
                    'zip_code': zip_code,
                    'plan_type': plan_type,
                    'CKD_Status': ['No CKD', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4', 'Stage 5', 'ESRD'][ckd_stage],
                    'ESRD_Flag': esrd_flag
                }

                for cond, val in has_flags.items():
                    row[cond] = int(val)

                all_records.append(row)

                # Update CKD and check for condition remission
                ckd_stage = update_ckd_stage(ckd_stage)
                for cond in remission_probs:
                    if has_flags[cond] and np.random.rand() < remission_probs[cond]:
                        has_flags[cond] = False

    return pd.DataFrame(all_records)

# ---------------------- EXECUTION ----------------------

if __name__ == "__main__":
    df_mm = generate_member_months(NUM_MEMBERS)
    df_mm.to_csv("member_months.csv", index=False)
    print("✅ Member-month table saved to member_months.csv")