import pandas as pd

# ---------------------- DEFINITIONS ----------------------

# Facility type (1st digit)
facilities = {
    '1': 'Hospital',
    '2': 'Skilled Nursing Facility (SNF)',
    '3': 'Home Health',
    '4': 'Religious Non-Medical Facility',
    '5': 'Clinic',
    '6': 'Intermediate Care Facility (ICF)',
    '7': 'Clinic – Federally Qualified Health Center',
    '8': 'Special Facility',
    '9': 'Other'
}

# Classification (2nd digit)
classifications = {
    '1': 'Inpatient (Medicare Part A)',
    '2': 'Inpatient (Medicare Part B only)',
    '3': 'Outpatient',
    '4': 'Other (e.g., referenced diagnostic)',
    '5': 'Intermediate/Mental Health',
    '7': 'Clinic or Special Services',
    '8': 'Swing Bed'
}

# Frequency (3rd digit)
frequencies = {
    '1': 'Original (Admit thru Discharge)',
    '2': 'Interim - First Claim',
    '3': 'Interim - Continuing Claim',
    '4': 'Interim - Last Claim',
    '7': 'Replacement of Prior Claim',
    '8': 'Void/Cancel Prior Claim'
}

# Map bill type to category of service (optional, based on common usage)
def map_category(fac, cls):
    if fac == '1' and cls == '1':
        return 'Inpatient'
    elif fac == '1' and cls == '3':
        return 'Outpatient'
    elif fac == '2':
        return 'SNF'
    elif fac == '3':
        return 'Home Health'
    elif fac == '5' and cls == '3':
        return 'Professional'
    elif fac == '7':
        return 'Clinic'
    elif fac == '1' and cls == '4':
        return 'ER'
    else:
        return 'Other'

# ---------------------- BUILD TABLE ----------------------

records = []

for fac_code, fac_desc in facilities.items():
    for class_code, class_desc in classifications.items():
        for freq_code, freq_desc in frequencies.items():
            bill_code = f"{fac_code}{class_code}{freq_code}"
            category = map_category(fac_code, class_code)
            records.append({
                "bill_type_code": bill_code,
                "facility_type": fac_desc,
                "classification": class_desc,
                "frequency": freq_desc,
                "category_of_service": category
            })

df = pd.DataFrame(records)
df.to_csv("bill_type_codes.csv", index=False)
print(f"✅ Saved {len(df)} bill type codes to 'bill_type_codes.csv'")