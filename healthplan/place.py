import pandas as pd

# ---------------------- POS CODE LIST ----------------------
# Source: CMS Place of Service Codes (https://www.cms.gov/medicare/coding/place-of-service-codes/place_of_service_code_set)

pos_data = [
    ("01", "Pharmacy"),
    ("02", "Telehealth Provided Other than in Patient’s Home"),
    ("03", "School"),
    ("04", "Homeless Shelter"),
    ("05", "Indian Health Service Free-standing Facility"),
    ("06", "Indian Health Service Provider-based Facility"),
    ("07", "Tribal 638 Free-standing Facility"),
    ("08", "Tribal 638 Provider-based Facility"),
    ("09", "Prison/Correctional Facility"),
    ("10", "Telehealth Provided in Patient’s Home"),
    ("11", "Office"),
    ("12", "Home"),
    ("13", "Assisted Living Facility"),
    ("14", "Group Home"),
    ("15", "Mobile Unit"),
    ("16", "Temporary Lodging"),
    ("17", "Walk-in Retail Health Clinic"),
    ("18", "Place of Employment–Worksite"),
    ("19", "Off Campus-Outpatient Hospital"),
    ("20", "Urgent Care Facility"),
    ("21", "Inpatient Hospital"),
    ("22", "On Campus-Outpatient Hospital"),
    ("23", "Emergency Room - Hospital"),
    ("24", "Ambulatory Surgical Center"),
    ("25", "Birthing Center"),
    ("26", "Military Treatment Facility"),
    ("31", "Skilled Nursing Facility"),
    ("32", "Nursing Facility"),
    ("33", "Custodial Care Facility"),
    ("34", "Hospice"),
    ("41", "Ambulance - Land"),
    ("42", "Ambulance - Air or Water"),
    ("49", "Independent Clinic"),
    ("50", "Federally Qualified Health Center"),
    ("51", "Inpatient Psychiatric Facility"),
    ("52", "Psychiatric Facility Partial Hospitalization"),
    ("53", "Community Mental Health Center"),
    ("54", "Intermediate Care Facility/Individuals with Intellectual Disabilities"),
    ("55", "Residential Substance Abuse Treatment Facility"),
    ("56", "Psychiatric Residential Treatment Center"),
    ("57", "Non-residential Substance Abuse Treatment Facility"),
    ("60", "Mass Immunization Center"),
    ("61", "Comprehensive Inpatient Rehabilitation Facility"),
    ("62", "Comprehensive Outpatient Rehabilitation Facility"),
    ("65", "End-Stage Renal Disease Treatment Facility"),
    ("71", "Public Health Clinic"),
    ("72", "Rural Health Clinic"),
    ("81", "Independent Laboratory"),
    ("99", "Other Place of Service")
]

# ---------------------- OPTIONAL CATEGORIZATION ----------------------

def map_category(code):
    code = int(code)
    if code in [21]:
        return "Inpatient"
    elif code in [22, 19]:
        return "Outpatient"
    elif code in [23]:
        return "ER"
    elif code in [31, 32]:
        return "SNF"
    elif code in [11, 17, 20, 72, 50]:
        return "Professional"
    elif code in [1, 10]:
        return "Rx"
    else:
        return "Other"

# ---------------------- BUILD AND EXPORT ----------------------

df_pos = pd.DataFrame(pos_data, columns=["pos_code", "description"])
df_pos["category_of_service"] = df_pos["pos_code"].apply(map_category)

df_pos.to_csv("pos_codes.csv", index=False)
print(f"✅ Saved {len(df_pos)} POS codes to 'pos_codes.csv'")