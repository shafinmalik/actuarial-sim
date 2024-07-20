import pandas as pd
import numpy as np

# Step 1: Define the number of policies and the period over which they are active
num_policies = 1000
start_date = pd.to_datetime('2015-01-01')
end_date = pd.to_datetime('2020-12-31')

# Set a fixed seed for reproducibility
np.random.seed(42)

# Generate policy data
policy_data = {
    'PolicyID': range(1, num_policies + 1),
    'InceptionDate': np.random.choice(pd.date_range(start_date, end_date, freq='M'), num_policies),
}
policy_df = pd.DataFrame(policy_data)
policy_df['ExpiryDate'] = policy_df['InceptionDate'] + pd.DateOffset(years=1)

# Step 2: Generate claims data
claims = []
claim_id = 1

for _, policy in policy_df.iterrows():
    num_claims = np.random.poisson(5)  # Assuming an average of 5 claims per policy
    for _ in range(num_claims):
        service_date = policy['InceptionDate'] + pd.DateOffset(days=np.random.randint(0, 365))
        submission_date = service_date + pd.DateOffset(days=np.random.randint(0, 30))
        paid_date = submission_date + pd.DateOffset(days=np.random.randint(0, 60))
        
        paid_amount = np.random.lognormal(mean=9, sigma=1)  # Log-normal distribution for claim severity
        reserved_amount = paid_amount * np.random.uniform(0.5, 1.5)  # Random factor for reserve amount
        
        diagnosis_code = f'DX{np.random.randint(1, 1000):03d}'  # Example diagnosis codes
        procedure_code = f'PR{np.random.randint(1, 1000):03d}'  # Example procedure codes
        provider_id = f'P{np.random.randint(1, 100):03d}'  # Example provider IDs
        
        claims.append({
            'ClaimID': claim_id,
            'PolicyID': policy['PolicyID'],
            'ServiceDate': service_date,
            'SubmissionDate': submission_date,
            'PaidDate': paid_date,
            'ProviderID': provider_id,
            'DiagnosisCode': diagnosis_code,
            'ProcedureCode': procedure_code,
            'PaidAmount': round(paid_amount, 2),
            'ReservedAmount': round(reserved_amount, 2)
        })
        claim_id += 1

claims_df = pd.DataFrame(claims)

# Display the first few rows of the dataset
print(claims_df.head())
