"""
Benchmark Dataset Generator for Dataset Quality Auditor
Generates realistic datasets with deliberate, measurable data quality issues.
"""

import os
import numpy as np
import pandas as pd

def create_loan_dataset(output_path: str, n_rows: int = 1200, seed: int = 42):
    np.random.seed(seed)
    
    # 1. Base clean features
    applicant_ids = [f"APP-{10000 + i}" for i in range(n_rows)]
    ages = np.random.normal(38, 11, n_rows).astype(int)
    ages = np.clip(ages, 18, 75)
    
    incomes = np.random.lognormal(mean=10.8, sigma=0.6, size=n_rows).round(2)
    loan_amounts = (incomes * np.random.uniform(0.15, 0.45, n_rows)).round(-2)
    credit_scores = np.random.normal(680, 75, n_rows).astype(int)
    credit_scores = np.clip(credit_scores, 350, 850)
    employment_years = np.random.exponential(scale=6, size=n_rows).round(1)
    
    home_ownership = np.random.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], size=n_rows, p=[0.42, 0.18, 0.38, 0.02])
    loan_purpose = np.random.choice(["debt_consolidation", "credit_card", "home_improvement", "small_business", "medical"], size=n_rows)
    
    # Target: loan_status with Severe Class Imbalance (91% Approved = 1, 9% Rejected = 0)
    rejection_prob = np.where(credit_scores < 580, 0.45, 0.04)
    rejection_prob = np.where(incomes < 25000, rejection_prob + 0.25, rejection_prob)
    loan_status = np.where(np.random.rand(n_rows) < rejection_prob, 0, 1)
    # Ensure severe imbalance
    idx_rejected = np.where(loan_status == 0)[0]
    if len(idx_rejected) > n_rows * 0.12:
        to_flip = np.random.choice(idx_rejected, size=int(len(idx_rejected) - n_rows * 0.09), replace=False)
        loan_status[to_flip] = 1

    # Highly Correlated Feature: calculated monthly payment
    interest_rates = 0.08 + (850 - credit_scores) * 0.00025
    monthly_payment = (loan_amounts * (interest_rates / 12) / (1 - (1 + interest_rates / 12)**(-36))).round(2)
    # Incorrect data type: monthly_payment formatted as currency string with commas and $
    monthly_payment_str = [f"${val:,.2f}" for val in monthly_payment]
    
    # High correlation feature: principal_amount (99% correlation with loan_amount)
    principal_amount = loan_amounts + np.random.normal(0, 5, n_rows).round(2)

    # Constant column
    country_origin = ["United States"] * n_rows
    
    # Data Leakage Column: 'underwriting_auto_reject_flag' (100% correlation with target!)
    underwriting_flag = [1 if s == 0 else 0 for s in loan_status]
    # Add minor 1% noise to make it realistic leakage
    for i in np.random.choice(n_rows, size=4, replace=False):
        underwriting_flag[i] = 1 - underwriting_flag[i]

    # Application Date string (incorrect datatype: should be datetime)
    dates = pd.date_range(start="2024-01-01", periods=180, freq="D").strftime("%Y/%m/%d").tolist()
    app_dates = [np.random.choice(dates) for _ in range(n_rows)]

    df = pd.DataFrame({
        "applicant_id": applicant_ids,
        "age": ages.astype(object),
        "annual_income": incomes.astype(object),
        "loan_amount": loan_amounts,
        "principal_amount": principal_amount,
        "credit_score": credit_scores.astype(object),
        "employment_years": employment_years.astype(object),
        "monthly_payment": monthly_payment_str,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "application_date": app_dates,
        "country_origin": country_origin,
        "underwriting_flag": underwriting_flag,
        "loan_status": loan_status
    })

    # --- INJECT INTENTIONAL FLAWS ---

    # 1. Missing Values (~14% total across various columns)
    nan_income_idx = np.random.choice(n_rows, size=int(n_rows * 0.16), replace=False)
    df.loc[nan_income_idx, "annual_income"] = np.nan

    nan_emp_idx = np.random.choice(n_rows, size=int(n_rows * 0.12), replace=False)
    df.loc[nan_emp_idx, "employment_years"] = np.nan

    nan_home_idx = np.random.choice(n_rows, size=int(n_rows * 0.08), replace=False)
    df.loc[nan_home_idx, "home_ownership"] = np.nan

    # 2. Suspicious & Sentinel Values
    # -999 in credit_score
    sentinel_idx = np.random.choice(n_rows, size=15, replace=False)
    df.loc[sentinel_idx, "credit_score"] = -999

    # Negative age & Impossible age
    df.loc[12, "age"] = -4
    df.loc[45, "age"] = 999
    df.loc[88, "age"] = 0

    # Negative income
    df.loc[22, "annual_income"] = -45000.00

    # Suspicious string values in categorical column
    df.loc[34, "employment_years"] = "N/A"
    df.loc[56, "employment_years"] = "? "
    df.loc[78, "employment_years"] = "missing"

    # 3. Outliers (Extreme values)
    df.loc[101, "loan_amount"] = 8_500_000.00
    df.loc[102, "loan_amount"] = 5_200_000.00

    # 4. Duplicate Rows (Exact duplicates)
    dup_rows = df.iloc[5:20].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {len(df)} rows and {df.shape[1]} columns.")
    return df


def create_churn_dataset(output_path: str, n_rows: int = 1000, seed: int = 101):
    np.random.seed(seed)
    
    cust_ids = [f"CUST-{20000 + i}" for i in range(n_rows)]
    tenure = np.random.randint(1, 72, size=n_rows)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_rows).round(2)
    # Total charges is collinear with tenure * monthly_charges
    total_charges = (tenure * monthly_charges + np.random.normal(0, 10, size=n_rows)).round(2)
    
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.55, 0.25, 0.20])
    payment = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_rows)
    
    # Target: churned (Severe class imbalance: ~8% churn)
    churn_prob = np.where(contract == "Month-to-month", 0.12, 0.03)
    churned = np.where(np.random.rand(n_rows) < churn_prob, 1, 0)
    
    # Target Leakage: 'cancellation_survey_score' populated mostly when churned
    cancellation_code = np.where(churned == 1, np.random.choice([101, 102, 103], size=n_rows), 0)
    
    # Constant column
    currency = ["USD"] * n_rows
    
    # Quasi-constant column (99.6% same value)
    account_status = ["ACTIVE"] * n_rows
    for i in np.random.choice(n_rows, size=4, replace=False):
        account_status[i] = "SUSPENDED"

    df = pd.DataFrame({
        "customer_id": cust_ids,
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges.astype(object),
        "contract_type": contract,
        "payment_method": payment,
        "cancellation_code": cancellation_code,
        "currency": currency,
        "account_status": account_status,
        "churned": churned
    })

    # Injections
    # Missing values
    df.loc[np.random.choice(n_rows, size=60, replace=False), "total_charges"] = np.nan
    # Sentinel values
    df.loc[np.random.choice(n_rows, size=10, replace=False), "total_charges"] = " "
    # Duplicate rows
    df = pd.concat([df, df.iloc[10:25]], ignore_index=True)

    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {len(df)} rows and {df.shape[1]} columns.")
    return df


def create_medical_dataset(output_path: str, n_rows: int = 800, seed: int = 202):
    np.random.seed(seed)
    
    patient_ids = [f"PAT-{30000 + i}" for i in range(n_rows)]
    ages = np.random.randint(18, 90, size=n_rows).astype(object)
    systolic_bp = np.random.normal(125, 18, size=n_rows).round(1).astype(object)
    diastolic_bp = (systolic_bp.astype(float) * 0.65 + np.random.normal(0, 4, size=n_rows)).round(1)
    bmi = np.random.normal(27, 5, size=n_rows).round(1).astype(object)
    glucose = np.random.normal(105, 30, size=n_rows).round(1).astype(object)
    smoking = np.random.choice(["never", "Never", " former ", "CURRENT", "Unknown", None], size=n_rows)
    hospital_system = ["MetroHealth Network"] * n_rows
    
    # Target
    readmitted = np.random.choice([0, 1], size=n_rows, p=[0.88, 0.12])

    df = pd.DataFrame({
        "patient_id": patient_ids,
        "age": ages,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "bmi": bmi,
        "glucose": glucose,
        "smoking_status": smoking,
        "hospital_system": hospital_system,
        "readmitted": readmitted
    })

    # Injections
    # Invalid ranges
    df.loc[5, "systolic_bp"] = -120.0
    df.loc[15, "diastolic_bp"] = -80.0
    df.loc[25, "bmi"] = 999.9  # Outlier
    df.loc[35, "age"] = 999     # Impossible age
    df.loc[45, "glucose"] = -999.0  # Sentinel
    # Missing
    df.loc[np.random.choice(n_rows, size=50, replace=False), "glucose"] = np.nan

    # Duplicates
    df = pd.concat([df, df.iloc[0:10]], ignore_index=True)

    df.to_csv(output_path, index=False)
    print(f"Generated {output_path} with {len(df)} rows and {df.shape[1]} columns.")
    return df


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    create_loan_dataset(os.path.join(out_dir, "loan_approval_dirty.csv"))
    create_churn_dataset(os.path.join(out_dir, "customer_churn_leaky.csv"))
    create_medical_dataset(os.path.join(out_dir, "medical_patient_anomalous.csv"))
    print("All benchmark datasets successfully generated!")
