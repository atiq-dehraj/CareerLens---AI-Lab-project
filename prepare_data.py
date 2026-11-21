import pandas as pd
import numpy as np
import os

# Aapke archive folder ka path
ARCHIVE_PATH = "archive" 

def prepare_dataset():
    print("Data load ho raha hai... (Isme thoda time lag sakta hai kyunki file badi hai)")
    
    postings_path = f"{ARCHIVE_PATH}/postings.csv"
    if not os.path.exists(postings_path):
        postings_path = f"{ARCHIVE_PATH}/postings.xlsx"
        
    if not os.path.exists(postings_path):
        print("Error: 'postings.csv' ya 'postings.xlsx' archive folder mein nahi mili.")
        return

    try:
        if postings_path.endswith('.csv'):
            jobs = pd.read_csv(postings_path, low_memory=False)
        else:
            jobs = pd.read_excel(postings_path)
    except Exception as e:
        print(f"Postings file read karne mein error: {e}")
        return

    # --- FIX: Kaggle dataset mein title column ka naam 'title' hai, use 'job_title' banao ---
    if 'title' in jobs.columns:
        jobs.rename(columns={'title': 'job_title'}, inplace=True)
    # ----------------------------------------------------------------------------------

    companies_path = f"{ARCHIVE_PATH}/companies/companies.csv"
    if not os.path.exists(companies_path):
        print("Error: 'companies/companies.csv' nahi mili.")
        return
    
    companies = pd.read_csv(companies_path)

    print("Companies ka naam merge kiya ja raha hai...")
    df = jobs.merge(companies[['company_id', 'name']], on='company_id', how='left')
    df.rename(columns={'name': 'company'}, inplace=True)

    print("Salary process ki ja rahi hai (Yearly format)...")
    def get_annual_salary(row):
        sal = row.get('med_salary', np.nan)
        if pd.isna(sal):
            if pd.notna(row.get('min_salary')) and pd.notna(row.get('max_salary')):
                sal = (row['min_salary'] + row['max_salary']) / 2
            elif pd.notna(row.get('min_salary')):
                sal = row['min_salary']
            elif pd.notna(row.get('max_salary')):
                sal = row['max_salary']
            else:
                return np.nan
                
        period = str(row.get('pay_period')).upper() if pd.notna(row.get('pay_period')) else 'YEARLY'
        if period == 'HOURLY':
            return sal * 2080 
        elif period == 'MONTHLY':
            return sal * 12
        return sal

    annual_salary = pd.to_numeric(
        df['normalized_salary'], errors='coerce'
    )

    df['salary'] = annual_salary.where(
        (df['currency'] == 'USD') & (annual_salary > 0)
    )

    print("Job Level map kiya ja raha hai...")

    def map_level(exp_level):
        if pd.isna(exp_level):
            return pd.NA
        return str(exp_level).strip()

    if 'formatted_experience_level' in df.columns:
        df['job_level'] = df['formatted_experience_level'].apply(map_level)
    else:
        df['job_level'] = pd.NA

    # Description check
    if 'description' not in df.columns:
         df['description'] = "No description available"
         
    cols_to_keep = ['job_title', 'company', 'salary', 'description', 'job_level']
    
    # Check if all columns exist before filtering
    for col in cols_to_keep:
        if col not in df.columns:
            print(f"Warning: {col} column dataset mein nahi mila!")

    final_df = df[cols_to_keep].copy()
    
    # Remove only records without a valid job title
    final_df.dropna(subset=['job_title'], inplace=True)

    # Preserve missing company and salary information
    final_df['company'] = final_df['company'].fillna('Unknown')

    if not os.path.exists('data'):
        os.makedirs('data')
        
    final_df.to_csv('data/job_postings.csv', index=False)
    print(f"\nSuccess! Data prepare ho gaya aur 'data/job_postings.csv' mein save ho gaya.")
    print(f"Total Rows: {len(final_df)}")

if __name__ == "__main__":
    prepare_dataset()
