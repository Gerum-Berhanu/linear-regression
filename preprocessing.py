import pandas as pd
import numpy as np

class SalaryDataPreprocessor:
    """
    Data preprocessor specifically tailored for the Salary Dataset.
    Handles Target Encoding for Job Titles, and ordinal encoding for Gender and Education.
    """
    def __init__(self):
        """
        Saves the predefined mappings (Gender, Education)
        Prepares object-level variables (job_title_means, rare_threshold, global_mean_salary)
        """
        self.job_title_means = {}
        self.rare_threshold = 10
        self.gender_mapping = {"Male": 0, "Female": 1}
        self.education_mapping = {
            "High School": 0,
            "Bachelor's Degree": 1, "Bachelor's": 1,
            "Master's Degree": 2, "Master's": 2,
            "PhD": 3, "phD": 3,
        }
        self.global_mean_salary = 0.0

    def fit(self, df: pd.DataFrame, target_col: str = "Salary"):
        """
        Fit the preprocessor on the training data.
        Calculates and stores the averages (for Target Encoding the Job Titles)
        """
        # Ensure target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in the dataset.")

        # 1. Job Title Target Encoding
        job_titles_clean = df["Job Title"].astype("string").str.strip()
        job_title_counts = job_titles_clean.value_counts()
        rare_mask = job_titles_clean.isin(job_title_counts[job_title_counts <= self.rare_threshold].index)
        
        job_titles_grouped = job_titles_clean.copy()
        job_titles_grouped[rare_mask] = "Other"
        
        # Calculate storing means
        self.job_title_means = df.groupby(job_titles_grouped)[target_col].mean().to_dict()
        self.global_mean_salary = float(df[target_col].mean())
        
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset using fitted parameters.
        """
        df_out = df.copy()

        # 1. Transform Job Title
        if "Job Title" in df_out.columns:
            job_titles_clean = df_out["Job Title"].astype("string").str.strip()
            # Map known titles, fallback to 'Other' mean, or global mean if 'Other' is missing
            fallback_value = self.job_title_means.get("Other", self.global_mean_salary)
            df_out["Job_Title_ID"] = job_titles_clean.map(self.job_title_means).fillna(fallback_value)

        # 2. Transform Gender
        if "Gender" in df_out.columns:
            gender_clean = df_out["Gender"].astype("string").str.strip()
            df_out["Gender_ID"] = gender_clean.map(self.gender_mapping).astype("Int64")
            # Drop rows with unsupported genders (NaN after mapping)
            df_out.dropna(subset=["Gender_ID"], inplace=True) 

        # 3. Transform Education
        if "Education Level" in df_out.columns:
            education_clean = df_out["Education Level"].astype("string").str.strip()
            df_out["Education_Level_ID"] = education_clean.map(self.education_mapping).astype("Int64")
            
            if df_out["Education_Level_ID"].isna().any():
                unexpected = education_clean[df_out["Education_Level_ID"].isna()].dropna().unique()
                raise ValueError(f"Unmapped Education Level values found: {list(unexpected)}")

        return df_out
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = "Salary") -> pd.DataFrame:
        """
        Fit to data, then transform it.
        """
        return self.fit(df, target_col).transform(df)
