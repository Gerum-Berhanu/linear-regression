import kagglehub
import os
import shutil
import pandas as pd

# 1. Define your target destination first
target_dir = "./datasets"
target_file = os.path.join(target_dir, "Salary_Data.csv")

# 2. Check if the file already exists locally
if os.path.exists(target_file):
    print(f"Dataset already exists at {target_file}. Skipping download...")
else:
    print("Dataset not found locally. Starting download...")
    
    # 3. Download to the default cache
    cache_path = kagglehub.dataset_download("mohithsairamreddy/salary-data")

    # 4. Create the directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 5. Locate the downloaded file and move it
    # * Note: sometimes kagglehub returns the file path directly, 
    # but usually it's a folder.
    if os.path.isdir(cache_path):
        downloaded_file = os.path.join(cache_path, "Salary_Data.csv")
    else:
        downloaded_file = cache_path

    if os.path.exists(downloaded_file):
        shutil.move(downloaded_file, target_file)
        print(f"File successfully moved to: {target_file}")
    else:
        print("Could not find Salary_Data.csv in the downloaded cache.")

# 6. A quick check (Always runs)
if os.path.exists(target_file):
    df = pd.read_csv(target_file)
    print("\nQuick check - First 5 rows:")
    print(df.head())
else:
    print("\nError: Could not load data because the file is missing.")