import numpy as np
import pandas as pd
import os

def generate_gd_datasets(output_dir="."):
    np.random.seed(42)
    
    # 1. Simple Univariate Dataset (1 Feature)
    # y = 3.5 * x + 2.0 + noise
    X_simple = 2 * np.random.rand(100, 1)
    y_simple = 2.0 + 3.5 * X_simple + np.random.randn(100, 1) * 0.5
    
    df_simple = pd.DataFrame({'X': X_simple.flatten(), 'y': y_simple.flatten()})
    df_simple.to_csv(os.path.join(output_dir, 'gd_univariate.csv'), index=False)
    print("Generated gd_univariate.csv")

    # 2. Multivariate Dataset (3 Features)
    # y = 4.2 * x1 - 2.5 * x2 + 1.8 * x3 + 10.0 + noise
    X_multi = 2 * np.random.rand(100, 3)
    weights = np.array([[4.2], [-2.5], [1.8]])
    y_multi = 10.0 + X_multi.dot(weights) + np.random.randn(100, 1) * 0.5
    
    df_multi = pd.DataFrame(X_multi, columns=['X1', 'X2', 'X3'])
    df_multi['y'] = y_multi.flatten()
    df_multi.to_csv(os.path.join(output_dir, 'gd_multivariate.csv'), index=False)
    print("Generated gd_multivariate.csv")

if __name__ == "__main__":
    # Save in the same directory as this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    generate_gd_datasets(current_dir)
