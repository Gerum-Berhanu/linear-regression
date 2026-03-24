import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
import models

def _validate_paired_datasets(dataset_1: npt.NDArray[np.float64], dataset_2: npt.NDArray[np.float64]) -> None:
    if len(dataset_1) != len(dataset_2):
        raise ValueError("Paired datasets must have the same length.")
    if len(dataset_1) == 0:
        raise ValueError("Paired datasets cannot be empty.")

class ModelMetrics:
    @staticmethod
    def mean_absolute_error(dataset_1: npt.NDArray[np.float64], dataset_2: npt.NDArray[np.float64]) -> np.float64:
        """Return the mean absolute error between two datasets."""
        _validate_paired_datasets(dataset_1, dataset_2)
        return np.mean(abs(dataset_1 - dataset_2)).astype(np.float64)
    
    @staticmethod
    def mean_absolute_percentage_error(predicted_dataset: npt.NDArray[np.float64], actual_dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the mean absolute percentage error excluding zero-valued targets."""
        _validate_paired_datasets(predicted_dataset, actual_dataset)
        non_zero_mask = actual_dataset != 0 # creates an array of True and False values with the same shape
        if not np.any(non_zero_mask):
            raise ValueError("MAPE is undefined when all actual values are zero.")
        
        filtered_predicted = predicted_dataset[non_zero_mask] # keep (drop) positions where the mask it True (False)
        filtered_actual = actual_dataset[non_zero_mask]
        return np.mean(abs(filtered_predicted - filtered_actual) / filtered_actual).astype(np.float64)
    
    @staticmethod
    def multicollinearity_metric(features: list[npt.NDArray[np.float64]]) -> list[np.float64]:
        VIF_list = []

        for i, target in enumerate(features):
            predictors = features.copy()
            predictors.pop(i)

            model = models.MultiLinearRegression()
            model.train(predictors, target)
            pred_target = model.predict(predictors)

            Rsqr = ModelMetrics.r_squared(predicted_dataset=pred_target, actual_dataset=target)

            if np.isclose(Rsqr, 1.0):
                VIF = np.inf
            else:
                VIF = 1 / (1 - Rsqr)
                
            VIF_list.append(np.float64(VIF))

        return VIF_list

    @staticmethod
    def root_mean_squared_error(predicted_dataset: npt.NDArray[np.float64], actual_dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the root mean squared error between two datasets."""
        _validate_paired_datasets(predicted_dataset, actual_dataset)
        return np.sqrt(np.mean(np.square(predicted_dataset - actual_dataset))).astype(np.float64)
    
    @staticmethod
    def r_squared(predicted_dataset: npt.NDArray[np.float64], actual_dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the R-squared value of a target dataset relative to a base dataset."""
        _validate_paired_datasets(predicted_dataset, actual_dataset)
        ss_res = np.sum(np.square(actual_dataset - predicted_dataset)) # variance from the line
        ss_tot = np.sum(np.square(actual_dataset - np.mean(actual_dataset))) # variance from the mean
        if np.isclose(ss_tot, 0):
            raise ValueError("R-squared is undefined when all actual values are the same.")
        return (1 - (ss_res / ss_tot)).astype(np.float64)

class DatasetKit:
    @staticmethod
    def bucket_dataset(X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], fold: int, random_state: int | None = None) -> list[tuple[npt.NDArray, npt.NDArray]]:
        """
        Divide dataset into shuffled k-fold buckets for cross-validation.
        
        Returns list of (X, y) tuples with nearly equal bucket sizes.
        """
        _validate_paired_datasets(X, y)
        if fold <= 1:
            raise ValueError("Fold count must be greater than 1.")
        if fold > len(X):
            raise ValueError("Fold count cannot be greater than dataset length.")

        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(X))
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        X_buckets = np.array_split(X_shuffled, fold)
        y_buckets = np.array_split(y_shuffled, fold)

        return list(zip(X_buckets, y_buckets))
    
    @staticmethod
    def one_hot_encode(df: pd.DataFrame, columns: list[str] | str, drop_first: bool = False) -> pd.DataFrame:
        """
        One-hot encode specified categorical columns in a DataFrame.
        
        Args:
            df: The pandas DataFrame containing the data.
            columns: A single column name or a list of column names to encode.
            drop_first: Whether to get k-1 dummies out of k categorical levels by removing the first level.
                        (True is recommended for linear regression to avoid perfect multicollinearity)
                        
        Returns:
            A new DataFrame with the target columns one-hot encoded and cast to float64.
        """
        if isinstance(columns, str):
            columns = [columns]
            
        # Get dummies drops the original categorical columns and replaces them with encoded boolean/int columns.
        encoded_df = pd.get_dummies(df, columns=columns, drop_first=drop_first, dtype=np.float64)
        return encoded_df
    
    @staticmethod
    def split_dataset(X: npt.NDArray[np.float64] | list[npt.NDArray[np.float64]], y: npt.NDArray[np.float64], train_percentage: float, random_state: int | None = None) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return X_train, X_test, y_train, y_test after shuffling.

        X may be 1-D (single feature) or 2-D with shape (n_samples, n_features).
        """
        if isinstance(X, list):
            if len(X) == 0:
                raise ValueError("X must contain at least one feature array.")
            X = np.column_stack(X).astype(np.float64)

        if X.ndim not in (1, 2):
            raise ValueError("X must be 1-D (single feature) or 2-D (multiple features).")
            
        _validate_paired_datasets(X, y)
        
        if train_percentage <= 0 or train_percentage >= 1:
            raise ValueError("Train percentage must be between 0 and 1, exclusive.")

        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(X))
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        split_point = int(len(X) * train_percentage)
        X_train = X_shuffled[:split_point]
        X_test = X_shuffled[split_point:]
        y_train = y_shuffled[:split_point]
        y_test = y_shuffled[split_point:]
        return X_train, X_test, y_train, y_test

class StatKit:
    @staticmethod
    def corr(dataset_1: npt.NDArray[np.float64], dataset_2: npt.NDArray[np.float64]) -> np.float64:
        cov = StatKit.cov(dataset_1, dataset_2)
        std_1 = StatKit.std(dataset_1)
        std_2 = StatKit.std(dataset_2)

        if np.isclose(std_1, 0) or np.isclose(std_2, 0):
            raise ValueError("Correlation is undefined when either dataset has zero variance.")

        return cov / (std_1 * std_2)
    
    @staticmethod
    def cov(dataset_1: npt.NDArray[np.float64], dataset_2: npt.NDArray[np.float64]) -> np.float64:
        _validate_paired_datasets(dataset_1, dataset_2)

        mean_1 = np.mean(dataset_1)
        mean_2 = np.mean(dataset_2)

        diff_1 = dataset_1 - mean_1
        diff_2 = dataset_2 - mean_2

        prod = diff_1 * diff_2

        return StatKit.mean(prod)

    @staticmethod
    def corr_matrix(datasets: list[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        """Return an NxN correlation matrix following the input list order."""
        if len(datasets) == 0:
            raise ValueError("At least one dataset is required.")
    
        expected_len = len(datasets[0])
        if expected_len == 0:
            raise ValueError("Datasets cannot be empty.")
    
        for idx, ds in enumerate(datasets):
            if len(ds) != expected_len:
                raise ValueError(
                    f"All datasets must have the same length. "
                    f"datasets[0] has length {expected_len}, datasets[{idx}] has length {len(ds)}."
                )
    
        n = len(datasets)
        matrix = np.empty((n, n), dtype=np.float64)
    
        for i in range(n):
            for j in range(i, n):
                score = StatKit.corr(datasets[i], datasets[j])
                matrix[i, j] = score
                matrix[j, i] = score
    
        return matrix

    @staticmethod
    def plot_corr_matrix_heatmap(datasets: list[npt.NDArray[np.float64]], labels: list[str]) -> None:
        """Plot a heatmap of the correlation matrix for the given datasets."""
        corr = StatKit.corr_matrix(datasets)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 6))
        im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, label="Correlation")

        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)

        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                plt.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black")

        plt.title("Correlation Matrix Heatmap")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def mean(dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the arithmetic mean of a dataset."""
        return np.mean(dataset).astype(np.float64)
    
    @staticmethod
    def std(dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the standard deviation of a dataset."""
        return np.sqrt(StatKit.var(dataset)).astype(np.float64)

    @staticmethod
    def var(dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the population variance of a dataset."""
        n = len(dataset)
        mean_val = StatKit.mean(dataset)
        sqr_sum = np.sum(np.square(dataset))
        variance = (sqr_sum / n) - np.square(mean_val)
        return variance
    
    @staticmethod
    def standardize(dataset: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the z-score standardized form of a dataset."""
        mean_val = StatKit.mean(dataset)
        std_dev = StatKit.std(dataset)
        return (dataset - mean_val) / std_dev