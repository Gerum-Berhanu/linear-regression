import numpy as np
import numpy.typing as npt

class ModelKit:
    def _validate_paired_datasets(self, dataset_1: npt.NDArray[np.float64], dataset_2: npt.NDArray[np.float64]) -> None:
        if len(dataset_1) != len(dataset_2):
            raise ValueError("Paired datasets must have the same length.")
        if len(dataset_1) == 0:
            raise ValueError("Paired datasets cannot be empty.")

    def bucket_dataset(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], fold: int, random_state: int | None = None) -> list[tuple[npt.NDArray, npt.NDArray]]:
        """
        Divide dataset into shuffled k-fold buckets for cross-validation.
        
        Returns list of (X, y) tuples with nearly equal bucket sizes.
        """
        self._validate_paired_datasets(X, y)
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

    def mean_absolute_error(self, dataset_1: npt.NDArray[np.float64], dataset_2: npt.NDArray[np.float64]) -> np.float64:
        """Return the mean absolute error between two datasets."""
        self._validate_paired_datasets(dataset_1, dataset_2)
        return np.mean(abs(dataset_1 - dataset_2)).astype(np.float64)
    
    def mean_absolute_percentage_error(self, predicted_dataset: npt.NDArray[np.float64], actual_dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the mean absolute percentage error excluding zero-valued targets."""
        self._validate_paired_datasets(predicted_dataset, actual_dataset)
        non_zero_mask = actual_dataset != 0 # creates an array of True and False values with the same shape
        if not np.any(non_zero_mask):
            raise ValueError("MAPE is undefined when all actual values are zero.")
        
        filtered_predicted = predicted_dataset[non_zero_mask] # keep (drop) positions where the mask it True (False)
        filtered_actual = actual_dataset[non_zero_mask]
        return np.mean(abs(filtered_predicted - filtered_actual) / filtered_actual).astype(np.float64)

    def root_mean_squared_error(self, predicted_dataset: npt.NDArray[np.float64], actual_dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the root mean squared error between two datasets."""
        self._validate_paired_datasets(predicted_dataset, actual_dataset)
        return pow(np.mean(pow((predicted_dataset - actual_dataset), 2)), 0.5).astype(np.float64)
    
    def r_squared(self, predicted_dataset: npt.NDArray[np.float64], actual_dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the R-squared value of a target dataset relative to a base dataset."""
        self._validate_paired_datasets(predicted_dataset, actual_dataset)
        ss_res = np.sum(pow((actual_dataset - predicted_dataset), 2)) # variance from the line
        ss_tot = np.sum(pow((actual_dataset - np.mean(actual_dataset)), 2)) # variance from the mean
        if np.isclose(ss_tot, 0):
            raise ValueError("R-squared is undefined when all actual values are the same.")
        return (1 - (ss_res / ss_tot)).astype(np.float64)
    
    def split_dataset(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], train_percentage: float, random_state: int | None = None) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return X_train, X_test, y_train, y_test after shuffling."""
        self._validate_paired_datasets(X, y)
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
    def mean(self, dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the arithmetic mean of a dataset."""
        return np.mean(dataset).astype(np.float64)
    
    def std(self, dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the standard deviation of a dataset."""
        return pow(self.var(dataset), 0.5)

    def var(self, dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the population variance of a dataset."""
        n = len(dataset)
        mean_val = self.mean(dataset)
        sqr_sum = np.sum(pow(dataset, 2))
        variance = (sqr_sum / n) - pow(mean_val, 2)
        return variance
    
    def standardize(self, dataset: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the z-score standardized form of a dataset."""
        mean_val = self.mean(dataset)
        std_dev = self.std(dataset)
        return (dataset - mean_val) / std_dev