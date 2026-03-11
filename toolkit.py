import numpy as np
import numpy.typing as npt

class ModelKit:
    def bucket_dataset(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], fold: int) -> list[tuple[npt.NDArray, npt.NDArray]]:
        """
        Divide dataset into shuffled k-fold buckets for cross-validation.
        
        Returns list of (X, y) tuples, each containing fold buckets of equal size
        """
        perm = np.random.permutation(len(X))
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        size = len(X) // fold
        bucket_list = []
        for n in range(fold):
            start = size * n
            end = size * (n + 1)
            bucket_set = (X_shuffled[start:end], y_shuffled[start:end])
            bucket_list.append(bucket_set)
        return bucket_list

    def mean_absolute_error(self, dataset_1: npt.NDArray[np.float64], dataset_2: npt.NDArray[np.float64]) -> np.float64:
        """Return the mean absolute error between two datasets."""
        return np.mean(abs(dataset_1 - dataset_2)).astype(np.float64)
    
    def mean_absolute_percentage_error(self, predicted_dataset: npt.NDArray[np.float64], actual_dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the mean absolute percentage error relative to the base dataset."""
        return np.mean(abs(predicted_dataset - actual_dataset) / actual_dataset).astype(np.float64)

    def root_mean_squared_error(self, predicted_dataset: npt.NDArray[np.float64], actual_dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the root mean squared error between two datasets."""
        return pow(np.mean(pow((predicted_dataset - actual_dataset), 2)), 0.5).astype(np.float64)
    
    def r_squared(self, predicted_dataset: npt.NDArray[np.float64], actual_dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the R-squared value of a target dataset relative to a base dataset."""
        ss_res = np.sum(pow((actual_dataset - predicted_dataset), 2)) # variance from the line
        ss_tot = np.sum(pow((actual_dataset - np.mean(actual_dataset)), 2)) # variance from the mean
        return (1 - (ss_res / ss_tot)).astype(np.float64)
    
    def split_dataset(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], train_percentage: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return X_train, X_test, y_train, y_test after shuffling."""
        perm = np.random.permutation(len(X))
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        split_point = int(len(X) * train_percentage)
        X_train = X_shuffled[:split_point]
        X_test = X_shuffled[split_point:]
        y_train = y_shuffled[:split_point]
        y_test = y_shuffled[split_point:]
        return X_train, X_test, y_train, y_test

class StatKit:
    def get_mean(self, dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the arithmetic mean of a dataset."""
        return np.mean(dataset).astype(np.float64)
    
    def get_var_std(self, dataset: npt.NDArray[np.float64]) -> tuple[np.float64, np.float64]:
        """Return the population variance and standard deviation of a dataset."""
        n = len(dataset)
        mean = self.get_mean(dataset)
        sqr_sum = np.sum(pow(dataset, 2))
        variance = (sqr_sum / n) - pow(mean, 2)
        return (variance, pow(variance, 0.5))
    
    def standardize(self, dataset: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the z-score standardized form of a dataset."""
        mean = self.get_mean(dataset)
        std_dev = self.get_var_std(dataset)[1]
        return (dataset - mean) / std_dev