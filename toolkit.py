import numpy as np
import numpy.typing as npt

class ModelKit:
    def mean_absolute_error(self, dataset_1: npt.NDArray[np.float64], dataset_2: npt.NDArray[np.float64]) -> np.float64:
        """Return the mean absolute error between two datasets."""
        return np.mean(abs(dataset_1 - dataset_2)).astype(np.float64)
    
    def mean_absolute_percentage_error(self, target_dataset: npt.NDArray[np.float64], base_dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the mean absolute percentage error relative to the base dataset."""
        return np.mean(abs(target_dataset - base_dataset) / base_dataset).astype(np.float64)

    def root_mean_squared_error(self, target_dataset: npt.NDArray[np.float64], base_dataset: npt.NDArray[np.float64]) -> np.float64:
        """Return the root mean squared error between two datasets."""
        return pow(np.mean(pow((target_dataset - base_dataset), 2)), 0.5).astype(np.float64)
    
    
    def split_data(self, dataset: npt.NDArray[np.float64], train_percentage: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Split a dataset into train and test slices without shuffling."""
        split_size = int(len(dataset) * train_percentage)
        train_data = dataset[:split_size]
        test_data = dataset[split_size:]
        return train_data, test_data

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