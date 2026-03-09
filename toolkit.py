import numpy as np
import numpy.typing as npt

class ModelKit:
    def mean_absolute_error(self, dataset_1: npt.NDArray[np.float64], dataset_2: npt.NDArray[np.float64]) -> np.float64:
        """Returns the mean absolute error between two datasets.

        Parameters
        ----------
        dataset_1 : NDArray[np.float64]
        dataset_2 : NDArray[np.float64]

        Returns
        -------
        np.float64
        """
        return np.sum(abs(dataset_1 - dataset_2)) / len(dataset_2)
    
    def mean_absolute_percentage_error(self, target_dataset: npt.NDArray[np.float64], base_dataset: npt.NDArray[np.float64]) -> np.float64:
        """Returns the mean absolute percentage error between two datasets.

        Parameters
        ----------
        target_dataset : NDArray[np.float64]
        base_dataset : NDArray[np.float64]

        Returns
        -------
        np.float64
        """
        return np.mean(abs(target_dataset - base_dataset) / base_dataset).astype(np.float64)
    
    def split_data(self, dataset: npt.NDArray[np.float64], train_percentage: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Splits a dataset into train and test data categories using the provided percentage value.

        Parameters
        ----------
        dataset : NDArray[np.float64]
        train_percentage : float
            The size of the train data.

        Returns
        -------
        tuple[NDArray[np.float64], NDArray[np.float64]]
            [0] is the train data
            [1] is the test data
        """
        split_size = int(len(dataset) * train_percentage)
        train_data = dataset[:split_size]
        test_data = dataset[split_size:]
        return train_data, test_data

class StatKit:
    def get_mean(self, dataset: npt.NDArray[np.float64]) -> np.float64:
        return np.sum(dataset) / len(dataset)
    
    def get_var_std(self, dataset: npt.NDArray[np.float64]) -> tuple[np.float64, np.float64]:
        """Returns the variance and standard deviation of the given dataset.

        Parameters
        ----------
        dataset : NDArray[np.float64]

        Returns
        -------
        tuple[np.float64, np.float64]
            [0] is variance
            [1] is standard deviation
        """
        n = len(dataset)
        mean = self.get_mean(dataset)
        sqr_sum = np.sum(pow(dataset, 2))
        variance = (sqr_sum / n) - pow(mean, 2)
        return (variance, pow(variance, 0.5))
    
    def standardize(self, dataset: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Returns the standardization form of a given dataset.

        Parameters
        ----------
        dataset : NDArray[np.float64]

        Returns
        -------
        NDArray[np.float64]
        """
        mean = self.get_mean(dataset)
        std_dev = self.get_var_std(dataset)[1]
        return (dataset - mean) / std_dev