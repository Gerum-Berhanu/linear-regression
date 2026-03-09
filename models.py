import numpy as np
from numpy.typing import NDArray

class SimpleLinearRegression:
    """Simple linear regression model with one input and one output.
    """
    def _nit__(self, input_train_data: NDArray[np.float64], output_train_data: NDArray[np.float64]) -> None: 
        self.input_train_data: NDArray[np.float64] = input_train_data
        self.output_train_data: NDArray[np.float64] = output_train_data
        self.n: int = len(input_train_data)

        self._calc_()
        self._fit_()
    
    def _calc_(self) -> None:
        """Calculates key terms later used in the calculations of finding the parameters.
        """
        self.Ex = np.sum(self.input_train_data)
        self.Ey = np.sum(self.output_train_data)
        self.Exy = np.sum(self.input_train_data * self.output_train_data)
        self.Ex_sqr = np.sum(pow(self.input_train_data, 2))
    
    def _fit_(self) -> None:
        r"""Solves for the best fit parameters b0 and b1. 
        
        Below are the two formulas (in LaTeX) used to solve for the parameters:

        $$b_0 = \frac{ \Sigma{y}\Sigma{x^2} - \Sigma{x}\Sigma{xy} }
            { n\Sigma{x^2} - (\Sigma{x})^2 }$$

        $$b_1 = \frac{ n\Sigma{xy} - \Sigma{x}\Sigma{y} }
            { n\Sigma{x^2} - (\Sigma{x})^2 }$$
        """
        common_denominator = (self.n * self.Ex_sqr) - pow(self.Ex, 2)
        self.b0 = ( (self.Ey * self.Ex_sqr) - (self.Ex * self.Exy) ) / common_denominator
        self.b1 = ( (self.n * self.Exy) - (self.Ex * self.Ey) ) / common_denominator

    def get_params(self) -> tuple[np.float64, np.float64]:
        return self.b0, self.b1

    def predict(self, input_test_data: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.b1 * input_test_data + self.b0
    
class ModelKit:
    def MAE(self, dataset_1: NDArray[np.float64], dataset_2: NDArray[np.float64]) -> np.float64:
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
    
    def split_data(self, dataset: NDArray[np.float64], train_percentage: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
    def get_mean(self, dataset: NDArray[np.float64]) -> np.float64:
        return np.sum(dataset) / len(dataset)
    
    def get_var_std(self, dataset: NDArray[np.float64]) -> tuple[np.float64, np.float64]:
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
    
    def standardize(self, dataset: NDArray[np.float64]) -> NDArray[np.float64]:
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