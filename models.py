import numpy as np

class SimpleLinearRegression:
    """
    - simple linear regression
    - accepts the training dataset
    - do the calculations Ex, Ey, Exy, Ex^2
    - solve for the parameters
    """
    def __init__(self, input_train_data, output_train_data): 
        """
        Input Type (of np.array): [[x1,x2,...,xn], [y1,y2,...,yn]]
        """
        self.input_train_data = input_train_data
        self.output_train_data = output_train_data
        self.n = len(input_train_data)
        self._calc_()
        self._solve_()
    
    def _calc_(self):
        self.Ex = np.sum(self.input_train_data)
        self.Ey = np.sum(self.output_train_data)
        self.Exy = np.sum(self.input_train_data * self.output_train_data)
        self.Ex_sqr = np.sum(pow(self.input_train_data, 2))
    
    def _solve_(self):
        common_denominator = (self.n * self.Ex_sqr) - (self.Ex * self.Ex)
        self.b0 = ( (self.Ey * self.Ex_sqr) - (self.Ex * self.Exy) ) / common_denominator
        self.b1 = ( (self.n * self.Exy) - (self.Ex * self.Ey) ) / common_denominator

    def get_params(self):
        return self.b0, self.b1

    def predict(self, input_test_data):
        """
        Input Type (of np.array): [x1,x2,...,xn]
        Output Type (of np.array): [y1,y2,...,yn]
        """
        return self.b1 * input_test_data + self.b0
    
class ModelKit:
    def MAE(self, dataset_1, dataset_2):
        """
        Returns the mean absolute error between the two given datasets.
        """
        return np.sum(abs(dataset_1 - dataset_2)) / len(dataset_2)
    
    def split_data(self, dataset, train_percentage):
        """
        Returns (train_data, test_data)
        """
        split_size = int(len(dataset) * train_percentage)
        train_data = dataset[:split_size]
        test_data = dataset[split_size:]
        return train_data, test_data

class StatKit:
    def get_mean(self, dataset):
        return np.sum(dataset) / len(dataset)
    
    def get_var_std(self, dataset):
        """
        Returns (variance, standard deviation)
        """
        n = len(dataset)
        mean = self.get_mean(dataset)
        sqr_sum = np.sum(pow(dataset, 2))
        variance = (sqr_sum / n) - pow(mean, 2)
        return (variance, pow(variance, 0.5))
    
    def standardize(self, dataset):
        """
        Returns (standardized data, mean, standard deviation)
        """
        mean = self.get_mean(dataset)
        std_dev = self.get_var_std(dataset)[1]
        return (dataset - mean) / std_dev, mean, std_dev