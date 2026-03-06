import numpy as np

class SLR:
    """
    - simple linear regression
    - accepts the training dataset
    - do the calculations Ex, Ey, Exy, Ex^2
    - solve for the parameters
    """
    def __init__(self, data_list): 
        """
        Input Type (of np.array): [[x1,x2,...,xn], [y1,y2,...,yn]]
        """
        self.dl = data_list
        self.n = len(data_list[0])
        self._calc_()
        self._solve_()
    
    def _calc_(self):
        self.s_x = np.sum(self.dl[0])
        self.s_y = np.sum(self.dl[1])
        self.s_xy = np.sum(self.dl[0] * self.dl[1])
        self.s_x_sqr = np.sum(self.dl[0] * self.dl[0])
    
    def _solve_(self):
        common_denominator = (self.n * self.s_x_sqr) - (self.s_x * self.s_x)
        self.m = ( (self.n * self.s_xy) - (self.s_x * self.s_y) ) / common_denominator
        self.b = ( (self.s_y * self.s_x_sqr) - (self.s_x * self.s_xy) ) / common_denominator

    def get_params(self):
        return self.m, self.b

    def predict(self, test_data):
        """
        Input Type (of np.array): [x1,x2,...,xn]
        Output Type (of np.array): [y1,y2,...,yn]
        """
        return self.m * test_data + self.b
    
class ModelKit:
    def MAE(self, target, base):
        """
        Returns the mean absolute error between the two given datasets.
        """
        return np.sum(abs(target - base)) / len(base)
    
    def split_data(self, data_list, train_percentage):
        """
        Returns (train_data, test_data)"""
        split_size = int(len(data_list) * train_percentage)
        train_data = data_list[:split_size]
        test_data = data_list[split_size:]
        return train_data, test_data

class StatKit:
    def get_mean(self, data_list):
        return np.sum(data_list) / len(data_list)
    
    def get_var_std(self, data_list):
        """
        Returns (variance, standard deviation)
        """
        n = len(data_list)
        mean = self.get_mean(data_list)
        sqr_sum = np.sum(pow(data_list, 2))
        variance = (sqr_sum / n) - pow(mean, 2)
        return (variance, pow(variance, 0.5))
    
    def standardize(self, data_list):
        """
        Returns (standardized data, mean, standard deviation)
        """
        mean = self.get_mean(data_list)
        std_dev = self.get_var_std(data_list)[1]
        return (data_list - mean) / std_dev, mean, std_dev