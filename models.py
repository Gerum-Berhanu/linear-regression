import numpy as np
import numpy.typing as npt

class SimpleLinearRegression:
    """Simple linear regression model with one input and one output.
    """
    def train(self, input_train_data: npt.NDArray[np.float64], output_train_data: npt.NDArray[np.float64]) -> None: 
        if input_train_data.ndim != 1:
            raise ValueError("Input training data must be one-dimensional for SimpleLinearRegression.")
        if output_train_data.ndim != 1:
            raise ValueError("Output training data must be one-dimensional for SimpleLinearRegression.")

        if len(input_train_data) != len(output_train_data):
            raise ValueError("Input and output train data don't have the same length.")
        if len(input_train_data) == 0 or len(output_train_data) == 0:
            raise ValueError("Either input or output train data has zero length.")

        self.input_train_data: npt.NDArray[np.float64] = input_train_data
        self.output_train_data: npt.NDArray[np.float64] = output_train_data
        self.n: int = len(input_train_data)

        self._calc()
        self._fit()

    def _ensure_fitted(self) -> None:
        if not hasattr(self, "b0") or not hasattr(self, "b1"):
            raise ValueError("Model must be trained before requesting parameters or predictions.")
    
    def _calc(self) -> None:
        """Calculates key terms later used in the calculations of finding the parameters.
        """
        self.Ex = np.sum(self.input_train_data)
        self.Ey = np.sum(self.output_train_data)
        self.Exy = np.sum(self.input_train_data * self.output_train_data)
        self.Ex_sqr = np.sum(pow(self.input_train_data, 2))
    
    def _fit(self) -> None:
        r"""Solves for the best fit parameters b0 and b1. 
        
        Below are the two formulas (in LaTeX) used to solve for the parameters:

        $$b_0 = \frac{ \Sigma{y}\Sigma{x^2} - \Sigma{x}\Sigma{xy} }
            { n\Sigma{x^2} - (\Sigma{x})^2 }$$

        $$b_1 = \frac{ n\Sigma{xy} - \Sigma{x}\Sigma{y} }
            { n\Sigma{x^2} - (\Sigma{x})^2 }$$
        """
        common_denominator = (self.n * self.Ex_sqr) - np.square(self.Ex)
        if np.isclose(common_denominator, 0):
            raise ValueError("All features values cannot be the same.")

        self.b0 = ( (self.Ey * self.Ex_sqr) - (self.Ex * self.Exy) ) / common_denominator
        self.b1 = ( (self.n * self.Exy) - (self.Ex * self.Ey) ) / common_denominator

    def params(self) -> tuple[np.float64, np.float64]:
        self._ensure_fitted()
        return self.b0, self.b1

    def predict(self, input_test_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if input_test_data.ndim != 1:
            raise ValueError("Prediction input data must be one-dimensional for SimpleLinearRegression.")
        self._ensure_fitted()
        return self.b0 + self.b1 * input_test_data
    