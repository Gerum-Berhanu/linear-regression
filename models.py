import numpy as np
import numpy.typing as npt
from toolkit import StatKit

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
    
class MultiLinearRegression:
    def train(self, features: list[npt.NDArray[np.float64]] | npt.NDArray[np.float64], label: npt.NDArray[np.float64], fit_intercept: bool = False):
        """
        Train the multiple linear regression model using Ordinary Least Squares (OLS).

        Args:
            features: A list of 1D numpy arrays, where each array represents a separate feature column, OR a pre-processed 2D numpy array design matrix (X).
            label: A 1D numpy array representing the target/output variable.
            fit_intercept: If True, prepends a column of ones to calculate the bias/intercept.
                           If False, no intercept is added.
            
        Raises:
            ValueError: If the label is not 1D, if there are no features, if features matrix is not 2D, 
                        or if the number of samples in the features and label arrays do not match.
        """
        if label.ndim != 1:
            raise ValueError("Label data must be one-dimensional.")

        # Save preference so predict() applies the same intercept logic later
        self.fit_intercept = fit_intercept

        # Handle pre-processed 2D matrix
        if isinstance(features, np.ndarray):
            if features.ndim != 2:
                raise ValueError("Pre-processed features matrix must be a 2D numpy array.")
            
            # Prepend intercept column if requested
            if fit_intercept:
                n_samples = features.shape[0]
                ones = np.ones((n_samples, 1), dtype=np.float64)
                self.X = np.hstack((ones, features))
            else:
                self.X = features
        # Handle list of 1D feature arrays
        else:
            if len(features) == 0:
                raise ValueError("At least one feature dataset is required.")
            self.X = self._features_matrix(features, fit_intercept)

        if len(label) != self.X.shape[0]:
            raise ValueError("Feature and label must have the same length.")

        self.y = label
        self._fit()

    def _features_matrix(self, features: list[npt.NDArray[np.float64]], fit_intercept: bool = False) -> npt.NDArray[np.float64]:
        """
        Construct and return the feature design matrix (X) by stacking the feature arrays.
        Optionally prepends a column of ones for the intercept term.

        Args:
            features: A list of 1D numpy arrays representing individual features.
            fit_intercept: Whether to prepend the bias/intercept column.

        Returns:
            A 2D numpy array of shape (n_samples, n_features) or (n_samples, n_features + 1) if intercept added.

        Raises:
            ValueError: If any feature array is empty, not 1D, or if the arrays have mismatched lengths.
        """
        if len(features) == 0:
            raise ValueError("At least one feature dataset is required.")

        n_samples = len(features[0])
        if n_samples == 0:
            raise ValueError("Feature datasets cannot be empty.")

        for idx, feature in enumerate(features):
            if feature.ndim != 1:
                raise ValueError(f"Feature datasets must be one-dimensional. features[{idx}] is not 1D.")
            if len(feature) != n_samples:
                raise ValueError(
                    f"All feature datasets must have the same length. "
                    f"features[0] has length {n_samples}, features[{idx}] has length {len(feature)}."
                )

        # Bind all arrays together into a single 2D design matrix
        X = np.column_stack(features).astype(np.float64)
        
        # Optionally prepend a column of 1s so the model can learn an intercept/baseline
        if fit_intercept:
            ones = np.ones((n_samples, 1), dtype=np.float64)
            return np.hstack((ones, X))
        
        return X

    def _fit(self):
        # \theta = (X^TX)^{-1}X^Ty
        X = self.X
        y = self.y
        T = X.T
        TX = T @ X

        det = np.linalg.det(TX)
        if np.isclose(det, 0):
            raise ValueError("Matrix is singular (not invertible).")
        
        inv = np.linalg.inv(TX)

        self.theta = inv @ (T @ y)

    def params(self) -> npt.NDArray[np.float64]:
        """
        Return the learned parameters (theta) of the multiple linear regression model.
        The first element is the intercept (bias), followed by the coefficients for each feature.
        """
        if not hasattr(self, "theta"):
            raise ValueError("Model must be trained before requesting parameters.")
        return self.theta

    def predict(self, new_dataset: list[npt.NDArray[np.float64]] | npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if not hasattr(self, "theta"):
            raise ValueError("Model must be trained before prediction.")

        # Match exactly how X was processed during training depending on input type
        if isinstance(new_dataset, np.ndarray):
            if new_dataset.ndim != 2:
                raise ValueError("Pre-processed features matrix must be a 2D numpy array.")
            
            # Apply identical intercept logic from the training phase directly to test set
            if self.fit_intercept:
                n_samples = new_dataset.shape[0]
                ones = np.ones((n_samples, 1), dtype=np.float64)
                X_new = np.hstack((ones, new_dataset))
            else:
                X_new = new_dataset
        else:
            # Reconstruct list of 1D arrays into matrix applying earlier intercept rules
            X_new = self._features_matrix(new_dataset, self.fit_intercept)
            
        return X_new @ self.theta
