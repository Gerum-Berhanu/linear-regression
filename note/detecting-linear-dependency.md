To check for linear dependencies (multicollinearity) in your dataset and mitigate them, you can use several statistical and algebraic techniques. The approach varies slightly depending on whether you are looking at simple 1-to-1 dependencies or complex multi-feature dependencies.

### 1. How to Check for Linear Dependency

#### A. Checking 1-to-1 Dependency (Between Two Features)
*   **Correlation Matrix**: The most common method is computing the Pearson correlation coefficient between all pairs of numeric features. 
    *   **How**: In pandas, use `df.corr()`. You can visualize this using a heatmap (e.g., `seaborn.heatmap`).
    *   **What to look for**: Values close to `1` or `-1` indicate strong linear dependency. A common threshold for concern is absolute correlation $> 0.8$ or $0.9$.
*   **Scatter Plots**: Plotting feature $A$ against feature $B$ (e.g., via `seaborn.pairplot`). A tight straight line indicates linear dependence.

#### B. Checking Multicollinearity (Among Multiple Features)
Sometimes a feature isn't simply a multiple of *one* other feature, but a linear combination of *several* features (e.g., $X_3 \approx 2X_1 - X_2$). A simple correlation matrix won't easily catch this.
*   **Variance Inflation Factor (VIF)**: This is the standard metric for multi-feature dependency. It measures how much the variance of an estimated regression coefficient increases when your predictors are correlated.
    *   **How**: For each feature $i$, run a regression predicting feature $i$ using all *other* features to get the $R^2$ value. The formula is $VIF_i = \frac{1}{1 - R_i^2}$.
    *   **What to look for**: $VIF = 1$ means no correlation. $VIF > 5$ represents moderate collinearity, and $VIF > 10$ indicates severe multicollinearity that needs addressing.
*   **Determinant of $X^T X$**: As explained in your near-linear-dependency.md notes, if $\det(X^T X)$ is exactly $0$ or extremely close to $0$, you have exact or near-linear dependence.
*   **Condition Number**: Computes the ratio of the largest to the smallest eigenvalue of the design matrix. In Python: `numpy.linalg.cond(X)`. A condition number $> 30$ generally indicates severe multicollinearity.

---

### 2. How to Avoid or Fix It

Once you've identified multicollinearity, you can resolve it using the following strategies:

#### A. Feature Elimination (Drop Features)
The simplest fix. If $X_1$ and $X_2$ are highly correlated, providing redundant information, simply drop one of them. For multi-feature dependencies, sequentially drop the feature with the highest VIF score until all remaining VIFs drop below your threshold (e.g., 5).

#### B. Feature Engineering (Combine Features)
Use domain knowledge to combine dependent features into a single meaningful metric.
*   *Example*: If you have `house_length` and `house_width` which are highly correlated, multiply them to create a single `house_area` feature and drop the original two.

#### C. Regularization Models
If you want to keep all features without manually dropping them, use models that natively handle the extreme sensitivities of the $X^T X$ matrix:
*   **Ridge Regression (L2)**: Adds a penalty term ($\lambda \times I$) to the diagonal of $X^T X$, artificially inflating its determinant. By solving $(X^T X + \lambda I)^{-1} X^T y$, Ridge stabilizes the inverse and prevents the coefficients ($\theta$) from randomly swinging to $\pm 1,000,000$.
*   **Lasso Regression (L1)**: Adds an absolute value penalty. Lasso has the added benefit of acting as automatic feature selection—it will force the coefficients of redundant collinear features exactly back to $0$.

#### D. Dimensionality Reduction (PCA)
Use techniques like **Principal Component Analysis (PCA)** or **Partial Least Squares (PLS)**.
*   PCA creates new, artificial features (Principal Components) by squashing the existing features together. 
*   By definition, every Principal Component is strictly orthogonal (perpendicular/uncorrelated) to every other component, guaranteeing zero multicollinearity in your new feature space.