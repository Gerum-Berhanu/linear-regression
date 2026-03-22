# Why does strongly (if not perfectly) correlated columns (features) result in a serious instability in the model?

## *Overview*
For any matrix A that is at least 2×2, if any two columns are perfectly linearly related (i.e., one is a scalar multiple of the other, regardless of their positions in the matrix), then the determinant of AᵀA is zero.

Geometrically: when the columns of A are viewed as vectors in the same space, linear dependence between two of them means one is simply a scaled version of the other — they lie along the same line through the origin. In this case, the columns fail to span a full-dimensional space because they span zero area (in 2D), zero volume (in 3D), or zero determinant (in higher dimensions), causing the Gram matrix AᵀA to become singular.

No matter how many columns you have (3, 4, 10, …), as soon as even one column can be made by adding and scaling some of the other columns, the columns become linearly dependent → det(A) = 0 → det(AᵀA) = 0.

However, if two columns are nearly linearly dependent — meaning one can be expressed approximately as a scaled version of the other plus small random noise/error — then the determinant of the Gram matrix AᵀA becomes very small (close to zero, but not exactly zero). This makes AᵀA ill-conditioned: its inverse has very large numbers, and even tiny changes in the data can completely change the computed inverse. That leads to unstable and unreliable results.
## *Question 1*
> If you just change one or two rows in your dataset (like splitting it into Train vs Test), that microscopic error changes slightly, and your multiplier might abruptly flip from +1,000,000,000 to −1,000,000,000.

To understand why this happens, let's look at the math with a concrete example. 

Remember that calculating your parameters ($\theta$) requires finding the inverse of the matrix $X^T X$, which involves dividing by its determinant:

$$ (X^T X)^{-1} = \frac{1}{\det(X^T X)} \times \text{Adjugate}(X^T X) $$

Imagine you have two highly correlated features (like $X_1$ and $X_2$). Because they are almost identical, their $X^T X$ matrix looks something like this:

$$ X^T X = \begin{pmatrix} 1 & 1 \\ 1 & 1.0001 \end{pmatrix} $$

The determinant is calculating the diagonal difference: $(1 \times 1.0001) - (1 \times 1) = \mathbf{0.0001}$.
So, your multiplier is $\frac{1}{0.0001} = \mathbf{10,000}$. 
Your model computes $\theta$ by multiplying the rest of the equation by **$+10,000$**.
## *Now, you change one row of your data*
Suppose you move one slightly unusual row out of your training set and into your test set. Now, $X_2$ is no longer $0.0001$ units larger than $X_1$ on average; it is $0.0001$ units *smaller*.

Your new $X^T X$ matrix becomes:
$$ X^T X = \begin{pmatrix} 1 & 1 \\ 1 & 0.9999 \end{pmatrix} $$

The new determinant is: $(1 \times 0.9999) - (1 \times 1) = \mathbf{-0.0001}$.
Your new multiplier is $\frac{1}{-0.0001} = \mathbf{-10,000}$.
Your model now computes $\theta$ by multiplying the exact same equation by **$-10,000$**.
## *The Result*
Because the intrinsic difference between the two features was so incredibly small, removing a single data point was enough to flip the relationship (the determinant) from slightly positive to slightly negative.

When you divide by that tiny baseline, your multiplier violently swings from $+10,000$ to $-10,000$. Consequently, your $\theta$ coefficients (which might have been $+50,000$ and $-49,000$) will abruptly flip signs and magnitudes, creating a completely unstable, unusable model.
## *Question 2*
> So, it's just the fear of a sudden sign flip and nothing else. But if we don't change any thing from the training data, everything is going to be fine, right?

Mathematically, the model will output correct predictions for the exact training data it memorized. However, "everything is fine" is incorrect for two major reasons once you actually try to use the model:
## 1. *It Destroys Real-World Predictions (Overfitting)*
You build a model to predict **new, unseen data**. Even if the training data doesn't change, the real-world data you feed into it later will have slight natural variations.

Imagine your collinear model learned these extreme coefficients based on your training data to achieve a prediction of $100$:
*   $\theta_1 = 1,000,000$
*   $\theta_2 = -999,900$

For a training instance where $X_1 = 1$ and $X_2 = 1$:
$$ Y = (1,000,000 \times 1) + (-999,900 \times 1) = 100 $$

Now, a new instance comes in from the real world. Let's say there is a microscopic, normal variance of $0.0001$ in the second feature:
*   $X_1 = 1$
*   $X_2 = 1.0001$

Let's see what your model predicts:
$$ Y = (1,000,000 \times 1) + (-999,900 \times 1.0001) $$
$$ Y = 1,000,000 - 1,000,000.09 = -0.09 $$

A fraction of a decimal of normal real-world variance caused your prediction to crash from $100$ down to $0$. The model completely shatters the moment it touches non-training data.
## *2. Loss of Interpretability*
In Linear Regression, coefficients are supposed to have real-world meaning. $\theta_1 = 10$ means "for every 1 unit increase in $X_1$, $Y$ increases by 10".

When collinearity forces $\theta$ into wildly inflated positive and negative numbers, the coefficients lose all logical meaning. You can no longer analyze your model to understand which features actually drive the target variable.
## *3. Floating-Point Limits*
Finally, computers cannot hold infinite decimal places. When calculating inverses of matrices with near-zero determinants, the computer rounds off microscopic decimals. This means your $\theta$ parameters might purely be the result of a CPU rounding error rather than actual data trends.
## *A Quick Illustration*
- Exact dependent (scalar of 3):

$$
A = \begin{bmatrix}
1 & 2 \\
3 & 6
\end{bmatrix}
$$

$$
\det(A) = 6 - 6 = 0
$$

- Near dependence (add small noise $\varepsilon \approx 0.001$)

    $$
    B = \begin{bmatrix}
    1 & 2.001 \\
    3 & 6.002
    \end{bmatrix}
    $$

    $$
    \det(A) = 6.002 - 6.003 = -0.001
    $$

    $$
    (B^TB)^{-1} = \frac{1}{-0.001} \left( \begin{bmatrix}
    6.002 & -2.001 \\
    -3 & 1
    \end{bmatrix} \right) = \begin{bmatrix}
    -6002 & 2001 \\
    3000 & -1000
    \end{bmatrix}
    $$

    - If instead:

    $$
    C = \begin{bmatrix}
    1 & 2.002 \\
    3 & 6.001
    \end{bmatrix}
    $$

    $$
    \det(A) = 6.001 - 6.006 = -0.006
    $$

    $$
    (C^TC)^{-1} = \frac{1}{-0.006} \left( \begin{bmatrix}
    6.001 & -2.002 \\
    -3 & 1
    \end{bmatrix} \right) = \begin{bmatrix}
    -1000.33 & 333.5 \\
    500 & -166.67
    \end{bmatrix}
    $$

    - Another example:

    $$
    D = \begin{bmatrix}
    1 & 2 \\
    3 & 6.002
    \end{bmatrix}
    $$

    $$
    \det(A) = 6.002 - 6 = 0.002
    $$

    $$
    (D^TD)^{-1} = \frac{1}{0.002} \left( \begin{bmatrix}
    6.002 & -2 \\
    -3 & 1
    \end{bmatrix} \right) = \begin{bmatrix}
    3001 & -1000 \\
    -1500 & 500
    \end{bmatrix}
    $$