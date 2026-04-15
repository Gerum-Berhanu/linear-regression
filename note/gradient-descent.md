# Briefly Explaining Gradient Descent

## Univariate Functions

The aim of gradient descent is to find the $x$ value for which $\frac{df}{dx}=0$ or near-zero for the funciton $f(x)$; specifically for bowl shaped $f$.

Here is a step-by-step guide for how this algorithm works:
1. Take a guess for $x$.
2. Calculate the slope at that point.
3. The next guess is calculated by: $\boxed{x_\text{next} = x_\text{current} - \text{slope} \cdot \text{lr}}$, where $\text{lr}$ is a pre-assigned learning rate.

Do these steps in a loop until the next guess is near-equal to the current one. This means, the $\text{slope} \cdot \text{lr}$ part is near-zero, and the way this becomes near-zero is when the slope itself is near-zero; meaning we've achieved our goal of finding the place where the slope is near-zero, if not exact-zero.

For a multivariate function, everything we do before applies the same for each variables with respect to their individual partial slopes.

## Multivariate Functions

The aim of gradient descent is to find the $x_1, x_2, \dots, x_n$ value for which $\frac{\partial f}{\partial x_1}=0, \frac{\partial f}{\partial x_2}=0, \dots, \frac{\partial f}{\partial x_n}=0$ or near-zero for the funciton $f$; specifically for a bowl shaped $f$ with one global minimum.

Here is a step-by-step guide for how this algorithm works:
1. Take initial guesses for $x_1, x_2, \dots, x_n$.
2. Calculate the partial slope at each point.
3. The next guess is calculated by: 
    $$x_\text{next}^{(1)} = x_\text{current}^{(1)} - \text{slope}_1 \cdot \text{lr} \\
    x_\text{next}^{(2)} = x_\text{current}^{(2)} - \text{slope}_2 \cdot \text{lr} \\
    \vdots \\
    x_\text{next}^{(n)} = x_\text{current}^{(n)} - \text{slope}_n \cdot \text{lr}$$
    where $\text{lr}$ is a pre-assigned learning rate.

Do these steps in a loop until the next guess is near-equal to the current one. This means, the $\text{slope} \cdot \text{lr}$ part is near-zero, and the way this becomes near-zero is when the slope itself is near-zero; meaning we've achieved our goal of finding the place where the slope is near-zero, if not exact-zero.

## The Learning Rate

What is it about the learning rate? Without the learning rate, $x_\text{next} = x_\text{current} - \text{slope}$, each guess has equal constant gaps $|slope|$. This is bad and mostly can't achieve our goal. With the application of learning rate, a value between 0 and 1, we manage the gaps to narrow as we approach the target.

So, how to choose one? There is no one universal (standard) learning rate. We choose the one that best suits our algorithm; in other words, the one that helps our gradient descent algorithm converge to the target point (the zero slope). For example, a 0.1 lr may take the next guess at some point to jump over the zero-point, leading to divergence. 0.001 lr may converge, but could be too slow (taking forever). So, maybe 0.01 is the best lr.

To choose the learning rate that best suits our function, we once again follow an iterative approach. Kind of like a brute-force appraoch; we checkout for multiple lr values, like 1, 0.1, 0.01, 0.001... and choose the best one. The best one is the one that takes the lowest number of iterations to reach the target point. More technically, the lowest number of iterations to take $c$ near to zero, where $c$ is the step size $c = \text{slope}\cdot\text{lr}$ so that in the next guess formula $x_\text{next} = x_\text{current} - c$, $c$ becomes neglibile, resulting in near-same next guess suggesting that we have got the input value for which output has a near-zero slope.

## Gradient Descent in Linear Regression

For a simple linear regression $\bar{y}=mx+b$, we aim to find the best fit $m$ and $b$. Which means, techniqually speaking, they are our variables. So, our function on which we perfom gradient descent is not $\bar{y}$, but the sum of squared errors:
    $$SSE = f(m, b) = \sum_{i=1}^n(y_i - \bar{y}_i)^2 = \sum_{i=1}^n(y_i - mx_i - b)^2$$

The partial slopes with respect to $m$ and $b$ are:
    $$\frac{\partial f}{\partial b} = -2\sum(y - mx - b) \qquad
    \frac{\partial f}{\partial m} = -2\sum(x)(y - mx - b)$$

With the closed-form solutions approach, we can directly set the partial derivates to zero and find out the values for $m$ and $b$, but that is not what we want do now. Right now, we want to see how gradient descent works in linear regression.

Once we find out the partial derivatives, applying gradient descent is easy; just follow the aforementioned three steps.

1. Take initial guesses, say $b=0$ and $m=0$.
2. Calculate each of their partial slopes at $b=0$ and $m=0$
3. next guesses are:
    $b_\text{next} = 0 - \frac{\partial f}{\partial b=0}\cdot lr$
    $m_\text{next} = 0 - \frac{\partial f}{\partial m=0}\cdot lr$

We do this until our $c$ values $c_b=\frac{\partial f}{\partial b}\cdot lr$ and $c_m=\frac{\partial f}{\partial m}\cdot lr$ are so close to zero. In the end, we take the final next guesses for both $b$ and $m$ as the best fit parameters. Because, remember that the goal of linear regression is to find the input values for which the sum of squared errors (loss function) is zero slope.

Gradient descent has a more significant impact on multivariate linear regressions because there, mostly in real world, the closed-form solutions require high computation power. One thing about it is that it doesn't care about linear dependencies unlike $\theta = (X^TX)^{-1}X^Ty$. With gradient descent, we don't have to worry about non-invertible matrices, near-zero determinant and such edge-cases.

One thing we do need to worry though is about the range (scale) of the input data. For a dataset containing age and salary, for example, the age feature is usually in the range between 1-100, while salary is between say 1,000-100,000. We use the same learning rate for all parameters of the features. And that's where we find a critical issue. The learning rate my converge the age feature but may not work for the salary. We can find different learning rates tailored to each features, or we can standardize (scale down to a specific range) the entire dataset and use one learning rate for all.