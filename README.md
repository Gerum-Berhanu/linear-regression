# Demystifying Simple Linear Regression - Building Everything From Scratch

## Overview

Simple linear regression is all about finding equation of a line that best represents a collection of data (input and output pair). Using this line, when a new datapoint is given, we can locate its corresponding coordinate at this line, which happens to be our best guess to what output that datapoint could have. The goal is to find the best parameters (y-intercept $\beta_0$ and slope $\beta_1$) that can be plugged into the below line equation:

$$Y=\beta_0+\beta_1x$$

I downloaded a dataset from *kaggle* entitled *Salary_Data* of shape *(6704, 6)*. It consists of age, gender, education level, job title, years of experience and salary columns, from which, for the purpose of building a simple linear regression model (one input feature), I selected years of experience as the independent variable and salary as the target.

I used NumPy, Pandas, and Matplotlib to handle calculations, data handling, and plotting respectively. Also, I used Mean Absolute Error to evaluate the model's performance.

*models.py* is where I wrote the functionality of the model (from scratch) and other reusable tools. As I progress forward, *models.py* will be a monolithic file where the core of other models-like that of a multi linear one-will be written (unless ofc I change my mind/it gets too intricate).

Here is the [link](https://www.kaggle.com/datasets/mohithsairamreddy/salary-data) to the dataset.

## Markdowns Inside salary.ipynb

Divided by sections, most code cells in the jupyter notebook have markdowns that give some insights of what they intend to do.

## Conclusion

### About the model

The model decided the best parameters to be:
- $\beta_0=57,507.76$
- $\beta_1=7,133.48$

meaning the best fit line is:
$$Y=57,507.76 + 7,133.48x$$

Key insights we get from this formula is:
- If one has zero years of experience (x=0), then their expected salary is $58K.
- A 1 year of experience increase is expected to have a $7K salary gain.

The Mean Absolute Error is *25,390.97* meaning that the overall prediction was off to the real-data by, on average, *$25K*. Even though this might be the best we can do with simple linear regression, when interpreted in real-life, this offset is a big issue/concern for someone who may be using this model to have some insight about, for example, the job market.

### Why is standardization not necessary?

A **closed-form solution** means the parameters for a model can be directly computed using prepared formulas. Linear Regression (simple or multi) has a closed-form solution $\theta=(X^TX)^{-1}X^Ty$ which works regardless of feature scales. So standardization is not required for Linear Regression.

However, much more complex models like Logistic Regression and Neural Networks have no closed-form solutions (parameters can't be isolated using basic algebra). Their parameters are solved by a method we call **iterative optimization** (basically by many trial and error steps). One optimization algorithm is the Gradient Descent.

**Gradient Descent** can converge slowly or inefficiently when features have very different scales. This is where standardization steps up; it fixes the extreme scale problem by bringing all features to roughly the same scale.

Though, it is important to remember that Linear Regression is often solved using Gradient Descent too. In practice, the Normal Equation becomes computationally expensive because inverting a massive matrix takes a lot of processing power. In those cases, we switch to Gradient Descent even for Linear Regression; and at that point, standardization becomes mandatory again.

## Next Steps

1. Learn and implement (from scratch) different kinds of model evaluation (beyond just MAE).
2. Implement *k-fold validation* (bucketing) to try to improve the model's accuracy.

## Quick Proof
A quick proof of mine for how, fundamentally and mathematically, a simple linear regression works (how the parameters are calculated):

<img src="./linear_regression_proof.png" width="70%" style="display: block; margin: 0 auto;" alt="Simple linear regression proof">