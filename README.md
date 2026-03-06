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

The model decided the best parameters to be:
- $\beta_0=115385.937$
- $\beta_1=42366.231$

meaning the best fit line is:
$$Y=115385.937 + 42366.231x$$

The Mean Absolute Error is 24458.960 meaning that the overall prediction was off to the real-data by, on average, $24K. Even though this might be the best we can do with simple linear regression, when interpreted in real-life, this offset is a big issue/concern for someone who may be using this model to have some insight about, for example, the job market.

## Notes

- When scaling both *y_train* and *y_test*, the MAE got some small changes to be 24299.501. I thought scaling wouldn't have any effect. Imma find out the reason. My guess is that it has something to do with precision problem of floats.

## Next Steps

1. Learn and implement (from scratch) different kinds of model evaluation (beyond just MAE).
2. Implement *k-fold validation* (bucketing) to try to improve the model's accuracy.

## Quick Proof
A quick proof of mine for how, fundamentally and mathematically, a simple linear regression works (how the parameters are calculated):

![Simple linear regression proof image](./linear_regression_proof.png)