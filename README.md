# Demystifying Simple Linear Regression - Building Everything From Scratch

## Overview

I downloaded a dataset from *kaggle* entitled *Salary_Data* of shape (6704, 6). It consists of age, gender, education level, job title, years of experience and salary columns, from which, for the purpose of building a simple linear regression model (one input feature), I selected years of experience as the independent variable and salary as the target.

I used NumPy, Pandas, and Matplotlib to handle calculations, data handling, and plotting respectively.

*models.py* is where I wrote the functionality of the model (from scratch) and other reusable tools. As I progress forward, models.py will be the monolithic file where the core of other models-like that of a multi linear one-will be written (unless ofc I change my mind/it gets too intricate).

Here is the [link](https://www.kaggle.com/datasets/mohithsairamreddy/salary-data) to the dataset.