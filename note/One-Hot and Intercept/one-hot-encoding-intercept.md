# One-Hot Encoding and the Intercept

## Overview
When performing one-hot encoding on categorical data for linear regression, keeping all category columns alongside an intercept term creates **perfect multicollinearity** (also known as the "dummy variable trap"). To avoid breaking the mathematical solver, we drop one encoded feature per category. 

Dropping a category does not result in any data loss. Instead, it changes the algebraic representation of the model: the dropped category becomes the **baseline** or **reference group**, and its expected value is absorbed mathematically by the intercept parameter ($b_0$). **All other one-hot coefficients represent the difference (offset) from this baseline.**

## Question 1: Dropping means data loss?

> Does dropping an encoded feature mean we are willingly losing valuable data and hurting the model's accuracy?

**No.** You are not losing any valuable information or negatively impacting the model's accuracy. You are simply changing how the information is represented:
1. **Implicit Knowledge**: If a record has `0` for all other one-hot columns, the model automatically knows it belongs to the dropped category.
2. **Identical Output**: The predictions made by the model will be exactly the same whether you keep all categories (without an intercept) or drop one category (with an intercept).
3. **The Intercept**: The intercept parameter ($b_0$) successfully absorbs the expected value of the dropped category, making it the mathematical foundation of your predictions.

## Question 2: Still dropping when no intercept?

> What if the model does NOT have an intercept term? Do we still drop a column?

**No.** If there is no intercept, you **must NOT drop** a column.
If you removed the intercept *and* dropped a category, the model would have no mathematical way to add a baseline value. It would be forced to predict a value of `0` for the dropped category. 
* **Rule of Thumb:**
  * **With Intercept:** Keep $k - 1$ columns. Let the intercept be the baseline.
  * **Without Intercept:** Keep all $k$ columns. Each category receives its own column and coefficient acting as its direct mean.

## Question 3: How model sets $b_0$ as baseline

> How does the model "know" $b_0$ is the baseline category's expected value (e.g., an Engineer's salary)?

The model doesn't "know" the concepts; it optimizes mathematics using Ordinary Least Squares to minimize error. For the rows belonging to the dropped category, all one-hot features ($x_1, x_2$, etc.) equal `0`. 
The equation simplifies to: `Target = b0 + 0 + 0...`
To make the error as small as possible for these rows, the math forces $b_0$ to equal their average target value.

## Question 4: In the existence of other features

> What if there are continuous features, like `Age`, in the model?

If you introduce a continuous variable, the baseline meaning of $b_0$ shifts. 
Example Equation: `Salary = b0 + b1(Nurse) + b2(Pilot) + b3(Age)`
If someone is in the baseline dropped category (Engineer), the equation is: `Salary = b0 + b3(Age)`.
In this case, $b_0$ specifically represents the expected salary of an Engineer **who is 0 years old**. Since a 0-year-old worker doesn't make sense practically, $b_0$ becomes purely a mathematical anchor point from which the age slope ($b_3$) builds.

## Question 5: Multiple categorical features to encode

> What if there are multiple categorical features (e.g., Job Title and City)? Do we lose data when dropping a baseline from both?

No data is lost. If you drop **Engineer** from Job Title and **Paris** from City, the intercept ($b_0$) becomes the **grand baseline**.
* $b_0$ represents the expected salary of an **Engineer living in Paris**. 
* The one-hot columns are structural offsets built on top of this foundation. 
  * A Nurse in Paris: `Salary = b0 + b_nurse`
  * An Engineer in London: `Salary = b0 + b_london`
  * A Pilot in NY: `Salary = b0 + b_pilot + b_ny`
Every variable adjusts mathematically from that pure foundation state.
