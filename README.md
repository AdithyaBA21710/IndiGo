# IndiGo Financials ML Model

Regression models to predict IndiGo's quarterly profit using historical financial and operational data.

## Models

This project implements and compares three regression models:

1. Linear Regression
2. Polynomial Regression
3. Support Vector Regression (SVR)

## Dataset

The dataset contains quarterly IndiGo financial and operational information.

### Features

- `Quarter` - The 3 month time period used to track the financial performance of 6E
- `Operating Revenue` - Revenue earned from day-to-day flying operations of the airline
- `Non-Operating Revenue` - Revenue not earned from operations (Money earned from Sale-Leasback of aircraft etc.)
- `ASK` - Available Seat Kilometers (Total capacity offered)
- `RPK` - Revenue Passenger Kilometers (Total passenger traffic)
- `PLF` - Passenger Load Factor (in %)
- `RASK` - Revenue Per Available Seat Kilometer (How much money an airline generates for every seat flown one kilometer). Higher the better
- `CASK` - Cost Per Available Seat Kilometer (Unit cost or operational efficiency). Lower the better
- `CASK ex-Fuel` - Cost Per Available Seat Kilometer excluding fuel costs
- `Fuel Costs` - Self explanatory
- `Other Costs` - Cost of new aircraft, spares purchased, airport fees etc..  
- `Fleet` - Size of fleet (ATR, A320 family, 737, 777 & 787 have been part of IndiGo's fleet. Most owned/dry-lease, some on wet-lease)
- `USD` - Exchange rate of USD with INR

### Target

- `Profit` - Net money made during the period

![alt text](image.png)
<p align="center">Graphs showing how different features influence Profit</p>  

## Preprocessing

`Quarter` is a categorical feature and can be one-hot encoded. Since Quarter does not affect the finance of a company, the feature has been dropped.

Preprocessing is performed with a `ColumnTransformer` and `Pipeline` so that transformations learned from the training data are reused consistently during testing and prediction.

The general workflow is:

```text
Raw Data
   |
   +-- Numeric Features --> StandardScaler
   |
   v
Processed Features
   |
   v
Regression Model
   |
   v
Predicted Profit
```

## 1. Linear Regression

Linear Regression is used as the baseline model.

```python
from sklearn.linear_model import LinearRegression

model = Pipeline([
    ("Preprocessor", preprocessor),
    ("Regressor", LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Linear Regression assumes a linear relationship between the input features and the target.

<p align="center">![alt text](image-1.png)</p>
<p align="center">Graph showing how Linear Regression fits the test set</p>

## 2. Polynomial Regression

Polynomial Regression extends Linear Regression by generating polynomial and interaction features.

The current implementation uses degree 2:

```python
PolynomialFeatures(
    degree=2,
    include_bias=False
)
```

The numerical features are standardized and then expanded into polynomial features.

```python
model = Pipeline([
    ("Preprocessor", preprocessor),
    ("Regressor", LinearRegression())
])
```

Polynomial Regression can model nonlinear relationships, but it can also produce many features and overfit small datasets. Therefore, higher polynomial degrees should be used carefully.

<p align="center">![alt text](image-3.png)</p>
<p align="center">Graph showing how Polynomial Regression fits the test set</p>

## 3. Support Vector Regression

Support Vector Regression is used as a nonlinear regression model.

A typical implementation is:

```python
from sklearn.svm import SVR

model = Pipeline([
    ("Preprocessor", preprocessor),
    ("Regressor", SVR(
        kernel="linear",
        C=90,
        epsilon=0.05
    ))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

SVR is sensitive to feature scale, so numerical features are standardized before training.

The RBF kernel allows SVR to learn nonlinear relationships without explicitly generating polynomial features.

<p align="center">![alt text](image-2.png)</p>
<p align="center">Graph showing how SVR fits the test set</p>

## Model Evaluation

The models are evaluated using MAE, MSE and RMSE.

## Current Results

On the evaluated test split, the current results were:

| Model | MAE | RMSE |
|---|---:|---:|
| Linear Regression | ₹2.91B | ₹3.17B |
| Polynomial Regression | ₹4.22B | ₹6.26B |
| SVR | ₹1.42B | ₹1.74B |

Based on the current test split, SVR performed better than Linear & Polynomial Regression.

These results are not definitive because the dataset contains only approximately 42 observations. A different train/test split can produce different results.

## Installation

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Running the Models

1. Place the dataset in the expected location.
2. Install the required Python packages.
3. Run each model script.
4. Record MAE, MSE and RMSE.
5. Compare Actual vs Predicted plots.

Example:

```bash
python "Linear Regression.py"
python "Polynomial Regression.py"
python "SVR.py"
```

## Important Considerations

### Small Dataset

The dataset contains approximately 42 observations. This is very small for machine learning, particularly for models that generate many features.

Polynomial Regression can generate a large number of features from the numerical variables, increasing the risk of overfitting.

### Train/Test Splitting

The current experiments use a train/test split. Cross-validation should be considered for more reliable model comparison because a single split can produce unstable results with a small dataset.

### Time-Series Structure

The data is quarterly and therefore has a temporal structure. A random train/test split may not accurately represent the real-world task of predicting future quarters.

A future version should consider a chronological split, where earlier quarters are used for training and later quarters are used for testing.

## Disclaimer

This project is an educational machine learning experiment based on historical data. Predictions should not be treated as financial advice or as an official forecast of IndiGo's future financial performance.
