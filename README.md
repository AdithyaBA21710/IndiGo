# IndiGo Financials ML Model

Regression models to predict IndiGo's quarterly profit using historical
financial and operational data.

## Models

This project implements and compares four regression models:

1.  Linear Regression
2.  Polynomial Regression
3.  Support Vector Regression (SVR)
4.  Artificial Neural Network (ANN)

## Dataset

The dataset contains quarterly IndiGo financial and operational
information.

### Features

-   `Quarter` - The 3 month time period used to track the financial
    performance of 6E
-   `Operating Revenue` - Revenue earned from day-to-day flying
    operations of the airline
-   `Non-Operating Revenue` - Revenue not earned from operations (Money
    earned from Sale-Leasback of aircraft etc.)
-   `ASK` - Available Seat Kilometers (Total capacity offered)
-   `RPK` - Revenue Passenger Kilometers (Total passenger traffic)
-   `PLF` - Passenger Load Factor (in %)
-   `RASK` - Revenue Per Available Seat Kilometer (How much money an
    airline generates for every seat flown one kilometer). Higher the
    better
-   `CASK` - Cost Per Available Seat Kilometer (Unit cost or operational
    efficiency). Lower the better
-   `CASK ex-Fuel` - Cost Per Available Seat Kilometer excluding fuel
    costs
-   `Fuel Costs` - Self explanatory
-   `Other Costs` - Cost of new aircraft, spares purchased, airport fees
    etc..\
-   `Fleet` - Size of fleet (ATR, A320 family, 737, 777 & 787 have been
    part of IndiGo's fleet. Most owned/dry-lease, some on wet-lease)
-   `USD` - Exchange rate of USD with INR

### Target

-   `Profit` - Net money made during the period

![alt text](README_Assets/Graphs.png)

```{=html}
<p align="center">
```
Graphs showing how different features influence Profit
```{=html}
</p>
```
## Preprocessing

`Quarter` is a categorical feature and can be one-hot encoded. Since
`Quarter` does not affect the finance of a company, the feature has been
dropped.

Preprocessing is performed with a `ColumnTransformer` and `Pipeline` so
that transformations learned from the training data are reused
consistently during testing and prediction.

The general workflow is:

``` text
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

``` python
from sklearn.linear_model import LinearRegression

model = Pipeline([
    ("Preprocessor", preprocessor),
    ("Regressor", LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Linear Regression assumes a linear relationship between the input
features and the target.

![alt text](README_Assets/LR.png)

```{=html}
<p align="center">
```
Graph showing how Linear Regression fits the test set
```{=html}
</p>
```
## 2. Polynomial Regression

Polynomial Regression extends Linear Regression by generating polynomial
and interaction features.

The current implementation uses degree 2:

``` python
PolynomialFeatures(
    degree=2,
    include_bias=False
)
```

The numerical features are standardized and then expanded into
polynomial features.

``` python
model = Pipeline([
    ("Preprocessor", preprocessor),
    ("Regressor", LinearRegression())
])
```

Polynomial Regression can model nonlinear relationships, but it can also
produce many features and overfit small datasets. Therefore, higher
polynomial degrees should be used carefully.

![alt text](README_Assets/PR.png)

```{=html}
<p align="center">
```
Graph showing how Polynomial Regression fits the test set
```{=html}
</p>
```
## 3. Support Vector Regression

Support Vector Regression is used as a nonlinear regression model.

A typical implementation is:

``` python
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

SVR is sensitive to feature scale, so numerical features are
standardized before training.

The RBF kernel allows SVR to learn nonlinear relationships without
explicitly generating polynomial features.

![alt text](README_Assets/SVR.png)

```{=html}
<p align="center">
```
Graph showing how SVR fits the test set
```{=html}
</p>
```
## 4. Artificial Neural Network

An Artificial Neural Network (ANN) is used as a nonlinear regression
model to predict quarterly profit.

The current implementation uses two hidden layers with ReLU activation:

``` text
Input Features
      |
Dense(90, ReLU)
      |
Dense(80, ReLU)
      |
Dense(1)
      |
Predicted Profit
```

The numerical input features and training target are standardized using
`StandardScaler`. The model uses an 80/20 train/test split with
`random_state=42`, the Adam optimizer, mean squared error loss, 100
epochs, and a batch size of 8.

``` python
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=90, activation='relu'))
ann.add(tf.keras.layers.Dense(units=80, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))

ann.compile(
    optimizer='adam',
    loss='mean_squared_error'
)
```

Predictions are inverse-transformed back to the original Profit scale
before evaluation.

The latest ANN run achieved:

-   **MAE:** ₹3.38B
-   **MSE:** `2.41 × 10^19`
-   **RMSE:** ₹4.90B

The ANN performed better than Polynomial Regression on the reported test
split, but worse than Linear Regression and SVR.

## Model Evaluation

The models are evaluated using MAE, MSE and RMSE.

## Current Results

On the evaluated test split, the current results were:

  Model                        MAE     RMSE
  ----------------------- -------- --------
  SVR                       ₹1.42B   ₹1.74B
  Linear Regression         ₹2.91B   ₹3.17B
  ANN                       ₹3.38B   ₹4.90B
  Polynomial Regression     ₹4.22B   ₹6.26B

Based on the current test split, SVR performed best, followed by Linear
Regression, ANN, and Polynomial Regression. The ANN improved over
Polynomial Regression but did not outperform SVR or Linear Regression.

These results are not definitive because the dataset contains only
approximately 42 observations. A different train/test split can produce
different results.

## Installation

Install the required packages:

``` bash
pip install pandas numpy scikit-learn matplotlib
```

## Running the Models

1.  Place the dataset in the expected location.
2.  Install the required Python packages.
3.  Run each model script, including the ANN implementation.
4.  Record MAE, MSE and RMSE.
5.  Compare Actual vs Predicted plots.

Example:

``` bash
python "Linear Regression.py"
python "Polynomial Regression.py"
python "SVR.py"
python "ANN.py"
```

## Important Considerations

### Small Dataset

The dataset contains approximately 42 observations. This is very small
for machine learning, particularly for models that generate many
features.

Polynomial Regression can generate a large number of features from the
numerical variables, increasing the risk of overfitting.

The ANN also has relatively high model capacity compared with the very
small dataset, so its results may be sensitive to the train/test split
and training configuration.

### Train/Test Splitting

The current experiments use a train/test split. Cross-validation should
be considered for more reliable model comparison because a single split
can produce unstable results with a small dataset.

### Time-Series Structure

The data is quarterly and therefore has a temporal structure. A random
train/test split may not accurately represent the real-world task of
predicting future quarters.

A future version should consider a chronological split, where earlier
quarters are used for training and later quarters are used for testing.

### ANN Result Note

The reported ANN result is from the current 80/20 split with
`random_state=42` and the current ANN configuration. Because the dataset
contains only approximately 42 observations, the ANN result should not
be treated as definitive. For a stronger comparison, all four models
should be evaluated on exactly the same split and, ideally, with
cross-validation or a chronological evaluation that respects the
quarterly time-series structure.

## Disclaimer

This project is an educational machine learning experiment based on
historical data. Predictions should not be treated as financial advice
or as an official forecast of IndiGo's future financial performance.
