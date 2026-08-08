import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\adith_a9r1d5f\IndiGo Financials ML Model\IndiGo Financials.csv")
rows=data.drop(columns=['Profit','Quarter'])
labels=data['Profit']

numeric_features = rows.select_dtypes(include='number').columns.tolist()

r_train, r_test, l_train, l_test = train_test_split(rows, labels, test_size=0.2, random_state=42)

ct=ColumnTransformer(transformers=
                     [('Scale',Pipeline([("Scaler",StandardScaler()),
                      ("Poly",PolynomialFeatures(degree=2,include_bias=False))]),numeric_features)])


model = Pipeline([
    ("Preprocessor", ct),
    ("Regressor", LinearRegression())
])

model.fit(r_train,l_train)

y_pred = model.predict(r_test)

results = pd.DataFrame({'Actual':l_test,'Predicted':y_pred})
print(results)

plt.scatter(l_test,y_pred)
plt.plot(
    [l_test.min(), l_test.max()],
    [l_test.min(), l_test.max()],
    color='red',
    linestyle='--'
)
plt.show()  
