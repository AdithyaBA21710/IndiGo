import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\adith_a9r1d5f\IndiGo Financials ML Model\IndiGo Financials.csv")
rows=data.drop(columns=['Profit','Quarter'])
labels=data['Profit']

numeric_features = rows.select_dtypes(include='number').columns.tolist()

r_train, r_test, l_train, l_test = train_test_split(rows, labels, test_size=0.2, random_state=42)

ct=ColumnTransformer(transformers=[('Scale',StandardScaler(),numeric_features)])

base_model = Pipeline([
    ("Preprocessor", ct),
    ("Regressor", SVR(kernel='linear',C=90,epsilon=0.05))
])

model = TransformedTargetRegressor(
    regressor=base_model,
    transformer=StandardScaler()
)

model.fit(r_train,l_train)

"""
test = pd.DataFrame({"Operating Revenue":,"Non-Operating Revenue":,"ASK":,"RPK":,"PLF":,"RASK":,"CASK":,"CASK ex-Fuel":,"Fuel Costs":,"Other Costs":,"Fleet":,"USD":,"Profit":})
"""

y_pred = model.predict(r_test)

diff=l_test - y_pred
sum=np.sum(np.abs(diff))
mae=sum/y_pred.size

print (mae)

sum=np.sum((np.abs(diff)).pow(2))
mse=sum/y_pred.size

print(mse)

print(np.sqrt(mse))

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