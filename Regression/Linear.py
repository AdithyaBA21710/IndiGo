import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt 

data=pd.read_csv(r"C:\Users\adith_a9r1d5f\IndiGo Financials ML Model\IndiGo Financials.csv")
rows=data.drop(columns=['Profit','Quarter'])
labels=data['Profit']

numeric_features = rows.select_dtypes(include='number').columns.tolist()

ct=ColumnTransformer(transformers=[('Scale',StandardScaler(),numeric_features)])
preprocessed_rows=ct.fit_transform(rows)

r_train, r_test, l_train, l_test = train_test_split(preprocessed_rows, labels, test_size=0.2, random_state=42)

lr=LinearRegression()
lr.fit(r_train,l_train)

y_pred=lr.predict(r_test)

diff=l_test - y_pred
sum=np.sum(np.abs(diff))
mae=sum/y_pred.size

print (mae)

sum=np.sum((np.abs(diff)).pow(2))
mse=sum/y_pred.size

print(mse)

sum=np.sum((np.abs(diff)).pow(2))
mse=sum/y_pred.size

print(mse^1/2)

comparison=pd.DataFrame({'Actual':l_test,'Predicted':y_pred})
print (comparison)

plt.scatter(l_test,y_pred)
plt.plot(
    [l_test.min(), l_test.max()],
    [l_test.min(), l_test.max()],
    color='red',
    linestyle='--'
)
plt.show()