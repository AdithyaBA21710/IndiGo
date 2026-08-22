import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv (r"C:\Users\adith_a9r1d5f\IndiGo Financials ML Model\IndiGo Financials.csv")
features = dataset.iloc[:,1:-1].values
labels = dataset.iloc[:,-1].values


r_train, r_test, y_train, y_test = train_test_split(features,labels, random_state=42, test_size=0.2)

rsc = StandardScaler()
ysc = StandardScaler()

r_train=rsc.fit_transform(r_train)
r_test = rsc.transform(r_test)
y_train = ysc.fit_transform(y_train.reshape(-1, 1))

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=90, activation='relu'))
ann.add(tf.keras.layers.Dense(units=80, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))

ann.compile(optimizer='adam', loss = 'mean_squared_error')

ann.fit(r_train, y_train, epochs=100, batch_size = 8)

pred = ann.predict(r_test)
pred = ysc.inverse_transform(pred).flatten()

diff=y_test - pred
sum=np.sum(np.abs(diff))
mae=sum/pred.size

print (mae)

sum=np.sum((np.abs(diff))**2)
mse=sum/pred.size

print(mse)

print(np.sqrt(mse))

results = pd.DataFrame({'Actual':y_test,'Predicted':pred.flatten()})
print(results)