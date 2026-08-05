import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

data=pd.read_csv(r"C:\Users\adith_a9r1d5f\IndiGo Financials ML Model\IndiGo Financials.csv")
rows=data.drop(columns='Profit')
labels=data['Profit']

ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),['Quarter'])],remainder='passthrough')
encoded_rows=ct.fit_transform(rows)

