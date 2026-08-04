import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

data=pd.read_csv(r"C:\Users\adith_a9r1d5f\IndiGo Financials ML Model\IndiGo Financials.csv")
rows=data.iloc[:,:-1].values
labels=data.iloc[:,-1].values

ct=ColumnTransformer(transformers=[('encode',OneHotEncoder(),[0])],remainder='passthrough')
x=ct.fit_transform(rows).toarray()

encoded_df = pd.DataFrame(x, columns=ct.get_feature_names_out())
print(encoded_df.head())