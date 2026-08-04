import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\adith_a9r1d5f\IndiGo Financials ML Model\IndiGo Financials.csv")
fig, ax = plt.subplots(4, 3, figsize=(15,15))
ax=ax.flatten()

a=data['Profit']

column1 = data.columns.drop(["Profit","Quarter"])

for i,v in enumerate(column1):
    ax[i].scatter(a,data[v])
    ax[i].set_title(v+" vs Profit")
    ax[i].set_xlabel("Profit")
    ax[i].set_ylabel(v)

plt.tight_layout(pad=5.0)
plt.show()