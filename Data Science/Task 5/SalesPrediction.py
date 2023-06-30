import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = pd.read_csv("D:\\Oasis\\Data Science\\Saless\\Advertising.csv")

print(data.head())
print(data.shape)
print(data.info())
print("Duplicates:", data.duplicated().sum())
print("Missing values:", data.isnull().sum())

plt.figure(figsize=(4, 4))
sns.scatterplot(data=data, x='TV', y='Sales')
plt.title("TV Advertising vs Sales")
plt.show()
plt.figure(figsize=(4, 4))
sns.scatterplot(data=data, x='Radio', y='Sales')
plt.title("Radio Advertising vs Sales")
plt.show()
plt.figure(figsize=(4, 4))
sns.scatterplot(data=data, x='Newspaper', y='Sales')
plt.title("Newspaper Advertising vs Sales")
plt.show()
plt.figure(figsize=(8, 6))
data[['TV', 'Radio', 'Newspaper']].sum().plot(kind='bar')
plt.title("Total Advertising Expenditure")
plt.xlabel("Media")
plt.ylabel("Expenditure")
plt.show()
labels = ['TV', 'Radio', 'Newspaper']
sizes = data[['TV', 'Radio', 'Newspaper']].sum()
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Advertising Media Expenditure")
plt.axis('equal')
plt.show()
X = data.drop('Sales', axis=1)
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
sale = LinearRegression()
sale.fit(X_train, y_train)
prediction = sale.predict(X_test)
print('MAE:', metrics.mean_absolute_error(prediction, y_test))
print('RMSE:', np.sqrt(metrics.mean_squared_error(prediction, y_test)))
print('R-Squared:', metrics.r2_score(prediction, y_test))
