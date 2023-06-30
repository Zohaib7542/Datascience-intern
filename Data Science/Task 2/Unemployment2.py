import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = pd.read_csv("D:\\Oasis\\Data Science\\Unemployment\\Unemployment_Rate_upto_11_2020.csv")

print(data.head())
print(data.isnull().sum())

data.columns = ["State", "Date", "Frequency", "Estimated Unemployment Rate (%)",
                "Estimated Employed", "Estimated Labour Participation Rate (%)", "Region", "Area", "Category"]

numeric_columns = ["Estimated Unemployment Rate (%)", "Estimated Employed", "Estimated Labour Participation Rate (%)"]
numeric_data = data[numeric_columns]
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
plt.figure(figsize=(12, 10))
plt.title("Distribution of Estimated Employed by Region")
sns.histplot(x="Estimated Employed", hue="Region", data=data)
plt.xlabel("Estimated Employed")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12, 10))
plt.title("Distribution of Estimated Unemployment Rate by Region")
sns.histplot(x="Estimated Unemployment Rate (%)", hue="Region", data=data)
plt.xlabel("Estimated Unemployment Rate")
plt.ylabel("Count")
plt.show()

aggregated_data = data.groupby(["Region", "State"], as_index=False)["Estimated Unemployment Rate (%)"].mean()

fig = px.sunburst(aggregated_data, path=["Region", "State"], values="Estimated Unemployment Rate (%)",
                  color_continuous_scale="RdYlGn", title="Unemployment Rate in India")
fig.show()
