import pandas as pd
import seaborn as sns
import subprocess
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = pd.read_csv("D:\Oasis\Data Science\Flower\Iris.csv")

print(iris.head())

output_file = "iris_dataset.txt"
iris.to_csv(output_file, index=False)

subprocess.Popen(['notepad.exe', output_file])

X = iris.drop(['Id', 'Species'], axis=1)
y = iris['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

plt.scatter(iris["SepalLengthCm"], iris["SepalWidthCm"])
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()

sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, height=5)
plt.show()

sns.scatterplot(x="SepalLengthCm", y="SepalWidthCm", hue="Species", data=iris)
plt.show()

sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
plt.show()

sns.violinplot(x="Species", y="PetalLengthCm", data=iris, height=6)
plt.show()

sns.pairplot(iris.drop("Id", axis=1), hue="Species", height=3)
plt.show()

iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))
plt.show()

