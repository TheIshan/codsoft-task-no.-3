
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

dataset = pandas.read_csv("C:/Users/dell/Downloads/IRIS.csv")
dataset.head()

dataset.info()

dataset['species'].unique()

x1 = dataset.loc[dataset['species'] == 'Iris-setosa', 'sepal_length']
y1 = dataset.loc[dataset['species'] == 'Iris-setosa', 'sepal_width']
​
x2 = dataset.loc[dataset['species'] == 'Iris-versicolor', 'sepal_length']
y2 = dataset.loc[dataset['species'] == 'Iris-versicolor', 'sepal_width']
​
x3 = dataset.loc[dataset['species'] == 'Iris-virginica', 'sepal_length']
y3 = dataset.loc[dataset['species'] == 'Iris-virginica', 'sepal_width']
​
plt.plot(x1, y1, '.', color = 'crimson')
plt.plot(x2, y2, '.', color = 'green')
plt.plot(x3, y3, '.', color = 'blue')
plt.xlabel('sepal-length')
plt.ylabel('sepal-width')
plt.show()

x1 = dataset.loc[dataset['species'] == 'Iris-setosa', 'petal_length']
y1 = dataset.loc[dataset['species'] == 'Iris-setosa', 'petal_width']
​
x2 = dataset.loc[dataset['species'] == 'Iris-versicolor', 'petal_length']
y2 = dataset.loc[dataset['species'] == 'Iris-versicolor', 'petal_width']
​
x3 = dataset.loc[dataset['species'] == 'Iris-virginica', 'petal_length']
y3 = dataset.loc[dataset['species'] == 'Iris-virginica', 'petal_width']
​
plt.plot(x1, y1, '.', color = 'crimson')
plt.plot(x2, y2, '.', color = 'green')
plt.plot(x3, y3, '.', color = 'blue')
plt.xlabel('petal-length')
plt.ylabel('petal-width')
plt.show()
​

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 6)


knn = KNeighborsClassifier(n_neighbors = 3, weights = 'distance')
dt = DecisionTreeClassifier()
lr = LogisticRegression(solver = 'liblinear')
acc = {}


knn.fit(X_train, y_train)
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)


a,b,c = dt.score(X_test, y_test), lr.score(X_test, y_test), knn.score(X_test, y_test)
acc = pandas.DataFrame({'models' :['DecisionTree', 'LogisticRegression', 'KNN'], 'accuracy': [a, b, c]})
acc

