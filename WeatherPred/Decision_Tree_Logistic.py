import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing data
data = pd.read_csv('C:\\GitHub\\Misha_Jain\\Task_4\\weatherAUS.csv')
x = data[['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm']]
y = data['RainTomorrow']
y = y.map({'Yes': 1, 'No': 0})
x = x.fillna(x.mean())
y = y.fillna(1)

# Splitting data into training data and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state= 42)

# Creates decision tree from x_train and y_train values
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

# Testing x_test
predictions = clf.predict(x_test)

# Getting accuracy
from sklearn.metrics import accuracy_score
print ("Accuracy: ", accuracy_score(y_test, predictions) * 100, "%")