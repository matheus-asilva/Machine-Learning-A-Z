#Data Preprocessing

#1 Importing the libraries
import numpy as np #mathematical tools
import matplotlib.pyplot as plt #plot charts
import pandas as pd #manage datasets
import os
np.set_printoptions(threshold=np.nan) # To print all values of an array when entered <array_name> in the console

#Changing the main diretory
os.chdir('C:\\Users\\Matheus\\Documents\\GitHub\\Machine-Learning-A-Z\\Machine Learning A-Z\\Part 1 - Data Preprocessing')

#2 Importing the dataset
dataset = pd.read_csv('Data.csv')

#Creating the Matrix of Features - Independent Variable Vector
X = dataset.iloc[:,:-1].values #iloc is for independent features
#Used [:,:-1] to remove the last column (Purchased)

#Creating the Matrix of Results- Dependent Variable Vector
Y = dataset.iloc[:,3].values
                
#3 Working with missing data
"""
In the dataset, we have missing data for Spain (age)
and for Germany (salary).
Ways to handle with this:
1- Remove the lines where are some missing data
    It's not so secure because we can find some crucial information
2- Take the mean of columns (most usual)
    Replace the missing information with the mean of all column's value
"""
from sklearn.preprocessing import Imputer #sklearn is a library for machine learning models and preprocess module contains a lot of classes, methods to preprocess any dataset
imputer = Imputer(missing_values='NaN', strategy="mean",axis=0)
imputer = imputer.fit(X[:,1:3]) # : to "catch" all the lines ; Columns 1 and 2 have missing data, so that's why 1:3 which 3 is not inclusive
X[:, 1:3] = imputer.transform(X[:, 1:3]) #transform method replace the missing data

#4 Categorical Data