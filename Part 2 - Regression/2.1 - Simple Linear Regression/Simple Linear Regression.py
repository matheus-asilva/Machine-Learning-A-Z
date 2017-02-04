#Simple Linear Regression

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#Changed the main diretory
os.chdir('C:\\Users\\Matheus\\Documents\\GitHub\\Machine-Learning-A-Z\\Part 2 - Regression\\2.1 - Simple Linear Regression')
os.getcwd()

#Data Preprocessing
dataset = pd.read_csv('Salary_Data.csv')
years = dataset.iloc[:,:-1].values               #Created years vector for x axis
salary = dataset.iloc[:,1].values             #Created salary vector for y axis

#Split the dataset into Training set and Test set                
from sklearn.cross_validation import train_test_split      #library to split training and test sets
years_train, years_test, salary_train, salary_test = train_test_split(years, salary, test_size=1/3, random_state=0)

#FOR SIMPLE LINEAR REGRESSION, WE DO NOT NEED FEATURE SCALING

#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(years_train, salary_train)

#Predicting the Test set results
salary_pred = regressor.predict(years_test)    #array of predicted values

#Visualizing the Training set resuts
plt.scatter(years_train, salary_train, c='r')
plt.plot(years_train, regressor.predict(years_train), c='b')                               
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test set resuts
plt.scatter(years_test, salary_test, c='r')
plt.plot(years_train, regressor.predict(years_train), c='b')                               
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()