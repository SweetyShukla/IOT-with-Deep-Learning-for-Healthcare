# -*- coding: utf-8 -*-
"""
Created on Fri Feb 1 00:44:03 2019

@author: Sweety SHukla
"""
#importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
liver_dataset = pd.read_csv('liver.csv') 

#inspecting the dataset
liver_dataset.info()

#checking for null values
liver_dataset.columns[liver_dataset.isnull().any()].tolist()

#Filling the missing values in the albumin and globulin ratio column with the mean values, ie, 0.6
liver_dataset["Albumin and Globulin Ratio"].fillna("0.6", inplace = True) 

#liver_dataset.isnull().sum()

#Making a dataframe having only the gender in the form of dummy variables, i.e, 1 for male and 0 for female
dataset_gender = pd.get_dummies(liver_dataset['Gender'])
dataset_new = pd.concat([liver_dataset, dataset_gender], axis=1)
Drop_gender = dataset_new.drop(labels=['Gender'],axis=1 )
Drop_gender.columns = ['Age', 'Total_Bilirubin', 'Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin and Globulin Ratio','Male','Female','Dataset']

#Final dataset with gender in 1 column, but encoded in the form of 1 and 0
X = Drop_gender.drop('Dataset',axis=1)

#The output, i.e, 1 if the paerson has liver disease and 0 if the patient does not liver diseases
y = Drop_gender['Dataset']

# splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # test size is 20% and training set is 80% of the dataset.
#Random state 0 will give the same output everytime one runs the code

#Feature scaling the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building the model

#Importing the important libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential() # Initialising the ANN

#Adding the input and the first hidden layer
classifier.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p=0.05)) #Dropout p=0.05 means 10 percent of neurons in the first hidden layer will be randomly disabled, therefore, there will less chances of overfitting

#Adding the second hidden layer
classifier.add(Dense(units = 12, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p=0.05)) 

#Adding the third hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p=0.05))

#Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.add(Dropout(p=0.05))

# compiling ANN
classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the data to the model
classifier.fit(X_train, y_train, batch_size = 4, epochs = 10)

#using the model to predict liver diseases occurence in test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #gives output true if y_pred>0.5 else false

#Creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)












