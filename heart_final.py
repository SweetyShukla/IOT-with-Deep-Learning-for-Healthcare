#importing the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
heart_dataset = pd.read_csv('heart1.csv') 


#checking for null values
heart_dataset.columns[heart_dataset.isnull().any()].tolist()


X = heart_dataset.iloc[:, 0:13].values
y = heart_dataset.iloc[:, 13:14].values


# splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) # test size is 30% and training set is 70% of the dataset.
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
classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
classifier.add(Dropout(rate=0.05)) #Dropout rate=0.05 means 5 percent of neurons in the first hidden layer will be randomly disabled, therefore, there will less chances of overfitting

#Adding the second hidden layer
classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(rate=0.05)) 

#Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.add(Dropout(rate=0.05))

# compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the data to the model
classifier.fit(X_train, y_train, batch_size = 4, epochs = 120)

#using the model to predict heart diseases occurence in test set
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5) #gives output true if y_pred>0.5 else false

#Creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)












