#data preprocessing 

#Importing libraries
import numpy as np #arithmetics
import matplotlib.pyplot as plt #plotting graphs
import pandas as pd #importing Data

#Importing dataset
dataset=pd.read_csv('___') #load dataset
X = dataset.iloc[:,:-1].values #load independent values
Y = dataset.iloc[:,3].values #dependent values

#Dealing with missing values
from sklearn.preprocessing import Imputer #import the Imputer class
myImputer = Imputer() #create a new object (Imputer)
myImputer = myImputer.fit(X[:,1:3]) #Fit the raw data
X[:,1:3] = myImputer.transform(X[:,1:3]) #Fill in the empty spaces

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoderX = LabelEncoder()
X[:,0] = labelencoderX.fit_transform(X[:,0])
onehotencoderX = OneHotEncoder(categorical_features = [0])
X = onehotencoderX.fit_transform(X).toarray()
labelencoderY = LabelEncoder()
Y = labelencoderY.fit_transform(Y) 

#creating training and test sets
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler 
scX = StandardScaler()
Xtrain = scX.fit_transform(Xtrain)
Xtest = scX.transform(Xtest)
