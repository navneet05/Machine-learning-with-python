# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:04:47 2020

@author: hp
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

my_data = pd.read_csv("drug200.csv")
my_data[0:5]
X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]

#handling categorial values
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 
le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])
le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 
X[0:5]
y = my_data["Drug"]
y[0:5]

#spliting data into training and testing 
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
#model
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree
drugTree.fit(X_trainset,y_trainset)
#prediction
predTree = drugTree.predict(X_testset)
print (predTree [0:5])
print (y_testset [0:5])

#accuracy
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))