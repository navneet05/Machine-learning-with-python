# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import matplotlib.pyplot as  plt
import numpy as np
df=pd.read_csv("titanic_data.csv")
row=df.iloc[:,0]
df[:5]
print(row)
df.head()
df.head(2)
df.tail()
print(df.columns)
df.Survived
df[['Survived','Age']]
df['Fare'].mean() #min,std,mean
df.Survived.describe()
df[df.Survived == 1]
df['Name'][df.Age == df.Age.max()]
df.Age
df['Age']
#matplotlib
bins = [0,10,20,30,40,50,60,70,80,90,100]
plt.hist(df['Age'],bins ,histtype='bar', rwidth=0.8)
plt.xlabel('age groups')
plt.ylabel('Number of people')
plt.title('Histogram')
plt.show()
df[df.Fare == df.Fare.min()]
df['Fare'].min()
binss = [0,50,100,150,200,250]
plt.hist(df['Fare'],binss ,histtype='bar', rwidth=0.8)
plt.xlabel('fare range')
plt.ylabel('Number of people')
plt.title('Histogram')
plt.show()
df.loc[2]

#bar charts
plt.figure()
xvals = range(len(df.Survived))
plt.bar(xvals,df.Survived, width = 1,color='b')

males=df[df.Sex == 'male']
females=df[df.Sex == 'female']
index = np.arange(2)
temp=[len(males),len(females)]
plt.bar(index, temp,width=0.25)
plt.xlabel('Gender', fontsize=5)
plt.ylabel('No of persons', fontsize=5)
label=['males','females']
plt.xticks(index, label, fontsize=25, rotation=30)
plt.title('bar chart for no. of males and females')
plt.show()

