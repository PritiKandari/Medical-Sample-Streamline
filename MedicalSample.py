#import library
import pandas as pd
import numpy as np

df=df=pd.read_excel(r"C:\Users\India\Desktop\Internship 69\Medical_Sample_Data.xlsx")
df.head()
df.info()

# Data Preprocessing
print(df['Patient_Gender'].unique())
print(df['Test_Name'].unique())
print(df['Sample'].unique())
print(df['Way_Of_Storage_Of_Sample'].unique())
print(df['Traffic_Conditions'].unique())
print(df['Mode_Of_Transport'].unique())

# Missing Value
df.isnull().sum()
df.describe()
df.columns

# using label encoder to convert the categorical(Class Variable) column to numerical
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df["Reached_On_Time"] = lb.fit_transform(df["Reached_On_Time"])
df['Patient_Gender']=lb.fit_transform(df['Patient_Gender'])
df['Test_Name']=lb.fit_transform(df['Test_Name'])
df['Sample']=lb.fit_transform(df['Sample'])
df['Way_Of_Storage_Of_Sample']=lb.fit_transform(df['Way_Of_Storage_Of_Sample'])
df['Cut-off Schedule']=lb.fit_transform(df['Cut-off Schedule'])
df['Traffic_Conditions']=lb.fit_transform(df['Traffic_Conditions'])
df['Mode_Of_Transport']=lb.fit_transform(df['Mode_Of_Transport'])

df1 = df.drop(['Patient_ID','Patient_Age','Test_Booking_Date','Sample_Collection_Date','Agent_ID','Mode_Of_Transport'],axis=1)
df1.columns

import matplotlib.pyplot as plt

plt.scatter(x = df1 ['Time_Taken_To_Reach_Lab_MM'],y=df1['Reached_On_Time'], color = 'blue') 
plt.scatter(x=df ['Time_Taken_To_Reach_Lab_MM'],y=df1['Reached_On_Time'], color = 'blue')

import seaborn as sns
sns.pairplot(df1.iloc[:, :])

sns.jointplot(x=df1['Time_Taken_To_Reach_Lab_MM'],y=df1['Reached_On_Time'])
sns.jointplot(x=df1['Cut-off time_HH_MM'], y=df1['Reached_On_Time'])
sns.jointplot(x=df1['Scheduled_Sample_Collection_Time_HH_MM'],y=df1['Reached_On_Time'])
sns.jointplot(x=df1['Test_Booking_Time_HH_MM'], y=df1['Reached_On_Time'])

#independent and dependent fearures
X = df1.iloc[:, :-1]
y = df1.iloc[:,-1]

 #train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)


#random forest classifier
from sklearn.ensemble import RandomForestClassifier

rf_rendom = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

rf_rendom.fit(x_train, y_train)

# Test Data Accuracy 
from sklearn.metrics import accuracy_score, confusion_matrix
print(confusion_matrix(y_test, rf_rendom.predict(x_test)))
accuracy_score(y_test, rf_rendom.predict(x_test))

# Train Data Accuracy
accuracy_score(y_train, rf_rendom.predict(x_train))# model is overfitting


# Creating new model testing with new parameters
forest_new = RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_split=20,criterion='gini')  # n_estimators is the number of decision trees
forest_new.fit(x_train, y_train)

print('Train accuracy: {}'.format(forest_new.score(x_train, y_train)))

print('Test accuracy: {}'.format(forest_new.score(x_test, y_test)))

import pickle
# open a file, where you want to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_rendom, file)


