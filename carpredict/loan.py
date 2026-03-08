import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as svm
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_excel('car_loan.xlsx')
data.info()
data.describe()
data.isnull().sum()

data.ffill(inplace=True)
data.bfill(inplace=True)

data = data.drop('Loan_ID', axis=1)


data['Gender'] = data['Gender'].map({'Male':1,'Female':0})
data['Married'] = data['Married'].map({'Yes':1,'No':0})
data['Education'] = data['Education'].map({'Graduate':1,'Not Graduate':0})
data['Self_Employed'] = data['Self_Employed'].map({'Yes':1,'No':0})
data['Property_Area'] = data['Property_Area'].map({'Urban':2,'Semiurban':1,'Rural':0})


data['Dependents'] = data['Dependents'].replace('3+','3')
data['Dependents'] = data['Dependents'].astype(int)   #astype(int) converts the column from text to integer numbers.

data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0})
print(data.head())


x = data.drop('Loan_Status',axis=1)
y = data['Loan_Status']

X_train,X_test,y_train,y_test =train_test_split(x,y,test_size = 0.2 , random_state = 2)

model = LogisticRegression()
model.fit(X_train,y_train)

prediction = model.predict(X_test)
print(prediction)

accuracy = accuracy_score(y_test,prediction)
print("the accuracy socre is:",accuracy)


sns.countplot(x='Loan_Status', data=data)
plt.show()

