import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,plot_confusion_matrix


# Load Data

df=pd.read_csv("processed.cleveland.data")

# Name Columns

df.columns=["Age","Sex","Cp","Restbp","Chol","Fbs","Restecg","Thalach","Exang","Oldpeak","Slope","Ca","Thal","Hd"]


# Check Data Types 
df.dtypes 


df["Ca"].unique()
df["Thal"].unique()

# Missing Value

filter1= df["Thal"]=='?'
filter2=df["Ca"] == '?'

df=df[~(filter1 | filter2)]

# Create x and y

y=df["Hd"]
x=df.drop(["Hd"],axis=1)

# Check All Columns 

for i in df.columns:
    print(i)
    print(df[i].unique())
    print("-----------")

# One Hot Encoding

x=pd.get_dummies(data=x,columns=["Cp","Restecg","Slope","Ca","Thal"])

# Make Binary Classification

filter3=y>0

y[filter3] = 1

# Train - Test Split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)

# Preliminary Model

model=RandomForestClassifier(random_state=42,n_estimators=100)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

confusion_matrix(y_test,y_pred)
plot_confusion_matrix(model,x_test,y_test)

model.score(x_test, y_test)

# Model Optimization

model2=RandomForestClassifier(random_state=42)

param_grid={"n_estimators":[100,200,500,1000],"max_features":[3,5,8,10,15,17,20,23],"max_depth":list(range(1,10))}

cv_model=GridSearchCV(model2, param_grid,cv=10,n_jobs=-1).fit(x_train,y_train)
cv_model.best_params_

# Final Model

final_model=RandomForestClassifier(max_depth=1, max_features=3, n_estimators=500,random_state=42)

final_model.fit(x_train,y_train)

final_model.score(x_test,y_test)
plot_confusion_matrix(final_model,x_test,y_test)
























