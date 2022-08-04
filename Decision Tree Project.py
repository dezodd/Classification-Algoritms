import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from sklearn.tree import plot_tree


df=pd.read_csv('processed.cleveland.data')


df.columns=["Age","Sex","Cp","Restbp","Chol","Fbs","Restecg","Thalach","Exang","Oldpeak","Slope","Ca","Thal","Hd"]

# Check Data Types 
df.dtypes 

df["Ca"].unique()
df["Thal"].unique()

filter1= df["Thal"]=='?'
filter2=df["Ca"] == '?'

df=df[~(filter1 | filter2)]

y=df["Hd"]
x=df.drop(["Hd"],axis=1)

# Check All Columns 

for i in df.columns:
    print(i)
    print(df[i].unique())
    print("-----------")

# One Hot Encoding

x=pd.get_dummies(data=x,columns=["Cp","Restecg","Slope","Ca","Thal"])

filter3=y>0

y[filter3] = 1


# Decision Tree Model

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)

model=DecisionTreeClassifier(random_state=42)
model.fit(x_train,y_train)

plot_confusion_matrix(model,x_test,y_test,display_labels=["Does not have HD","Has HD"])


# Model Optimization / Alpha Optimization

path=model.cost_complexity_pruning_path(x_train,y_train)
ccp_alphas=path["ccp_alphas"]

train_score=[]
test_score=[]

for ccp_alpha in ccp_alphas:
    model1=DecisionTreeClassifier(random_state=42,ccp_alpha=ccp_alpha)
    model1.fit(x_train,y_train)
    train_score.append(model1.score(x_train,y_train))
    test_score.append(model1.score(x_test,y_test))

# Find Alpha

alpha_index=np.argmax(test_score)
alpha=ccp_alphas[alpha_index]

fig,ax=plt.subplots()

ax.set_xlabel("ccp_alpha")
ax.set_ylabel("Score")

ax.plot(ccp_alphas,train_score,marker="o",label="train",drawstyle="steps-post")
ax.plot(ccp_alphas,test_score,marker="o",label="test",drawstyle="steps-post")
ax.legend()


model_pruned=DecisionTreeClassifier(random_state=42,ccp_alpha=alpha)
model_pruned.fit(x_train,y_train)



# Cross Validation

cv_scores=[]

score=0
final_alpha=0

for ccp_alpha in ccp_alphas:
    model2=DecisionTreeClassifier(random_state=42,ccp_alpha=ccp_alpha)
    scores=cross_val_score(model2,x_train,y_train,cv=5)
    cv_scores.append([ccp_alpha,scores,np.mean(scores)])
    if  np.mean(scores) > score:
        score=np.mean(scores)
        final_alpha=ccp_alpha
    
# Create Final Model

final_model=DecisionTreeClassifier(random_state=42,ccp_alpha=final_alpha)

final_model.fit(x_train,y_train)

plot_confusion_matrix(final_model,x_test,y_test)


































