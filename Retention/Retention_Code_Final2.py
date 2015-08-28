# Retention Code Version 2

# Get Training data 
import pandas as pd
filename='TI_TM10 back up.xlsx'
sheet='Sheet4'
Main_data = pd.read_excel(filename,sheetname=sheet,header=0)

# Get features and labels
import numpy as np
y = Main_data[["Status"]]
y =np.ravel(y)
colsToDrop = ['Emplid','Name','Status']
X = Main_data.drop(colsToDrop, axis=1)

# Training and K-fold cross validation - Random Forest
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
n_range = [range(10,900,100)]
n_score=[]
for n in n_range:
    rf=RandomForestClassifier(n_estimators=n)
    score = cross_val_score(rf,X,y,cv=10,scoring="roc_auc").mean() 
    n_score.append(score)
n_optimal = n_range[np.argmax(n_score)]

rf1=RandomForestClassifier(n_estimators=n_optimal) 
rf1.fit(X,y) 

trf = RandomForestClassifier()
score_trf = cross_val_score(rf,X,y,cv=10,scoring="roc_auc").mean() 

# Training and K-fold cross validation - KNN
from sklearn.neighbors import KNeighborsClassifier
k_range = range(1,530,20)
k_score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(rf,X,y,cv=10,scoring="roc_auc") 
    k_score.append(score.mean())
k_optimal = k_range[np.argmax(k_score)]
    
knn1=KNeighborsClassifier(n_neighbors=k_optimal) 
knn1.fit(X,y) 



# Training and K-fold cross validation 
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier 


alist=[] 
for i in range(100,900,100): 
	rf=RandomForestClassifier(n_estimators=i) 
	roc_auc=cross_val_score(rf,X,y,cv=10,scoring="roc_auc") 
	mean_roc_auc=np.mean(roc_auc) 

	adict={i:mean_roc_auc} 
	alist.append(adict) 

	check={k:v for d in alist for k,v in d.items()} 
	max1=max(check.values()) 
	key=[k for k,v in check.items() if v==max1] 
	value=key[0] 

# Fitting and predicting 
rf1=RandomForestClassifier(n_estimators=rf_param()) 
rf1.fit(x,y) 

y_pred=rf1.predict(X_test) 

# Cross validation with test set
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)

# Fitting on current data and predicting risks
rf1=RandomForestClassifier(n_estimators=rf_param()) 
rf1.fit(X_train,y_train) 
y_pred=rf1.predict(X_test) 

Prediction=pd.DataFrame(y_pred) 
Prediction.columns=["Risk_flag"]


