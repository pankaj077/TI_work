# Get the data 
import pandas as pd
filename='TI_TM10 back up.xlsx'
sheet='Sheet4'
Main_data = pd.read_excel(filename,sheetname=sheet,header=0)

# Split the data into Training and Test set
split1 = 10.0
split2 = 0.0
from sklearn.cross_validation import train_test_split
from __future__ import division
while True:  
    Main_data_train, Main_data_test = train_test_split(Main_data,test_size=0.25)

    Split_train= Main_data_train['Status'].value_counts()
    Split_test= Main_data_test['Status'].value_counts()
    
    split1 = Split_train[1]/(Split_train[1]+Split_train[0])
    split2 = Split_test[1]/(Split_test[1]+Split_test[0])
    dif= abs(split1-split2)*100   
    
    if dif<1:
        break
    else:
        continue
        
# Get features and labels
import numpy as np
y = Main_data_train[["Status"]]
y_train=np.ravel(y)
colsToDrop = ['Emplid','Name','Status']
X_train = Main_data_train.drop(colsToDrop, axis=1)

y2 = Main_data_test[["Status"]]
y_test=np.ravel(y2)
X_test = Main_data_test.drop(colsToDrop, axis=1)

# Training and cross validation 
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier 

def rf_param(): 
	alist=[] 
	for i in range(100,900,100): 
		rf=RandomForestClassifier(n_estimators=i) 
		roc_auc=cross_val_score(rf,X_train,y_train,cv=10,scoring="roc_auc") 
		mean_roc_auc=np.mean(roc_auc) 
		adict={i:mean_roc_auc} 
		alist.append(adict) 
		check={k:v for d in alist for k,v in d.items()} 
		max1=max(check.values()) 
		key=[k for k,v in check.items() if v==max1] 
		value=key[0] 
	return value 

# Fitting and predicting 
rf1=RandomForestClassifier(n_estimators=rf_param()) 
rf1.fit(X_train,y_train) 
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
