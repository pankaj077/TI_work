''' 
Generalized Classification Tool
Authors : Pankaj.Ajit

Description:
This is a generalized classification tool. 
The tool is to be used when you need to predict discrete outcomes.

Instructions:
1. The inputs required are a history dataset and the new dataset for prediction.
2. They should be named GCT_train and GCT_test respectively.
3. The class/target/dependent variable should be named 'target' - case insensitive.
4. There should be no unique id fields / serial numbers. The data is inherently handled in an order sensitive manner.

'''
 
# Packages 
# Importing the necessary libraries here
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score 

# Setting the path - should be changed accordingly
import os
os.chdir('/Users/z013sy0/Documents/Python scripts')

# get data 
'''
This section of codes imports the data. The data handled is only csv as of now.
After import the column headers are all changed to lower cases.
Missing values recognized are '#', Blanks, '?'.  
Missing values are converted to 'nan' format
'''

data_filename= 'GCT_train.csv'
test_filename= 'GCT_test.csv'

def getdata(dataname,testname):
	d = pd.read_csv(data_filename)
	t = pd.read_csv(test_filename)
	
	#consistency in column names
	d.columns = map(str.lower, d.columns)
	t.columns = map(str.lower, t.columns)	
	
	#consistency in missing values
	d = d.replace(r'\?+', np.nan, regex=True)
	d = d.replace(r'\#+', np.nan, regex=True)
	index_missTarget = d.target.isnull()
	d = d[index_missTarget != True]
	 
	t = t.replace(r'\?+', np.nan, regex=True)
	t = t.replace(r'\#+', np.nan, regex=True)
 
	X = d.drop('target',1)
	y = d['target']
	y=np.ravel(y)
	shape = d.shape[0]

	return X,y,t,shape

X,y,test,shape = getdata(data_filename,test_filename)
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


# get consistency in number of columns, drop if not in train, add if not in test
'''
This section of code aims at getting a consistency in the columns present in test set and train set.
If a columns is persent in train and not in test, that columns is added to test with all null values.
If a columns is persent in test and not in train, that columns is deleted from train.
'''

def consistency(d,t):
    data_labels = d.columns.values.tolist()
    test_labels = t.columns.values.tolist()
    
    not_data = set(test_labels) - set(data_labels)
    for name in not_data:
        t.drop(name,1,inplace = True)
    
    not_test = set(data_labels) - set(test_labels)
    for name in not_test:
        t[name] = np.nan
        
    return d,t
    
X,test = consistency(X,test)

# Missing value treatment
'''
This section of code aims at missing value imputation.
Median imputation for numeric features.
Mode imputation for string features.
'''
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

X = X.append(test)
X = DataFrameImputer().fit_transform(X)         
X.dropna(axis=1,how='all', inplace= True)

# Feature engineering
'''
Categorical features are rolled out to form binary variables for each category.
'''
def ft_eng(d,s):
    strings = d.dtypes == 'object'; strings = strings[strings].index.tolist();    
    a = pd.concat([pd.get_dummies(d[col]) for col in strings], axis=1)
    d = pd.concat([d, a], axis=1)
    for col in strings:
        d.drop(col,1, inplace = True)
    t = d[s::]
    d = d[0:s]
    sp_col = [col for col in d if d[col].nunique() == 2]
    al_col = d.columns.values.tolist()
    num_col = list(set(al_col) - set(sp_col))
    
    return d,t,sp_col,num_col
    
X,test,sp_col,num_col = ft_eng(X,shape)


'''
Scaling and Dimensionality reduction have to be done while at the same time 
ensuring that the sparsity of the data is not broken.
This section focuses on treating dense and sparse features separately.
'''

# Split into numeric and sparse features 
def splitter(d,t,s,n):
    d_sp = d[s]
    d_num = d[n]
    t_sp = t[s]
    t_num = t[n]
    
    return d_sp,d_num,t_sp,t_num

X_sparse, X_numeric, test_sparse, test_numeric = splitter(X,test,sp_col,num_col)


# Scale numeric
'''
It's important to account for the presence of outliers while scaling the data. 
This can be done through median centring rather than mean centering. 
Hence a robust scaler is used. Note that we scale numeric features only.
'''
def num_scaler(d_num,t_num):
    scl = RobustScaler()
    scl.fit(d_num)
    d_num = scl.transform(d_num)
    t_num = scl.transform(t_num)
    
    return d_num, t_num
    
X_numeric, test_numeric = num_scaler(X_numeric, test_numeric)

# PCA numeric
'''
Obtain linearly transformed features along the axis of highest variance.
PCA is run for dense/numeric features.
We set a target of explaining 90% of the variance in the data
'''
def num_dimred(d_num,t_num):
    pca = PCA(.9)
    pca.fit(d_num)
    d_num = pca.transform(d_num)
    t_num = pca.transform(t_num)

    return d_num, t_num

X_numeric, test_numeric = num_dimred(X_numeric, test_numeric)

# Truncated SVD
'''
For sparse data.
We set a target of explaining 90% of the variance in the data
'''
def sparse_dimred(d_sp,t_sp):
    if (len(sp_col)>3):
        expl = 0
        comp = 3
        while (expl < .9):
            svd = TruncatedSVD(n_components=comp)
            svd.fit(d_sp) 
            expl = svd.explained_variance_ratio_.sum()
            comp = comp + 1
        svd = TruncatedSVD(n_components=comp, random_state=22)
        svd.fit(d_sp)
        d_sp = svd.transform(d_sp)
        t_sp = svd.transform(t_sp)

    return d_sp,t_sp

X_sparse, test_sparse  = num_dimred(X_sparse, test_sparse)
    
# Merge sparse and numeric features
X_sparse = pd.DataFrame(X_sparse)
X_numeric = pd.DataFrame(X_numeric)
X = pd.concat([X_sparse, X_numeric],axis=1)

test_sparse = pd.DataFrame(test_sparse)
test_numeric = pd.DataFrame(test_numeric)
test = pd.concat([test_sparse, test_numeric],axis=1)

del X_sparse
del X_numeric
del test_sparse
del test_numeric


X = X.as_matrix()    
test = test.as_matrix()

# Split data into Train and hold out for model blending
X, X_holdout, y, y_holdout = train_test_split(X,y,test_size=0.25, random_state=22)

'''
This section of the code runs different moodels. Currently our tool is using logistic regression, Knnn
,Random Forests, and eXtreme Gradient Boosting.
Important consideration is to calibrate the posterior probabilities.
Certain algorithms give out under-confident predictions 
'''    

# Logistic regression
clf = LogisticRegression()
calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
score = cross_val_score(calibrated_clf,X,y,cv=3,scoring="roc_auc").mean() 
calibrated_clf.fit(X,y)
roc_auc_score(y_holdout,calibrated_clf.predict(X_holdout))
ypred_lr = calibrated_clf.predict(test)

# Random Forest
clf = RandomForestClassifier(n_estimators=np.int(np.sqrt(.75*shape)),min_samples_split=20, n_jobs=-1)
calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
calibrated_clf.fit(X, y)
roc_auc_score(y_holdout,calibrated_clf.predict(X_holdout))

ypred_rf = calibrated_clf.predict(test)

# KNN
clf = KNeighborsClassifier(n_neighbors = np.int(np.sqrt(.75*shape)))
calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
score = cross_val_score(calibrated_clf,X,y,cv=3,scoring="roc_auc").mean() 
calibrated_clf.fit(X,y)

ypred_knn = calibrated_clf.predict(test)
    
roc_auc_score(y_holdout,calibrated_clf.predict(X_holdout))

# Xgboost

#X, X_test, y, y_test = train_test_split(X,y,test_size=0.25, random_state=33)

test = xgb.DMatrix(test)
X = xgb.DMatrix(X , label=y)
X_holdout = xgb.DMatrix(X_holdout , label=y_holdout)
y_distinct = set(y)
cl_num = len(y_distinct)


param = {}
# use softmax multi-class classification
param['objective'] = 'binary:logistic'
#scale weight of positive examples
param['eta'] =0.1
#param['max_depth'] = 10
#param['nthread'] = 4
#param['num_class'] = 2
param['min_child_weight']=20
#param['colsample_bytree']=0.5
#param['max_delta_step']=1
#param['subsample']=0.6
param['gamma']= 1
param['eval_metric'] = 'auc'

num_round = 200
xgb.cv(param, X, num_round, nfold=5,seed = 0)

watchlist  = [(X_holdout,'eval'), (X,'train')]
xgb.train(param, X, num_round, watchlist,early_stopping_rounds=10)

ypred = mdl.predict(test)    
    
