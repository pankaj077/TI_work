# Packages 
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin

'''
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as auc
'''

# get data 
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

# get consistency in number of columns, drop if not in train, add if not in test
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

# missing value treatment
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
def ft_eng(d,s):
    strings = d.dtypes == 'object'; strings = strings[strings].index.tolist();    
    a = pd.concat([pd.get_dummies(d[col]) for col in strings], axis=1)
    d = pd.concat([d, a], axis=1)
    for col in strings:
        d.drop(col,1, inplace = True)
    t = d[s::]
    d = d[0:s]
    return d,t

X,test = ft_eng(X,shape)

        