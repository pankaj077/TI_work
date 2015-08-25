# Stratified sampling

from sklearn.cross_validation import train_test_split
import pandas as pd
from __future__ import division
from sklearn.metrics import mean_squared_error

# Get data

filename='Sampling_Data.xlsx'
sheet='Raw'
Main_data = pd.read_excel(filename,sheetname=sheet,header=0)

sheet='Sample_info'
Sample_info = pd.read_excel(filename,sheetname=sheet,header=0)

job_groups = Sample_info["Job_Group"].tolist()
sample_size = Sample_info["Sample_Size"].tolist()    

stratum = ['Race','Gender','Job_Level','Location']
non_stratum = ['Emplid','Job_Group']

# Sampling

i=0    
Sample_Set= pd.DataFrame()
for job in job_groups:
    y = Main_data[(Main_data.Job_Group == job)]
    # Split
    while True:
        y_train, y_sample = train_test_split(y,test_size=sample_size[i])
        error_list = list()
        for strata in stratum:
            g_count = y_sample[strata].value_counts()
            Sample_ratio = pd.DataFrame()
            Sample_ratio['Ratio']= g_count/g_count.sum()
            l1 = len(Sample_ratio.index)
            
            g_count = y[strata].value_counts()
            y_ratio = pd.DataFrame()
            y_ratio['Ratio']= g_count/g_count.sum()
            l2 = len(y_ratio.index)
            
            if (l1==l2):
                Error= mean_squared_error(y_ratio,Sample_ratio)*100
                error_list.append(Error)    
            else:
                 break
        if not error_list:
            break
        elif max(error_list)<5:
            break
        else: 
            continue    
    Sample_Set=Sample_Set.append(y_sample,ignore_index = True)            
    i=i+1

writer = pd.ExcelWriter('Sample_Out.xlsx', engine='xlsxwriter')
Sample_Set.to_excel(writer, sheet_name='Sheet1')
writer.save()

##################### END CODE ##############################################

#Pre-processing
filename='Sampling_Data1.xlsx'
sheet='Sample_info'
df = pd.read_excel(filename,sheetname=sheet,header=0)

def encode_target(df, target_column):
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)
    return (df_mod, targets)
	
df2, JG2 = encode_target(df, "Job_Group")

writer = pd.ExcelWriter('Mapping.xlsx', engine='xlsxwriter')
df2.to_excel(writer, sheet_name='Sheet1')
writer.save()
