# Stratified sampling
import os
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error

path = 'C:\\Sampler'
os.chdir(path)

# Welcome messsage - no returns
def welcome():
    print 'Representative Sampler Created By HR Analytics \n'

# Get data - Function that imports data for sampling as well as size info
# This returns the main data set, list of job groups and the sample size
def get_data():
    filename='Sampling_Data.xlsx'
    sheet='Raw'
    Main_data = pd.read_excel(filename,sheetname=sheet,header=0).set_index("Emplid")
    
    sheet='Sample_info'
    Sample_info = pd.read_excel(filename,sheetname=sheet,header=0)
    
    job_groups = Sample_info["Job_Group"].tolist()
    sample_size = Sample_info["Sample_Size"].tolist()    
    return Main_data, job_groups, sample_size


# Get stratum- fucntion that returns the various levels of strata

def get_levels():
    Stratum_Input =[]
    #print 'Representative Sampler Created By HR Analytics \n'
    state=1
    run = 0
    while (state==1):
        if (run==0):
            strata_input = raw_input('Input strata level \n')
            Stratum_Input.append(strata_input)
            run = 1
        else:
            strata_input = raw_input('Press 0 to start sampling \nElse input another strata level \n')
            try:
                state = int(strata_input)
            except:
                Stratum_Input.append(strata_input)
    return Stratum_Input

# Sampling
def sampler(Main_data, job_groups, sample_size, stratum):
    run_limit = 1000
    run = 0
    random = 0
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
                #l1 = len(Sample_ratio.index)
                
                g_count = y[strata].value_counts()
                y_ratio = pd.DataFrame()
                y_ratio['Ratio']= g_count/g_count.sum()
                #l2 = len(y_ratio.index)
                try:
                    Error= mean_squared_error(y_ratio,Sample_ratio)*100
                    error_list.append(Error)    
                except:
                    Error = 100.000
                    error_list.append(Error)  
            if max(error_list)<5:
                break
            else: 
                run= run+1
                if (run < run_limit):
                    continue
                else:    
                    run = 0
                    random = random +1
                    break
        Sample_Set=Sample_Set.append(y_sample,ignore_index = False)            
        i=i+1
    return Sample_Set

# Main function which calls all the other functions and writes output to an excel file
def Main():
        
    welcome()
    
    Main_data, job_groups, sample_size = get_data() 
    stratum = get_levels()    
    
    Sample_Set = sampler(Main_data, job_groups, sample_size,stratum)
        
    writer = pd.ExcelWriter('Sample_Out.xlsx', engine='xlsxwriter')
    Sample_Set.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    
Main()

##################### END CODE ##############################################

