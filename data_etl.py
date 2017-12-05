##Use Python 2.7

import pandas as pd
import numpy as np
import sys
import scipy.stats
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os 

#Read drug files 
path = os.getcwd()
drugFiles = glob.glob(path + "/data/drug/drug*.txt")

path = list()
path.append(drugFiles)
path = [item for sublist in path for item in sublist]    

#Get n most frequent drugs
n = 100
list_ = []
for file_ in path:
    df = pd.read_csv(file_,sep="$")
    df2 = pd.DataFrame(df.drugname)
    list_.append(df2)
    
drugs = pd.concat(list_)
drug_count = pd.DataFrame(drugs.groupby('drugname').drugname.count())
drug_count['count'] = drug_count.drugname
drug_count = drug_count.drop('drugname',1)
drug_count['drugname'] = drug_count.index
count = pd.DataFrame(drug_count['count'])
count = count.sort_values(['count'], ascending=0)
topDrugs = count[:n]

#Extract drugs data with n most frequent drugs 
list_ = []
for file_ in path:
    df = pd.read_csv(file_,sep="$")
    df2 = df[df['drugname'].isin(topDrugs.index.tolist())]
    df3 = df2[['primaryid','role_cod','drugname','route','cum_dose_chr','cum_dose_unit','dose_freq']]
    list_.append(df3)

drugs = pd.concat(list_)

#Read demographic files 
path = os.getcwd()
demoFiles = glob.glob(path + "/data/demo/demo*.txt")

path = list()
path.append(demoFiles)
path = [item for sublist in path for item in sublist]    

#Extract demographic data 
list_ = []
for file_ in path:
    df = pd.read_csv(file_,sep="$")
    if 'gndr_cod' in df.columns: 
        df = df.rename(columns={'gndr_cod': 'sex'}) 
    df2 = df[['primaryid','event_dt','age','age_cod','sex','wt','wt_cod','occr_country']]
    list_.append(df2)

demo = pd.concat(list_)

#Read outcome files 
path = os.getcwd()
outcomeFiles = glob.glob(path + "/data/outc/outc*.txt")

path = list()
path.append(outcomeFiles)
path = [item for sublist in path for item in sublist]    

#Extract outcome data 
list_ = []
for file_ in path:
    df = pd.read_csv(file_,sep="$")
    if 'outc_code' in df.columns: 
        df = df.rename(columns={'outc_code': 'outc_cod'}) 
    df2 = df[['primaryid','outc_cod']]
    list_.append(df2)

outc = pd.concat(list_)

#Join files
table1 = pd.merge(outc, demo, on='primaryid')
table2 = pd.merge(table1, drugs, on='primaryid')

master_table = table2.dropna(how='any')

#convert age into years
master_table['age_yr'] = np.nan
master_table['age_yr'][master_table['age_cod']=='YR'] = master_table['age'][master_table['age_cod']=='YR'].astype(float)
master_table['age_yr'][master_table['age_cod']=='MON'] = master_table['age'][master_table['age_cod']=='MON'].astype(float)/12
master_table['age_yr'][master_table['age_cod']=='DY'] = master_table['age'][master_table['age_cod']=='DY'].astype(float)/365
master_table['age_yr'][master_table['age_cod']=='WK'] = master_table['age'][master_table['age_cod']=='WK'].astype(float)/52
master_table['age_yr'][master_table['age_cod']=='HR'] = master_table['age'][master_table['age_cod']=='HR'].astype(float)/8760
master_table = master_table.drop(['age','age_cod'],1)
master_table = master_table.dropna(how='any')

#all weights are in kg 
master_table = master_table.rename(columns={'wt': 'wt_kg'})
master_table = master_table.drop(['wt_cod'],1)

#dosage: convert mg, ug to grams, keep the rest as is for now
#see http://estri.ich.org/e2br22/ICH_ICSR_Specification_V2-3.pdf (p. 47) for unit codes
master_table['cum_dose_chr'][master_table['cum_dose_unit']=='MG'] = master_table['cum_dose_chr'][master_table['cum_dose_unit']=='MG'].astype(float)/1000
master_table['cum_dose_chr'][master_table['cum_dose_unit']=='UG'] = master_table['cum_dose_chr'][master_table['cum_dose_unit']=='UG'].astype(float)/1000000

#Create dummy variables
outc_dummies = pd.get_dummies(pd.Series(pd.Series(master_table['outc_cod']))).rename(columns=lambda x: 'outc_' + str(x)) 
sex_dummies = pd.get_dummies(pd.Series(pd.Series(master_table[(master_table['sex']=='M') | (master_table['sex'] == 'F')]['sex']))).rename(columns=lambda x: 'sex_' + str(x))  
role_dummies = pd.get_dummies(pd.Series(pd.Series(master_table['role_cod']))).rename(columns=lambda x: 'role_' + str(x)) 
route_dummies = pd.get_dummies(pd.Series(pd.Series(master_table['route']))).rename(columns=lambda x: 'route_' + str(x)) 
dose_freq_dummies = pd.get_dummies(pd.Series(pd.Series(master_table['dose_freq']))).rename(columns=lambda x: 'dose_freq_' + str(x)) 
country_dummies = pd.get_dummies(pd.Series(pd.Series(master_table['occr_country']))).rename(columns=lambda x: 'country_' + str(x)) 
drug_dummies = pd.get_dummies(pd.Series(pd.Series(master_table['drugname']))).rename(columns=lambda x: 'drug_' + str(x)) 

# master_table = pd.concat([master_table, outc_dummies, role_dummies, sex_dummies, route_dummies, dose_freq_dummies], axis=1, join_axes=[master_table.index])
master_table = pd.concat([master_table, outc_dummies, role_dummies, sex_dummies, route_dummies, country_dummies, drug_dummies, dose_freq_dummies], axis=1, join_axes=[master_table.index])

dropped_columns = ['outc_cod', 'event_dt', 'sex', 'role_cod', 'drugname', 'route', 'dose_freq', 'occr_country']

master_table = master_table.drop(dropped_columns, 1)

#Standardize features
weights = pd.to_numeric(master_table['wt_kg'], errors='coerce')
mean = weights.mean()
sd = weights.std()
master_table['wt_kg'] = (weights-mean)/sd

age = pd.to_numeric(master_table['age_yr'], errors='coerce')
mean = age.mean()
sd = age.std()
master_table['age_yr'] = (age-mean)/sd

dosage = pd.to_numeric(master_table['cum_dose_chr'], errors='coerce')
mean = dosage.mean()
sd = dosage.std()
master_table['cum_dose_chr'] = (dosage-mean)/sd

#####################Get total number of drugs used for each patient

filter_col = [col for col in master_table if col.startswith('drug') or col.startswith('dose') or col.startswith('role') or col.startswith('route') or col.startswith('outc_DE') or col.startswith('country')]
filter_col.append('primaryid')

drugs2 = master_table[filter_col]

filter_col = [col for col in master_table if col.startswith('drug')]

for col in filter_col: 
    drugs2[col] = drugs2[col]*master_table['cum_dose_chr']


drugs2 = drugs2.groupby('primaryid').sum()

drugs2.reset_index(level=0, inplace=True)

filter_col = [col for col in drugs2 if col.startswith('drug')]

drugs2['num_of_drugs'] = (drugs2[filter_col] != 0).sum(1)

#check
drugs2[filter_col].sum(1).describe()
drugs2.sort_values('num_of_drugs')

filter_col = [col for col in master_table if col.startswith('dose') or col.startswith('role') or col.startswith('route') or col.startswith('outc_DE') or col.startswith('country')]

for col in filter_col: 
    drugs2.loc[drugs2[col]>0, col] = 1

#####################

master_table2 = master_table[['primaryid', 'wt_kg', 'age_yr']].drop_duplicates()

master_table2 = pd.merge(master_table2, drugs2, on='primaryid')

master_table2.to_csv('master_table.csv')

positives = master_table2[master_table2.outc_DE == 1]
negatives = master_table2[master_table2.outc_DE == 0]

for i in range(0,10): 
    negatives_sample = negatives.sample(frac=0.106, replace=True)
    master_table2 = pd.concat([positives, negatives_sample])
    filename = 'master_table' + str(i+1) + '.csv'
    master_table2.to_csv(filename)
