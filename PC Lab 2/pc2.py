"""
Data Analytics II: PC2.
Spring Semester 2022.
University of St. Gallen.

Jonas Husmann | 16-610-917
Niklas Leander Kampe | 16-611-618
"""

# Data Analytics II: PC Project 2

import sys
import pandas as pd
pd.set_option('display.max_columns', None)

# Part 1a)
PATH = '/Users/jonashusmann/Documents/GitHub/Data_Analytics_II/PC Lab 2/'
sys.path.append(PATH)

import pc2_functions as pc
from pc2_functions import summary_statistics
from pc2_functions import histogram
from pc2_functions import balance_check
from pc2_functions import OLS_regression

OUTPUT_NAME = 'pc2_output'

orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

DATANAME = 'data_pc2.csv'
data = pd.read_csv(PATH + DATANAME)

# Part 1a)
print(round(data.head(n=5),2))
# Part 1b)
summary_stats = summary_statistics(data)
continuous_variables = ['bweight', 'mage', 'medu', 'nprenatal', 'monthslb']
histogram(data, continuous_variables)
# Part 1c)
data.drop(columns = ['msmoke', 'monthslb'], inplace = True)
for col in ['order', 'prenatal']:
    data = data[data[col].notna()]
print(round(data.head(n=5),2))
# Part 1d)
for i in data.index:
    for col in ['order', 'prenatal']:
        if data.loc[i, col] == 1:
           data.loc[i, col] = 1
        else:
            data.loc[i, col] = 0
print(round(data.head(n=5),2))
# Part 1e)
summary_stats = summary_statistics(data)
data.to_csv('data_pc2_cleaned')
# Part 1f)
balance_check(data, 'mbsmoke', data.drop(columns = ['mbsmoke', 'bweight']).columns)
balance_check(data, 'mbsmoke', ['bweight'])

# Part 2a)
#defining the variables
X = data['mbsmoke']
Y = data['bweight']
#running the OLS regression
OLS_reg = OLS_regression(Y, X)

# Part 2b)
#checking correlations
cor = pd.DataFrame(round(data.corr(),3))
print(cor)
# defining the variables
X = data.drop(['bweight', 'mhisp', 'mrace'], axis = 1)
Y = data['bweight']
#running the OLS regression
OLS_reg2 = OLS_regression(Y, X)


# close the output file
sys.stdout.output.close()
sys.stdout = orig_stdout

