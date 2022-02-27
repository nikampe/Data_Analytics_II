"""
Data Analytics II: PC1.
Spring Semester 2021.
University of St. Gallen.

Jonas Husmann | 
Niklas Leander Kampe | 16-611-618
"""

import sys
import pandas as pd

# Part 1a)
PATH = '/Users/niklaskampe/Documents/GitHub/Data_Analytics_II/PC Lab 1/'
sys.path.append(PATH)

import pc1_functions as pc
from pc1_functions import summary_statistics
from pc1_functions import histogram
from pc1_functions import balance_check

OUTPUT_NAME = 'pc1_output'

orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

DATANAME = 'data_pc1.csv'

data = pd.read_csv(PATH + DATANAME)

# Part 1a)
print(round(data.head(n=5),2))
# Part 1b)
descriptive_stats = data.describe()
print(round(descriptive_stats,2))
# Part 1c)
summary_stats = summary_statistics(data)
# Part 1d)
data.drop(columns = ['age2'], inplace=True)
print(round(data.head(n=5),2))
# Part 1e)
histogram(data)
# Part 1f)
data.dropna(axis=0, inplace=True)
data.to_csv('data_pc1_cleaned')
# Part 1g)
balance_check(data, 'treat', data.columns[1:])

sys.stdout.output.close()
sys.stdout = orig_stdout