
"""
Data Analytics II: PC4.
Spring Semester 2022.
University of St. Gallen.

Jonas Husmann | 16-610-917
Niklas Leander Kampe | 16-611-618
"""

# Data Analytics II: PC Project 4

import sys
import pandas as pd
pd.set_option('display.max_columns', None)

# Part 1a)
PATH = '/Users/niklaskampe/Documents/GitHub/Data_Analytics_II/PC Lab 4/'
sys.path.append(PATH)

import pc4_functions as pc
from pc4_functions import summary_statistics
from pc4_functions import histogram, histogram_change
from pc4_functions import dummy_check
from pc4_functions import table_mean_obs

OUTPUT_NAME = 'pc4_output'

orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

DATANAME = 'data_pc4.csv'
data = pd.read_csv(PATH + DATANAME)

# Part 1a)
print(round(data.head(n = 5), 2))
# Part 1b)
summary_statistics(data)
# Part 1c)
histogram(data, "fte", ["year", "state"], bins = 30)
histogram_change(data, "fte", ["year", "state"], bins = 30)
# Part 1d)
# dummy_check(data, ["southj", "centralj", "northj", "pa1", "pa2"])
dummy_check(data, ["southj", "centralj", "northj"], 1)
dummy_check(data, ["pa1", "pa2"], 0)
# Part 1e)
table_mean_obs(data, ['year', 'state'],[ 'fte', 'wage_st','hrsopen', 'price'])
# Part 1f)
for i in data.index:
    data.loc[i, 'year'] = 1 if data.loc[i, 'year'] == 93 else 0
data = pd.get_dummies(data, columns = ["chain"])
data.rename(columns={'chain_1': 'Burgerking', 'chain_2': 'KFC', 'chain_3': 'Royrogers', 'chain_4': 'Wendys'}, inplace = True)
print(round(data.head(n = 5), 2))

sys.stdout.output.close()
sys.stdout = orig_stdout
