"""
Data Analytics II: PC5.
Spring Semester 2022.
University of St. Gallen.

Jonas Husmann | 16-610-917
Niklas Leander Kampe | 16-611-618
"""

# Data Analytics II: PC Project 5

import sys
import pandas as pd


# Part 1a)
PATH = '/Users/niklaskampe/Documents/GitHub/Data_Analytics_II/PC Lab 5/'
sys.path.append(PATH)

import pc5_functions as pc
from pc5_functions import summary_statistics
from pc5_functions import histogram
from pc5_functions import table_mean_obs
from pc5_functions import cross_table
from pc5_functions import my_ols
from pc5_functions import my_tsls

OUTPUT_NAME = 'pc5_output'

orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

DATANAME = 'data_pc5.csv'
data = pd.read_csv(PATH + DATANAME)

# Part 1a)
print(round(data.head(n = 5), 2))
# Part 1b)
summary_statistics(data)
histogram(data, ['kidcount', 'weeks_work'], bins = 30)
# Part 1c)
table_mean_obs(data, ['kidcount'], ['employed'])
# Part 1d)
cross_table(data, ['morekids', 'multi2nd'])
# Part 2a)
exog = ['morekids', 'black', 'hisp', 'age_mother', 'hsgrad', 'colgrad']
ols = my_ols(data[exog], #exog
              data['weeks_work']) #outcome
# Part 2b)
tsls = my_tsls(data[['black', 'hisp', 'age_mother', 'hsgrad', 'colgrad']], #exog
             data['morekids'], #endog
             data['multi2nd'], #instrument
             data['weeks_work']) #outcome

sys.stdout.output.close()
sys.stdout = orig_stdout
