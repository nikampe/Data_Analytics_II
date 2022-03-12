"""
Data Analytics II: PC3.
Spring Semester 2022.
University of St. Gallen.

Jonas Husmann | 16-610-917
Niklas Leander Kampe | 16-611-618
"""

# Data Analytics II: PC Project 3

import sys
import pandas as pd

# Part 1a)
PATH = '/Users/niklaskampe/Documents/GitHub/Data_Analytics_II/PC Lab 3/'
sys.path.append(PATH)

import pc3_functions as pc
from pc3_functions import my_summary_stats
from pc3_functions import regression_tree_root_node

OUTPUT_NAME = 'pc3_output'

orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

DATANAME = 'data_pc3.csv'
data = pd.read_csv(PATH + DATANAME)

# Part 1a)
print(round(data.head(n=5),2))
summary_stats = my_summary_stats(data)
# Part 1b)
regression_tree_root_node(data, 'X', 'Y')

sys.stdout.output.close()
sys.stdout = orig_stdout
