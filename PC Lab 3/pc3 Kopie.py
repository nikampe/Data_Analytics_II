"""
Data Analytics II: PC3.

Spring Semester 2021.

University of St. Gallen.
"""

# Data Analytics II: PC Project 3

# import modules here
import sys
import pandas as pd

# set working directory
PATH = '/Users/Reshmeen/Documents/Uni/Master/Semester 2/Data Analytics II/PC Project 3/'
sys.path.append(PATH)

# load own functions
import pc3_functions as pc

# define the name for the output file
OUTPUT_NAME = 'pc3_output'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

# define data name
DATANAME = 'data_pc3.csv'

# load in data using pandas
data = pd.read_csv(PATH + DATANAME)

# your solutions start here
# --------------------------------------------------------------------------- #
# Question 1a: Descriptive statistics
pc.my_summary_stats(data)

# Question 1b:
pc.root_node(data, 'X', 'Y')

# Question 1c:
data_sorted = data.sort_values(by=['X'],ignore_index=True) # Sort data: increasing order acc. to X + reset index
pc.predictive_tree(data_sorted, 'X', 'Y', 1, 10) # Max depth = 1; min obs = 10
# --------------------------------------------------------------------------- #
# your solutions end here

# close the output file
sys.stdout.output.close()
sys.stdout = orig_stdout

# end of the PC 3 Session #
