"""
Data Analytics II: PC1.

Spring Semester 2021.

University of St. Gallen.
"""

# Data Analytics II: PC Project 1

# import modules here
import sys
import pandas as pd

# set working directory
PATH = 'C:/your_path/'
sys.path.append(PATH)

# load own functions
import pc1_functions as pc

# define the name for the output file
OUTPUT_NAME = 'pc1_output'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

# define data name
DATANAME = 'data_pc1.csv'

# load in data using pandas
data = pd.read_csv(PATH + DATANAME)

# your solutions start here
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# your solutions end here

# close the output file
sys.stdout.output.close()
sys.stdout = orig_stdout

# End of the PC 1 Session #
