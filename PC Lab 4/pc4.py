"""
Data Analytics II: PC4.

Spring Semester 2021.

University of St. Gallen.
"""

# Data Analytics II: PC Project 4

# import modules
import sys
import pandas as pd

# set working directory
PATH = 'Q:/SEW/Lechner/Veranstaltungen/FS21/DA2/PClabs/PC4/'
sys.path.append(PATH)

# load own functions
import pc4_functions as pc

# define the name for the output file
OUTPUT_NAME = 'pc4_output'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = pc.Output(path=PATH, name=OUTPUT_NAME)

# define data name
DATANAME = 'data_pc4.csv'

# load in data using pandas
data = pd.read_csv(PATH + DATANAME)

# your solutions start here
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# your solutions end here

# close the output file
sys.stdout.output.close()
sys.stdout = orig_stdout

# end of the PC 4 Session #
