"""
Data Analytics II: PC3 Functions.

Spring Semester 2021.

University of St. Gallen.
"""

# import modules
import sys
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import export_text
from sklearn import tree
import matplotlib.pyplot as plt

# write a function to produce summary stats:
# mean, variance, standard deviation, maximum and minimum,
# the number of missings, unique values and number of observations
def my_summary_stats(data):
    """
    Summary stats: mean, variance, standard deviation, maximum and minimum.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: dataframe for which descriptives will be computed
    Returns

    -------
    None. Prints descriptive table od the data
    """
    # generate storage for the stats as an empty dictionary
    my_descriptives = {}
    # loop over columns
    for col_id in data.columns:
        # fill in the dictionary with descriptive values by assigning the
        # column ids as keys for the dictionary
        my_descriptives[col_id] = [data[col_id].mean(),                  # mean
                                   data[col_id].var(),               # variance
                                   data[col_id].std(),                # st.dev.
                                   data[col_id].max(),                # maximum
                                   data[col_id].min(),                # minimum
                                   sum(data[col_id].isna()),          # missing
                                   len(data[col_id].unique()),  # unique values
                                   data[col_id].shape[0]]      # number of obs.
    # convert the dictionary to dataframe for a nicer output and name rows
    # pandas dataframe will automatically take the keys as columns if the
    # data input is a dictionary. Transpose for having the stats as columns
    my_descriptives = pd.DataFrame(my_descriptives,
                                   index=['mean', 'var', 'std', 'max', 'min',
                                          'na', 'unique', 'obs']).transpose()
    # define na, unique and obs as integers such that no decimals get printed
    ints = ['na', 'unique', 'obs']
    # use the .astype() method of pandas dataframes to change the type
    my_descriptives[ints] = my_descriptives[ints].astype(int)
    # print the descriptives, (\n inserts a line break)
    print('Descriptive Statistics:', '-' * 80,
          round(my_descriptives, 2), '-' * 80, '\n\n', sep='\n')


# define an Output class for simultaneous console - file output
class Output():
    """Output class for simultaneous console/file output."""

    def __init__(self, path, name):

        self.terminal = sys.stdout
        self.output = open(path + name + ".txt", "w")

    def write(self, message):
        """Write both into terminal and file."""
        self.terminal.write(message)
        self.output.write(message)

    def flush(self):
        """Python 3 compatibility."""
        
# MY FUNCTIONS:
def root_node(df, X, Y):
    """ Predictive Regression Tree: Function provides tree prediction of Y with
    only the root node (i.e. predicts mean of outcome Y in training data) """
    
    # Variables
    X = df[[X]]
    y = df[[Y]]
    
    # Training/test split of data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)
    
    # Predict mean of outcome in training data
    y_pred = float(np.mean(y_train))
    y_test_list = y_test['Y'].tolist()
    
    # Calculate SSE and MSE
    sse = 0
    for i in y_test_list:
        dev = (i - y_pred)**2
        sse = sse + dev
    mse = sse/len(y_test_list)
    print("Predictive Regression Tree: Root Node Only\n" + "-"*80 +
          "\ny_pred: " + str(round(y_pred,2)) + "\nSSE: " + str(round(sse,2)) +
          "\nMSE: " + str(round(mse,2)) + "\n" + "-"*80 + "\n\n")
    return mse
    
def predictive_tree(df, X, Y, maxdepth, minobs):
    """ SSE Optimizing Regression Tree: Predict outcome Y with covariate X.
    Requires min. 10 obs. in leaves.
    Prints best-splitting value of X and corresponding row index. Plots tree.
    Returns MSE. """
    # Variables
    df.head()
    X = df[[X]]
    y = df[[Y]]
    y = y.astype('float')
    
    # Training/test split of data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=1)
    
    # Building model (min. obs in leaves = 10, no max depth)
    reg = DecisionTreeRegressor(criterion="mse", splitter = "best",
                                max_depth = maxdepth, min_samples_leaf = minobs)
    reg = reg.fit(X_train,y_train) # train tree regressor
    y_pred = reg.predict(X_test)
    
    # Best-splitting value of X
    tree_rules = export_text(reg, feature_names=list(X_train.columns))
    X_splitval = float(tree_rules.split("X <= ", 1)[1].rstrip().split("\n|", 1)[0])
    
    # Corresponding row index of best-split X value
    X_index = df['X'].sub(X_splitval).abs().values.argmin()
    
    # Plot tree
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(reg, filled=True)
    
    # Calculate MSE
    mse = metrics.mean_squared_error(y_test, y_pred)
    
    # Print results
    print("SSE Optimizing Regression Tree\n"+ "-"*80 +
          "\nBest Splitting Value of X: " + str(X_splitval) +
          "\nCorresponding Row Index of X: " + str(X_index) +
          "\nMSE: " + str(round(mse,2)) + "\n" + "-"*80 + "\n\n")
    
    return mse





