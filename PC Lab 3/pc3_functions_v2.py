"""
Data Analytics II: PC3.
Spring Semester 2022.
University of St. Gallen.

Jonas Husmann | 16-610-917
Niklas Leander Kampe | 16-611-618
"""

# import modules
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn import metrics

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

# summary statistics
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
    my_descriptives = {}
    for col_id in data.columns:
        my_descriptives[col_id] = [data[col_id].mean(),             
                                   data[col_id].var(),             
                                   data[col_id].std(),             
                                   data[col_id].max(),              
                                   data[col_id].min(),               
                                   sum(data[col_id].isna()),          
                                   len(data[col_id].unique())]  
    my_descriptives = pd.DataFrame(my_descriptives,
                                   index=["mean", "var", "std", "max",
                                          "min", "na", "unique"]).transpose()
    print('\nDescriptive Statistics:', '-' * 80,
          round(my_descriptives, 2), '-' * 80, sep='\n')
    
def sse_mse(y_test, y_pred):
    """
    Regression tree: Outcome Y only on root node.

    Parameters
    ----------
    y_test : TYPE: np.array/list
        DESCRIPTION: array/list of test set of the outcome variable
    y_pred : TYPE: np.array/list
        DESCRIPTION: array/list of predictions of the outcome variable
    Returns

    -------
    Returns the SSE and MSE accuracy metrics.
    """
    sse = 0
    # print(y_test.shape, y_pred.shape)
    # print(y_test)
    for i in y_test:
        # print(i)
        dev = (i - y_pred)**2
        # break
        sse = sse + dev
    mse = sse/len(y_test)
    return sse, mse

def regression_tree_root_node(df, X, Y):
    """
    Regression tree: Outcome Y only on root node --> predicts mean of outcome variable in train data set

    Parameters
    ----------
    df : TYPE: pd.DataFrame
        DESCRIPTION: dataframe of raw data
    X : TYPE: np.array/list
        DESCRIPTION: names of covariates
    Y : TYPE: np.array/list
        DESCRIPTION: name of outcome variable
    Returns

    -------
    Prints and returns the MSE accuracy metric.
    """
    df_X = df[[X]]
    df_y = df[[Y]]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, train_size = 0.8, random_state = 42)
    
    # Predict mean of outcome in training data
    y_pred = float(np.mean(y_train))
    y_test = y_test['Y'].tolist()
    
    # Calculate SSE and MSE
    sse, mse = sse_mse(y_test, y_pred)
    print('\nPredictive Regression Tree: Root Node Only', '-' * 80,
          f'Predicted Y: {round(y_pred, 2)}',
          f'SSE: {round(sse, 2)}',
          f'MSE: {round(mse, 2)}', '-' * 80, '\n', sep='\n')
    return mse

def sse_opt_regression_tree (df, X, Y, min_observations, maxdepth):
    """
    SSE optimizing regression tree: Predict the outcome Y with the covariate X.
    Required min_observations in leaves and setting depth = maxdepth
    Prints the bet-spliiting value of X and its row index
    Returns MSE
    """    
    df_X = df[[X]]
    df_y = df[[Y]]
    y = df[[Y]]
    y = y.astype('float')

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, train_size = 0.8, random_state = 42)
    
    # setting up the model
    regression = DecisionTreeRegressor(criterion="mse", splitter = "best", max_depth = maxdepth, min_samples_leaf = min_observations)
    regression = regression.fit(X_train,y_train)
    y_pred = regression.predict(X_test)
    
    # finding best splitting value of X
    tree_r = export_text(regression, feature_names=list(X_train.columns))
    X_bestsplitval = float(tree_r.split("X <= ", 1)[1].rstrip().split("\n|", 1)[0])
    
    # Row index of best splitting X
    X_index = X_test.sub(X_bestsplitval).abs().values.argmin()

    # Calculate SSE and MSE
    y_test = y_test['Y'].to_numpy()
    sse, mse = sse_mse(y_test, y_pred)

    #Print results
    print('SSE Optimizing Regression Tree', '-' * 80,
          f'Best splitting value of covariate X: {round(X_bestsplitval,4)}',
          f'Row Index of X: {round(X_index, 4)}',
          f'SSE: {round(sse[X_index], 4)}', 
          f'MSE: {round(mse[X_index], 4)}', 
          '-' * 80, sep='\n')
    return mse
    
    
