"""
Data Analytics II: PC5.
Spring Semester 2022.
University of St. Gallen.

Jonas Husmann | 16-610-917
Niklas Leander Kampe | 16-611-618
"""

# import modules
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy import stats


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


def summary_statistics(data):
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


def my_hist(data, varname, path, nbins=10):
    """
    Plot histograms.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: dataframe containing variables of interest
    varname : TYPE: string
        DESCRIPTION: variable name for which histogram should be plotted
    path : TYPE: string
        DESCRIPTION: path where the plot will be saved
    nbins : TYPE: integer
        DESCRIPTION. Number of bins. The default is 10.

    Returns
    -------
    None. Prints and saves histogram.
    """
    data[varname].plot.hist(grid=True, bins=nbins, rwidth=0.9, color='grey')
    plot.title('Histogram of ' + varname)
    plot.xlabel(varname)
    plot.ylabel('Counts')
    plot.grid(axis='y', alpha=0.75)
    plot.savefig(path + '/histogram_of_' + varname + '.png')
    plot.show()


# use own ols procedure
def my_ols(exog, outcome, intercept=True, display=True):
    """
    OLS estimation.

    Parameters
    ----------
    exog : TYPE: pd.DataFrame
        DESCRIPTION: covariates
    outcome : TYPE: pd.Series
        DESCRIPTION: outcome
    intercept : TYPE: boolean
        DESCRIPTION: should intercept be included? The default is True.
    display : TYPE: boolean
        DESCRIPTION: should results be displayed? The default is True.

    Returns
    -------
    result: ols estimates with standard errors
    """
    # check if intercept should be included
    # the following condition checks implicitly if intercept == True
    if intercept:
        # if True, prepend a vector of ones to the covariate matrix
        exog = pd.concat([pd.Series(np.ones(len(exog)), index=exog.index,
                                    name='intercept'), exog], axis=1)
    # compute (x'x)-1 by using the linear algebra from numpy
    x_inv = np.linalg.inv(np.dot(exog.T, exog))
    # estimate betas according to the OLS formula b=(x'x)-1(x'y)
    betas = np.dot(x_inv, np.dot(exog.T, outcome))
    # compute the residuals by subtracting fitted values from the outcomes
    res = outcome - np.dot(exog, betas)
    # estimate standard errors for the beta coefficients
    s_e = np.sqrt(np.diagonal(np.dot(np.dot(res.T, res), x_inv) /
                              (exog.shape[0] - exog.shape[1])))
    # compute the t-values
    tval = betas / s_e
    # compute pvalues based on the students t-distribution (requires scipy)
    # sf stands for the survival function (also defined as 1 - cdf)
    pval = stats.t.sf(np.abs(tval),
                      (exog.shape[0] - exog.shape[1])) * 2
    # put results into dataframe and name the corresponding values
    result = pd.DataFrame([betas, s_e, tval, pval],
                          index=['coef', 'se', 't-value', 'p-value'],
                          columns=list(exog.columns.values)).transpose()
    # check if the results should be printed to the console
    # the following condition checks implicitly if display == True
    if display:
        # if True, print the results (\n inserts a line break)
        print('OLS Estimation Results:', '-' * 80,
              round(result, 2), '-' * 80, '\n\n', sep='\n')
    # return the resulting dataframe too
    return result

def histogram(data, columns, bins = 30):
    """
    Plot Continuous Variables.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: data including all variables
    columns : TYPE: np.array/list
        DESCRIPTION: target variables for plotting

    Returns
    -------
    Plots and Saves all Histograms
    """
    for col in columns:
        data.hist(column = col, bins = bins)
        plot.suptitle(f'Histogram of {col}')
        plot.savefig(f'histogram_{col}.png', format='png')
        plot.show()
        
def table_mean_obs(data, groups, covariates):
    """
    Aggregated Mean and Count Statistics.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: data including all variables
    groups : TYPE: np.array/list
        DESCRIPTION: Variables to group by
    covariates : TYPE: np.array/list
        DESCRIPTION: Target variables for aggregated statistics (mean, count)

    Returns
    -------
    Mean and Count Statistics for each unique value in group columns for target variables
    """
    overview = data.groupby(groups)[covariates].agg(['mean', 'count'])
    print('\nMeans & Observations:', '-' * 80, round(overview, 4), '-' * 80, sep = '\n')       
        
def cross_table(data, columns):
    """
    Cross Table.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: data including all variables
    columns : TYPE: np.array/list
        DESCRIPTION: Variables to compare against

    Returns
    -------
    Cross table with unique values of target variables against each other
    """
    ct = pd.crosstab(data[columns[0]], data[columns[1]])
    print('\nCross Table:', '-' * 80, round(ct, 4), '-' * 80, sep = '\n')
    
    
def TSLS(exog, endog, instrument, outcome, intercept=True, display=True):
    """
    Two Stage Least Squares regression

    Parameters
    ----------
    exog : TYPE: pd.DataFrame
        DESCRIPTION: covariates
    endog : TYPE: pd.DataFrame
        DESCRIPTION: covariates
    instrument : TYPE: pd.DataFrame
        DESCRIPTION: instrument for endog variable
    outcome : TYPE: pd.Series
        DESCRIPTION: outcome
    intercept : TYPE: boolean
        DESCRIPTION: should intercept be included? The default is True.
    display : TYPE: boolean
        DESCRIPTION: should results be displayed? The default is True.

    Returns
    -------
    result: Two stage least squares regression results with standard errors
    """
    if intercept:
        exog1 = pd.concat([pd.Series(np.ones(len(exog)), index=exog.index,
                                    name='intercept'), exog, instrument], axis=1)
    else:
        exog1 = pd.concat([exog, instrument], axis=1)
    x_inv1 = np.linalg.inv(np.dot(exog1.T, exog1))
    betas1 = np.dot(x_inv1, np.dot(exog1.T, endog))
    prediction1 = pd.Series(np.dot(exog1, betas1))
    if intercept:
        exog2 = pd.concat([pd.Series(np.ones(len(exog)), index=exog.index,
                                    name='intercept'), exog, prediction1], axis=1)
    else:
        exog2 = pd.concat([exog, prediction1], axis=1)
    x_inv2 = np.linalg.inv(np.dot(exog2.T, exog2))
    betas2 = np.dot(x_inv2, np.dot(exog2.T, outcome))
    res = outcome - np.dot(exog2, betas2)
    s_e = np.sqrt(np.diagonal(np.dot(np.dot(res.T, res), x_inv2) /
                              (exog2.shape[0] - exog2.shape[1])))
    tval = betas2 / s_e
    pval = stats.t.sf(np.abs(tval),
                      (exog2.shape[0] - exog2.shape[1])) * 2
    result = pd.DataFrame([betas2, s_e, tval, pval],
                          index=['coef', 'se', 't-value', 'p-value'],
                          columns=list(exog2.columns.values)).transpose()
    if display:
        print('OLS Estimation Results:', '-' * 80,
              'Dependent Variable: ' + outcome.name, '-' * 80,
              round(result, 2), '-' * 80, '\n\n', sep='\n')
    return result
