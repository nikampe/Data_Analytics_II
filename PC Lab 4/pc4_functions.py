"""
Data Analytics II: PC4.
Spring Semester 2022.
University of St. Gallen.

Jonas Husmann | 16-610-917
Niklas Leander Kampe | 16-611-618
"""

import sys
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
import statsmodels as stats
import matplotlib.pyplot as plt

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


# own procedure to do histograms
def my_hist(data, varname, path, nbins=10, label=""):
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
    label: Type: string
        DESCRIPTION. Label for the title. The default is none.

    Returns
    -------
    None. Prints and saves histogram.
    """
    # produce nice histograms
    data[varname].plot.hist(grid=True, bins=nbins, rwidth=0.9, color='grey')
    # add title
    if label == "":
        plot.title('Histogram of ' + varname)
    else:
        plot.title('Histogram of ' + varname + ' for ' + label)
    # add labels
    plot.xlabel(varname)
    plot.ylabel('Counts')
    plot.grid(axis='y', alpha=0.75)
    # save the plot
    if label == "":
        plot.savefig(path + '/histogram_of_' + varname + '.png')
    else:
        plot.savefig(path + '/histogram_of_' + varname + '_' + label + '.png')
    # print the plot
    plot.show()

# ATE estimation by mean differences
def ate_md(outcome, treatment):
    """
    Estimate ATE by differences in means.

    Parameters
    ----------
    outcome : TYPE: pd.Series
        DESCRIPTION: vector of outcomes
    treatment : TYPE: pd.Series
        DESCRIPTION: vector of treatments

    Returns
    -------
    results : ATE with Standard Error
    """
    # outcomes y according to treatment status by logical vector of True/False
    # set treated and control apart using the location for subsetting
    # using the .loc both labels as well as booleans are allowed
    y_1 = outcome.loc[treatment == 1]
    y_0 = outcome.loc[treatment == 0]
    # compute ATE and its standard error and t-value
    ate = y_1.mean() - y_0.mean()
    ate_se = np.sqrt(y_1.var() / len(y_1) + y_0.var() / len(y_0))
    ate_tval = ate / ate_se
    # compute pvalues based on the normal distribution (requires scipy)
    # sf stands for the survival function (also defined as 1 - cdf)
    ate_pval = stats.norm.sf(abs(ate_tval)) * 2  # twosided
    # alternatively ttest_ind() could be used directly
    # stats.ttest_ind(a=y_1, b=y_0, equal_var=False)
    # convert the dictionary to dataframe for a nicer output and name rows
    # pandas dataframe will automatically take the keys as columns if the
    # data input is a dictionary. Transpose for having the stats as columns
    result = pd.DataFrame([ate, ate_se, ate_tval, ate_pval],
                          index=['ATE', 'SE', 'tValue', 'pValue'],
                          columns=['MeanDiff']).transpose()
    # return and print result (\n inserts a line break)
    print('ATE Estimate by Difference in Means:', '-' * 80,
          'Dependent Variable: ' + outcome.name, '-' * 80,
          round(result, 2), '-' * 80, '\n\n', sep='\n')
    # return the resulting dataframe too
    return result

def histogram(data, variable, filter_variables, bins = 10):
    """
    Plot Continuous Variables.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: data including all variables
    variable : TYPE: String
        DESCRIPTION: target variable
    filter_variables : TYPE: n.array/list
        DESCRIPTION: variables to be filtered on
    bins : TYPE: int
        DESCRIPTION: bins for the histogrm bar width 

    Returns
    -------
    Plots and Saves all Histograms
    """
    filter_values_1 = data[filter_variables[0]].unique()
    filter_values_2 = data[filter_variables[1]].unique()
    for i in filter_values_1:
        for j in filter_values_2:
            plt.hist(data[(data[filter_variables[0]] == i) & (data[filter_variables[1]] == j)][variable], bins = bins)  
            plt.title(f"{variable} for {filter_variables[0]}={i} and {filter_variables[1]}={j}", size = 14)
            plt.ylabel('Count', size = 12)
            plt.xlabel('Value', size = 12)
            plt.savefig(f'histogram_{variable}_{i}_{j}.png', format='png')
            plt.show()
            
def histogram_change(data, variable, filter_variables, bins = 10):
    """
    Plot Continuous Variables.

    Parameters
    ----------
    variable : TYPE: String
        DESCRIPTION: target variable
    filter_variables : TYPE: n.array/list
        DESCRIPTION: variables to be filtered on
    bins : TYPE: int
        DESCRIPTION: bins for the histogrm bar width 

    Returns
    -------
    Plots and Saves all Histograms
    """
    filter_values_1 = data[filter_variables[0]].unique()
    filter_values_2 = data[filter_variables[1]].unique()
    for i in filter_values_2:
        plt.hist(data[(data[filter_variables[1]] == i) & (data[filter_variables[0]] == filter_values_1[0])][variable].to_numpy() - data[(data[filter_variables[1]] == i) & (data[filter_variables[0]] == filter_values_1[1])][variable].to_numpy(), bins = 30) 
        plt.title(f"Change in {variable} for {filter_variables[1]}={i} from 19{filter_values_1[0]} to 19={filter_values_1[1]}", size = 14)
        plt.ylabel('Count', size = 12)
        plt.xlabel('Value', size = 12)
        plt.savefig(f'histogram_{variable}_change_{i}_{filter_values_1[0]}_{filter_values_1[1]}.png', format='png')
        plt.show()
        
def dummy_check(data, variables):
    """
    Plot Continuous Variables.

    Parameters
    ----------
    data : TYPE: pd.DataFrame
        DESCRIPTION: data including all variables
    variable : TYPE: String
        DESCRIPTION: target variable

    Returns
    -------
    Returns a prints a table with fundamental checks for correct dummy variable specifications
    """
    overview = pd.DataFrame(index = variables, columns = ['Unique', 'Max', 'Min', '|', 'Check'])
    for i in variables:
        overview.loc[i, 'Unique'] = len(data[i].unique())
        overview.loc[i, 'Min'] = data[i].min()
        overview.loc[i, 'Max'] = data[i].max()
        overview.loc[i, '|'] = '|'
        if (overview.loc[i, 'Unique'] <= 2) & (overview.loc[i, 'Min'] >= 0) & (overview.loc[i, 'Max'] <= 1):
            overview.loc[i, 'Check'] = 'OK'
        else:
            overview.loc[i, 'Check'] = 'CHECK'
    print('\nDummy Variable Check:', '-' * 80, round(overview, 2), '-' * 80, sep = '\n')
    return overview   

def table(data, variables, filter_variables):
    """
    Plot Continuous Variables.

    Parameters
    ----------
    variables: TYPE: np.array/list
        DESCRIPTION: target variables
    filter_variables : TYPE: n.array/list
        DESCRIPTION: variables to be filtered on

    Returns
    -------
    Returns and prints a table with means and numbers of observations for target variables conditioned on filter variables
    """
    filter_values_1 = data[filter_variables[0]].unique()
    filter_values_2 = data[filter_variables[1]].unique()
    overview = pd.DataFrame(index = variables, columns = [f'Mean (year={filter_values_1[0]})', f'Mean (year={filter_values_1[1]})', f'Mean (state={filter_values_2[0]})', f'Mean (state={filter_values_2[1]})', f'Obs (year={filter_values_1[0]})', f'Obs (year={filter_values_1[1]})', f'Obs (state={filter_values_2[0]})', f'Obs (state={filter_values_2[1]})'])
    for variable in variables:
        for i in np.concatenate((filter_values_1, filter_values_2)):
            if i in filter_values_1:
                mean = data[data[filter_variables[0]] == i][variable].mean()
                obs = len(data[data[filter_variables[0]] == i][variable])
                overview.loc[variable, f'Mean (year={i})'] = mean
                overview.loc[variable, f'Obs (year={i})'] = obs
            elif i in filter_values_2:
                mean = data[data[filter_variables[1]] == i][variable].mean()
                obs = len(data[data[filter_variables[1]] == i][variable])
                overview.loc[variable, f'Mean (state={i})'] = mean
                overview.loc[variable, f'Obs (state={i})'] = obs
    print('\nMeans & Numbers of Observations:', '-' * 80, round(overview, 2), '-' * 80, sep = '\n')
    return overview  

def table_combined(data, variables, filter_variables):
    """
    Plot Continuous Variables.

    Parameters
    ----------
    variables: TYPE: np.array/list
        DESCRIPTION: target variables
    filter_variables : TYPE: n.array/list
        DESCRIPTION: variables to be filtered on

    Returns
    -------
    Returns and prints a table with means and numbers of observations for target variables combinedly conditioned on filter variables
    """
    filter_values_1 = data[filter_variables[0]].unique()
    filter_values_2 = data[filter_variables[1]].unique()
    overview = pd.DataFrame(index = variables, columns = [f'Mean (year={filter_values_1[0]}, state={filter_values_2[0]})', f'Mean (year={filter_values_1[0]}, state={filter_values_2[1]})', f'Mean (year={filter_values_1[1]}, state={filter_values_2[0]})', f'Mean (year={filter_values_1[1]}, state={filter_values_2[1]})', f'Obs (year={filter_values_1[0]}, state={filter_values_2[0]})', f'Obs (year={filter_values_1[0]}, state={filter_values_2[1]})', f'Obs (year={filter_values_1[1]}, state={filter_values_2[0]})', f'Obs (year={filter_values_1[1]}, state={filter_values_2[1]})'])
    for variable in variables:
        for i in filter_values_1:
            for j in filter_values_2:
                mean = data[(data[filter_variables[0]] == i) & (data[filter_variables[1]] == j)][variable].mean()
                obs = len(data[(data[filter_variables[0]] == i) & (data[filter_variables[1]] == j)][variable])
                overview.loc[variable, f'Mean (year={i}, state={j})'] = mean
                overview.loc[variable, f'Obs (year={i}, state={j})'] = obs
    print('\nMeans & Numbers of Observations (Combined):', '-' * 80, round(overview, 2), '-' * 80, sep = '\n')
    return overview  
