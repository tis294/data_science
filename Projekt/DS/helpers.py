#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   

#1.1
def convert_columns_to_datetime(df, columns):
    """
    Converts specified columns of dtype 'object' in a Pandas DataFrame to dtype 'datetime'.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        columns (list): A list of column names to convert.

    Returns:
        pandas.DataFrame: The modified DataFrame with converted columns.
    """

    for column in columns:
        try:
            df[column] = pd.to_datetime(df[column])
        except ValueError:
            print(f"Error converting column '{column}' to datetime. Invalid date format.")
    
    return df

#1.2
def NaN_col_info(df, nan_col, dist_col):
    '''
    Create Graphs to compare infos of rows with NaN-value in input column with Not-NaN values.

    Input: 
    - nan_col = column with NaN values 
    - dist_col = column to plot the distribution for

    Return: 
    Graphs of dist_col distribution, separated by NaN & Not-NaN values of nan_col
    '''
    # Create main figure
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10, 8))
    fig.suptitle(f'{dist_col} Distribution for NaN and Not-NaN values of {nan_col} column')

    # Plotting distribution of policy_startDate for NaN & not-NaN
    sns.histplot(df[df[nan_col].isna() == False][dist_col], kde=True, bins=10, color='blue', label='Not-NaN', ax=ax1) #Not-NaN -> blue
    sns.histplot(df[df[nan_col].isna() == True][dist_col], kde=True, bins=10, color='red', label='NaN', ax=ax2) #NaN -> red
     
    ax1.set_ylabel('Distribution of Not-NaN rows')
    ax2.set_ylabel('Distribution of NaN rows')
    ax2.set_xlabel(dist_col)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()

#1.3
def plot_distributions(df, nan_col, comp_cols):
    """
    Generates a graph showing the distributions of specified columns in 'comp_cols'
    based on NaN values and non-NaN values of the 'nan_col' column.

    Args:
        nan_col (str): The name of the column with NaN values.
        comp_cols (list): A list of column names to analyze.
        df (pandas.DataFrame): The DataFrame containing the data. (Default: df)
        
    Returns:
        None (displays a graph)

    """
    nan_values = df[df[nan_col].isna()]
    non_nan_values = df[df[nan_col].notna()]
    
    for col in comp_cols:
        nan_dist = nan_values[col].value_counts(normalize=True)
        non_nan_dist = non_nan_values[col].value_counts(normalize=True)
        
        plt.figure(figsize=(10, 6))
        plt.bar(nan_dist.index, nan_dist.values, label='NaN Values', alpha=0.5)
        plt.bar(non_nan_dist.index, non_nan_dist.values, label='Non-NaN Values', alpha=0.5)
        plt.xlabel(col)
        plt.ylabel('Distribution')
        plt.title(f'Distribution of {col} for NaN and Non-NaN Values of {nan_col}')
        plt.legend()
        plt.show()
        
#1.4
def get_unique_values(df, columns=['Nation', 'premium_Country']):
    """
    Get unique values from specified columns in a DataFrame.

    Parameters:
        df (pandas.DataFrame, optional): Input DataFrame. (Default: df)
        columns (list, optional): List of column names. Defaults to ['Nation', 'premium_Country'].

    Returns:
        list: List of unique values from the specified columns.

    Example Usage:
        unique_values = get_unique_values(columns=['Nation', 'premium_Country'])
        print("Unique Values:", unique_values)
    """
    unique_values = df[columns].values.ravel()
    unique_values = pd.unique(unique_values).tolist()
    return unique_values
        
#1.5 
import requests

def generate_country_region_mapping(iso_codes):
    """
    Generates a dictionary mapping country ISO codes to their corresponding regions.

    Parameters:
        iso_codes (list): List of unique country ISO codes.

    Returns:
        dict: Dictionary mapping ISO codes to regions.

    Example Usage:
        iso_codes = ['PL', 'DE', 'RO', ...]  # List of ISO codes
        mapping = generate_country_region_mapping(iso_codes)
        print("Country to Region Mapping:", mapping)
    """
    mapping = {}

    # Special feature: 'DE' has its own region called 'DE'
    mapping['DE'] = 'DE'

    # Special feature: 'XX' has its own region called 'XX'
    mapping['XX'] = 'XX'

    # Make a request to the REST Countries API
    response = requests.get('https://restcountries.com/v2/all')

    if response.status_code == 200:
        countries_data = response.json()

        # Iterate over the countries data and extract ISO code and region information
        for country_data in countries_data:
            iso_code = country_data['alpha2Code']
            region = country_data['region']

            if iso_code in iso_codes:
                if iso_code == 'DE':
                    mapping[iso_code] = 'DE'
                elif iso_code == 'XX':
                    mapping[iso_code] = 'XX'
                else:
                    mapping[iso_code] = region

    return mapping

#1.6
def premium_status_timeline(df, index):
    '''
    Input: 
    df
    index = number of row in df sorted by sum of premiumAmount with status_code = 'S'
    
    Output:
    prints: df with sum of premium amounts of ContractID for each status
    returns: df with all premium amounts of ContractID, their status and premiumMonth, sorted by premiumMonth
    '''
    #check if input index is in range
    n = len(df.loc[df.status_code == 'S'][['ContractID']])
    if index not in range(-n, n):
        return 'index out of range'
    else:
        #create df with contractIDs ordered by sum of premiumAmount of cancelled lines 
        S_sums = df.loc[df.status_code == 'S'][['ContractID', 'premiumAmount']].groupby(['ContractID']).sum('premiumAmount').sort_values('premiumAmount', ascending=False)

        #get contractID of input index
        ID = S_sums.index[index]

        #print premium overview per status for this ID
        print(f'premiumAmount overview for contract ID {ID}:\n',
            df.loc[df.ContractID == ID][['premiumMonth', 'status_code', 'premiumAmount']].groupby(['status_code']).sum('premiumAmount'))

        #create and return overview of premiums for this ID
        premiums = df.loc[df.ContractID == ID][['premiumMonth', 'status_code', 'premiumAmount']].sort_values('premiumMonth')
        premiums.set_index('premiumMonth')
        return premiums