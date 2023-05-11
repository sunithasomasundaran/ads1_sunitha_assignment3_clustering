import pandas as pd
import matplotlib.pyplot as plt

def scaler(df):
    """ Expects a dataframe and normalises all 
        columnsto the 0-1 range. It also returns 
        dataframes with minimum and maximum for
        transforming the cluster centres"""

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max

# Read CSV
file_name = 'co2_clustering.csv'
df = pd.read_csv(file_name, skiprows=4)

# Selecting the columns to be used for clustering
columns_to_use = [str(year) for year in range(1990, 2020)]
df_years = df[['Country Name', 'Country Code'] + columns_to_use]

# Fill missing values with the mean
df_years = df_years.fillna(df_years.mean())

# Normalize the data
df_norm, df_min, df_max = scaler(df_years[columns_to_use])


