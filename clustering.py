import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.metrics import r2_score

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

def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr

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

# Find the optimal number of clusters using the silhouette method
silhouette_scores = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(df_norm)
    silhouette_avg = silhouette_score(df_norm, cluster_labels)
    silhouette_scores.append(silhouette_avg)
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Apply the KMeans clustering algorithm
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df_years['Cluster'] = kmeans.fit_predict(df_norm)

# Plot the clustering results
plt.figure(figsize=(12, 8))
for i in range(optimal_clusters):
    plt.scatter(df_years[df_years['Cluster'] == i].index, df_years[df_years['Cluster'] == i]['2019'], label=f'Cluster {i}')

# Plot the cluster centers
cluster_centers = backscale(kmeans.cluster_centers_, df_min, df_max)
for i in range(optimal_clusters):
    plt.scatter(len(df_years), cluster_centers[i, -1], marker='*', s=150, c='black', label=f'Cluster Center {i}')

plt.title('Total Greenhouse Gas Emissions Clustering')
plt.xlabel('Country Index')
plt.ylabel('Total Greenhouse Gas Emissions (kt of CO2 equivalent) in 2019')
plt.legend()
plt.show()

# Display countries in each cluster
for i in range(optimal_clusters):
    cluster_countries = df_years[df_years['Cluster'] == i][['Country Name', 'Country Code']]
    print(f'Countries in Cluster {i}:')
    print(cluster_countries)
    print()
    
# Fitting the data to a linear model

# Read CSV
file_name = 'co2_clustering.csv'
df = pd.read_csv(file_name, skiprows=4)

# Selecting the columns to be used
columns_to_use = [str(year) for year in range(1990, 2020)]
df_years = df[['Country Name', 'Country Code'] + columns_to_use]

# Fill missing values with the mean
df_years = df_years.fillna(df_years.mean())

# Fit a polynomial model for a specific country
country = 'China'
df_country = df_years[df_years['Country Name'] == country][columns_to_use].values.flatten()

# X values (years)
x = np.arange(1990, 2020)

# Fit the model
degree = 3  # Degree of the polynomial fit
coefficients = np.polyfit(x, df_country, degree)
polynomial_model = np.poly1d(coefficients)

# Make predictions
x_pred = np.arange(1990, 2040)  # Predict 20 years into the future
y_pred = polynomial_model(x_pred)

# Calculate R-squared
r2 = r2_score(df_country, polynomial_model(x))

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(x, df_country, 'o', label='Actual Data')
plt.plot(x_pred, y_pred, label=f'Polynomial Fit (Degree: {degree}, R^2: {r2:.4f})')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (metric tons per capita)')
plt.title(f'{country} CO2 Emissions - Polynomial Model')
plt.legend()
plt.show()

# Comparing First Cluster Result

# Read CSV
file_name = 'co2_clustering.csv'
df = pd.read_csv(file_name, skiprows=4)

# Selecting the columns to be used
columns_to_use = [str(year) for year in range(1990, 2020)]
df_years = df[['Country Name', 'Country Code'] + columns_to_use]

# Fill missing values with the mean
df_years = df_years.fillna(df_years.mean())

# Choose countries
countries = ['Zimbabwe', 'Zambia', 'Ghana', 'Yemen, Rep.']

# Plot the data for the chosen countries
plt.figure(figsize=(12, 8))

for country in countries:
    df_country = df_years[df_years['Country Name'] == country][columns_to_use].values.flatten()
    x = np.arange(1990, 2020)
    plt.plot(x, df_country, label=country)

plt.xlabel('Year')
plt.ylabel('CO2 Emissions (metric tons per capita)')
plt.title('CO2 Emissions by Country (1990-2019)')
plt.legend()
plt.show()

#Comparing Second Cluster Results

# Read CSV
file_name = 'co2_clustering.csv'
df = pd.read_csv(file_name, skiprows=4)

# Selecting the columns to be used
columns_to_use = [str(year) for year in range(1990, 2020)]
df_years = df[['Country Name', 'Country Code'] + columns_to_use]

# Fill missing values with the mean
df_years = df_years.fillna(df_years.mean())

# Choose countries
countries = ['High income', 'Upper middle income', 'Middle income', 'Late-demographic dividend']

# Plot the data for the chosen countries
plt.figure(figsize=(12, 8))

for country in countries:
    df_country = df_years[df_years['Country Name'] == country][columns_to_use].values.flatten()
    x = np.arange(1990, 2020)
    plt.plot(x, df_country, label=country)

plt.xlabel('Year')
plt.ylabel('CO2 Emissions (metric tons per capita)')
plt.title('CO2 Emissions by Country (1990-2019)')
plt.legend()
plt.show()


