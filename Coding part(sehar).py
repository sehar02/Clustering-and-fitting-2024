import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def read_data(file_path, value_name):
    """
    Read and preprocess data from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file.
    value_name (str): Name of the value column.

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(file_path, skiprows=4)
    df = df.melt(id_vars=['Country Name', 'Country Code'], var_name='Year', value_name=value_name)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    return df

# Reading datasets
co2_emissions = read_data('CO2 Emission.csv', 'CO2_Emissions')
gdp_per_capita = read_data('GDP per capita (current us).csv', 'GDP_per_Capita')
population = read_data('Population total.csv', 'Population')
renewable_energy = read_data('Renewable Energy.csv', 'Renewable_Energy')

# Merging datasets
df = co2_emissions.merge(gdp_per_capita, on=['Country Name', 'Country Code', 'Year'])
df = df.merge(renewable_energy, on=['Country Name', 'Country Code', 'Year'])
df = df.merge(population, on=['Country Name', 'Country Code', 'Year'])

# Dropping rows with missing values
df.dropna(inplace=True)

# Scaling data
features = df[['GDP_per_Capita', 'CO2_Emissions', 'Renewable_Energy', 'Population']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Calculating silhouette score
silhouette_avg = silhouette_score(scaled_features, df['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')

# Normalizing and backscaling cluster centers
df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
df_scaled['Cluster'] = kmeans.labels_
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=features.columns)
print(cluster_centers_df)

# Clustering graph
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='GDP_per_Capita', y='CO2_Emissions', hue='Cluster', palette='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red', label='Cluster Centers')
plt.title('Clustering of Countries based on GDP and CO2 Emissions')
plt.xlabel('GDP per Capita (current US$)')
plt.ylabel('CO2 Emissions per Capita (metric tons)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Heatmap for correlation
correlation_matrix = df[['GDP_per_Capita', 'CO2_Emissions', 'Renewable_Energy', 'Population']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Time series analysis with subplots for selected countries
countries = ['United States', 'China', 'India']
for country in countries:
    country_data = df[df['Country Name'] == country]
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_title(f'Time Series Analysis for {country}')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('CO2 Emissions per Capita (metric tons)', color='tab:blue')
    ax1.plot(country_data['Year'], country_data['CO2_Emissions'], label='CO2 Emissions', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('GDP per Capita (current US$)', color='tab:orange')
    ax2.plot(country_data['Year'], country_data['GDP_per_Capita'], label='GDP per Capita', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Renewable Energy (% of total)', color='tab:green')
    ax3.plot(country_data['Year'], country_data['Renewable_Energy'], label='Renewable Energy', color='tab:green')
    ax3.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9), bbox_transform=ax1.transAxes)
    plt.show()

def exponential_growth(x, a, b, c):
    """
    Exponential growth model.

    Parameters:
    x (array-like): Independent variable.
    a (float): Parameter a.
    b (float): Parameter b.
    c (float): Parameter c.

    Returns:
    array-like: Dependent variable.
    """
    return a * np.exp(b * (x - c))

# Data fitting for United States CO2 emissions
x_data = df[df['Country Name'] == 'United States']['Year']
y_data = df[df['Country Name'] == 'United States']['CO2_Emissions']

params, covariance = curve_fit(exponential_growth, x_data, y_data, p0=[1, 0.01, 2000])

# Predicting and visualizing fit
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = exponential_growth(x_fit, *params)

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_fit, y_fit, color='red', label='Exponential Fit')
plt.fill_between(x_fit, y_fit - 1.96 * np.sqrt(np.diag(covariance)[1]), y_fit + 1.96 * np.sqrt(np.diag(covariance)[1]), color='pink', alpha=0.3, label='Confidence Interval')
plt.title('Exponential Fit for CO2 Emissions in the United States')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions per Capita (metric tons)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Predictions with uncertainty
future_years = np.array([2030, 2040, 2050])
future_predictions = exponential_growth(future_years, *params)
print(f'Predicted CO2 Emissions for future years: {future_predictions}')

# Plot predictions
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Historical Data')
plt.plot(x_fit, y_fit, color='red', label='Exponential Fit')
plt.errorbar(future_years, future_predictions, yerr=1.96 * np.sqrt(np.diag(covariance)[1]), fmt='o', color='blue', label='Predictions')
plt.fill_between(x_fit, y_fit - 1.96 * np.sqrt(np.diag(covariance)[1]), y_fit + 1.96 * np.sqrt(np.diag(covariance)[1]), color='pink', alpha=0.3, label='Confidence Interval')
plt.title('Predicted CO2 Emissions for the United States')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions per Capita (metric tons)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()