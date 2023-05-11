import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
file_name = 'co2_clustering.csv'
df = pd.read_csv(file_name, skiprows=4)
