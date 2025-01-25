import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Load the dataset
df = pd.read_csv("Bicycle_Parking_20241213_Updated.csv")

# Check for missing values and drop them
print(df.isnull().sum())
df = df.dropna()  

#Get non-numeric values
non_numeric_columns = df.select_dtypes(exclude=np.number).columns
print("The following columns are non-numeric")
print(non_numeric_columns)
#Index(['Program', 'RackType', 'FEMAFldz'], dtype='object')


# Perform One-Hot Encoding on non-numeric columns
df_encoded = pd.get_dummies(df, columns=non_numeric_columns, drop_first=True)
print("Data after One-Hot Encoding:")
print(df_encoded.head())

# Inspect encoded data
print("Encoded DataFrame Info:")
print(df_encoded.info())

# Identify boolean columns (created by One-Hot Encoding)
bool_columns = df_encoded.select_dtypes(include=bool).columns
print("Boolean Columns Found After One-Hot Encoding:", bool_columns)

# Convert boolean columns to numeric (True = 1, False = 0)
df_encoded[bool_columns] = df_encoded[bool_columns].astype(int)

# Confirm conversion
print("After Converting Boolean Columns:")
print(df_encoded[bool_columns].head())


# Compute the correlation matrix
correlation_matrix = df_encoded.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(100, 80))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
correlation_matrix.to_csv("correlation_matrix.csv")