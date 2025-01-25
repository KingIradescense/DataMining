import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
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

# Calculate the variance of each feature
feature_variance = df_encoded.var()
print("Feature Variances:")
print(feature_variance)

# Plot the variance of each feature before applying VarianceThreshold
plt.figure(figsize=(10, 6))
feature_variance.plot(kind='bar', color='skyblue')
plt.title("Variance of Each Feature Before Variance Threshold")
plt.xlabel("Features")
plt.ylabel("Variance")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.show()

# Apply Variance Threshold with a threshold of 0.1
selector = VarianceThreshold(threshold=0.1)
df_selected = selector.fit_transform(df_encoded)

# Get the selected features (those with variance > 0.1)
selected_features = df_encoded.columns[selector.get_support()]

# Show selected features
print("Selected Features after Variance Threshold (0.1):", selected_features)

# Visualize the selected features after applying VarianceThreshold
plt.figure(figsize=(10, 6))
selected_variance = feature_variance[selector.get_support()]
selected_variance.plot(kind='bar', color='lightgreen')
plt.title("Selected Features After Variance Threshold (0.1)")
plt.xlabel("Selected Features")
plt.ylabel("Variance")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(np.arange(0, step=0.2),rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.show()


