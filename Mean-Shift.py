import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv("Bicycle_Parking_20241213_AfterConversion.csv")

# Relevant features for clustering based on Variance Threshold
features = ['Longitude','Latitude','BoroCode',  "HrcEvac",'RackType_Large Hoop','RackType_U Rack','RackType_Small Hoop'] 

# Standardize the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])


#Mean Shift clustering Algorithm
mean_shift = MeanShift(bandwidth=2)
df['Cluster'] = mean_shift.fit_predict(df_scaled)

df.to_csv("bicycle_data_with_clusters.csv", index=False)


# Visualizing the clustering result 
plt.figure(figsize=(8, 6))
plt.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'], cmap='cividis', marker='o')
plt.title("Mean Shift Cluster Results")
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Cluster')
plt.show()

# Display the number of clusters
num_clusters = len(set(df['Cluster']))
print(f"Number of clusters identified: {num_clusters}")