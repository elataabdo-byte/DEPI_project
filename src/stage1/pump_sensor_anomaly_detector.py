import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_file = "sensor_baseline.csv"
df = pd.read_csv(input_file)
 
df = df.interpolate(method='linear', limit=5, limit_direction='both')
df = df.bfill().ffill()

data_matrix = df.values
mean_vector = np.mean(data_matrix, axis=0)
cov_matrix = np.cov(data_matrix, rowvar=False)

cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
inv_cov_matrix = np.linalg.inv(cov_matrix)

def mahalanobis_dist(row, mean, inv_cov):
    diff = row - mean
    return np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))

distances = [mahalanobis_dist(row, mean_vector, inv_cov_matrix) for row in data_matrix]
df['Mahalanobis_Distance'] = distances

threshold = np.percentile(distances, 99)
df['Anomaly'] = df['Mahalanobis_Distance'] > threshold

df.to_csv("anomaly_results.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(df['Mahalanobis_Distance'].values, label='Mahalanobis Distance', color='blue')
plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
plt.title('Sensor Anomaly Detection - Baseline Mode')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
