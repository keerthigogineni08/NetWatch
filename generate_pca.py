import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load raw IoT data
df = pd.read_csv("output/iot_clustered.csv")

# === 1. Group by device to get mean and std of frame lengths
device_stats = df.groupby("device")["frame_len"].agg(["mean", "std"]).reset_index()

# === 2. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(device_stats[["mean", "std"]])

# === 3. Apply PCA (only if more than 1 sample)
if X_scaled.shape[0] >= 2:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    device_stats["PCA1"] = X_pca[:, 0]
    device_stats["PCA2"] = X_pca[:, 1]
else:
    device_stats["PCA1"] = 0
    device_stats["PCA2"] = 0

# === 4. Merge cluster labels from original df
device_stats = device_stats.merge(df[["device", "cluster"]].drop_duplicates(), on="device", how="left")

# === 5. Save final file for the dashboard
device_stats.to_csv("output/iot_clustered_pca.csv", index=False)
print("✅ PCA data saved to output/iot_clustered_pca.csv")
