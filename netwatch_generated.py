#!/usr/bin/env python
# coding: utf-8

# In[1]:


# NetWatch_GeneratedData_Models.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

# === 1. LOAD & CLEAN GENERATED DATA ===

print("\n📥 Loading simulated NetWatch data...")
df = pd.read_csv("Generated_data/netwatch_wifi_data_dirty.csv")

df = df.sort_values(['home_id', 'device_id', 'timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Visual inspection
print(df.info())
print(df.describe())
print(df.head())


# ## Visualize Nulls, Outliers, Weird Disconnects 

# In[2]:


# === 2. EXPLORATION ===

print("\n🧼 Visualizing missing/nulls...")
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['rssi'], kde=True, ax=ax[0])
sns.histplot(df['latency_ms'], kde=True, ax=ax[1])
sns.histplot(df['jitter_ms'], kde=True, ax=ax[2])
ax[0].set_title("RSSI Distribution")
ax[1].set_title("Latency Distribution")
ax[2].set_title("Jitter Distribution")
plt.tight_layout()
plt.show()

# Weird disconnects
weird = df[(df['is_connected'] == 0) & (df['rssi'] > -65) & (df['latency_ms'] < 80)]
print("⚠️ Weird Disconnects:", weird.shape[0])


# In[3]:


# === 3. CLEANING ===

print("\n🧹 Cleaning simulated dataset...")
df.drop_duplicates(inplace=True)
df['rssi'].fillna(df['rssi'].median(), inplace=True)
df['latency_ms'].fillna(df['latency_ms'].median(), inplace=True)
df['jitter_ms'].fillna(df['jitter_ms'].median(), inplace=True)
df['jitter_ms'] = df['jitter_ms'].clip(upper=300)

print("\n🛠️ Labeling predicted outages based on real + synthetic logic...")

# Drop rows with missing signal values
df.dropna(subset=['rssi', 'latency_ms', 'jitter_ms', 'packet_loss'], inplace=True)

# 💥 Clear any existing labels (to prevent stale 0-only labels)
if 'predicted_outage' in df.columns:
    df.drop(columns=['predicted_outage'], inplace=True)

# Step 1: Apply realistic outage logic
df['predicted_outage'] = (
    (df['rssi'] < -82) &
    (df['latency_ms'] > 220) &
    (df['packet_loss'] > 0.02)
).astype(int)

# Step 2: Force synthetic outages if < 5%
num_outages = df['predicted_outage'].sum()
min_required = int(0.05 * len(df))  # 5% of total

if num_outages < min_required:
    num_to_add = min_required - num_outages
    print(f"⚠️ Only {num_outages} real outages found — injecting {num_to_add} synthetic ones...")
    synthetic = df[df['predicted_outage'] == 0].sample(n=num_to_add, random_state=42).index
    df.loc[synthetic, 'predicted_outage'] = 1

# Final check
print("✅ Final outage distribution:")
print(df['predicted_outage'].value_counts())

df['timestamp'] = pd.to_datetime(df['timestamp'])
df.sort_values(by=['home_id', 'device_id', 'timestamp'], inplace=True)
df.to_csv("Generated_data/netwatch_wifi_cleaned.csv", index=False)


# In[4]:


# === 4. MODEL TRAINING ===

print("\n📊 ML Training for Regression + Classification")
df = pd.read_csv("Generated_data/netwatch_wifi_cleaned.csv")
print("🧪 Checking reloaded class distribution:")
print(df['predicted_outage'].value_counts())

df_model = df.dropna(subset=['rssi', 'latency_ms', 'jitter_ms', 'packet_loss', 'experience_score', 'predicted_outage'])
features = ['rssi', 'latency_ms', 'jitter_ms', 'packet_loss']

# Regression – Experience Score
X = df_model[features]
y_reg = df_model['experience_score']
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("\n📈 RMSE (experience_score):", rmse)
joblib.dump(regressor, "models/experience_score_model.pkl")

# Classification – Outage Prediction
y_clf = df_model['predicted_outage']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)


# Show class balance
print("Class distribution in training set:")
print(y_train_c.value_counts())

# Train classifier with class_weight to handle imbalance
classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
classifier.fit(X_train_c, y_train_c)

# Predict probabilities instead of just 0/1
y_probs = classifier.predict_proba(X_test_c)[:, 1]

# Custom threshold – make it more sensitive
threshold = 0.3
y_pred_c = (y_probs > threshold).astype(int)

# Evaluate
acc = accuracy_score(y_test_c, y_pred_c)
print("\n🚨 Accuracy (outage prediction):", acc)
print(classification_report(y_test_c, y_pred_c))

# Save model and threshold together
model_bundle = {"model": classifier, "threshold": threshold}
joblib.dump(model_bundle, "models/outage_detector_model.pkl")




# In[5]:


# === 5. EXPERIENCE SCORE TREND ===

print("\n📈 Plotting experience score trend...")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
df['experience_score'].resample('1H').mean().plot(figsize=(12,4), title="📊 Avg Experience Score Over Time")
plt.ylabel("Experience Score")
plt.tight_layout()
plt.show()


# In[6]:


# === 6. OUTAGE DETECTION LOGIC ===
print("\n⚠️ Running outage detection logic...")

# Ensure timestamp exists
if 'timestamp' not in df.columns:
    df = df.reset_index()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['home_id', 'timestamp'])

# Set datetime index
df = df.set_index('timestamp')

# Filter disconnected events only
disconnect_df = df[df['is_connected'] == 0].copy()

# We need a numeric column for rolling, so let's use 'is_connected'
disconnect_df['dummy'] = 1  # just a placeholder

# Store alerts here
alerts_list = []

# Group by home_id
for home_id, group in disconnect_df.groupby("home_id"):
    rolled = group.rolling('15min')

    # Create rolling window list of device_ids
    for window_end in group.index:
        window_start = window_end - pd.Timedelta(minutes=15)
        window = group.loc[window_start:window_end]
        unique_devices = window['device_id'].nunique()

        if unique_devices >= 3:
            alerts_list.append({
                "home_id": home_id,
                "timestamp": window_end,
                "affected_devices": unique_devices,
                "alert_type": "Multi-device dropout",
                "severity": "High"
            })

# Convert to DataFrame
alerts_df = pd.DataFrame(alerts_list)
print(alerts_df.head())
alerts_df.to_csv("logs/incident_log.csv", index=False)


# In[7]:


# === 7. TIME SERIES EXPLORER EXAMPLE ===

print("\n📈 Sample time series per home")
df.reset_index(inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
home_sample = df[df['home_id'] == df['home_id'].iloc[0]]
home_sample.set_index('timestamp')['rssi'].plot(figsize=(12,4), title=f"Signal Over Time – {home_sample['home_id'].iloc[0]}")
plt.tight_layout()
plt.show()


# In[8]:


# === 8. CLUSTERING BEHAVIOR PROFILES ===

print("\n🧠 Clustering device behavior...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model[features])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=4, random_state=42)
df_model['cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,5))
plt.scatter(pca_result[:,0], pca_result[:,1], c=df_model['cluster'], cmap='tab10')
plt.title("🧠 PCA + KMeans: WiFi Behavior Clustering")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()

