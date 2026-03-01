import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

import os

if not os.path.exists("enriched_data.csv"):
    print("Generating synthetic training data...")
    import numpy as np
    rows = []
    for _ in range(10000):
        hour = np.random.randint(0, 24)
        day = np.random.randint(0, 7)
        is_weekend = day >= 5
        capacity = np.random.choice([1, 2, 4, 8])
        lat = np.random.uniform(50.0, 56.0)
        lon = np.random.uniform(-3.0, 0.5)
        rows.append([f"2026-01-01 {hour:02d}:00:00", hour, day, 
                     is_weekend, "other", capacity, lat, lon, "unknown"])
    import csv
    with open("enriched_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","hour","day_of_week","is_weekend",
                         "location_type","capacity","lat","lon","operator"])
        writer.writerows(rows)
    print("Synthetic data generated!")

# Load your enriched data
df = pd.read_csv("enriched_data.csv")

# Clean capacity column
df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(1)

# Convert is_weekend to integer
df["is_weekend"] = df["is_weekend"].astype(int)

# Create realistic busy/free simulation
np.random.seed(42)
def simulate_occupancy(row):
    busy_probability = 0.1
    if row["hour"] in [8, 9]:
        busy_probability += 0.4
    if row["hour"] in [17, 18, 19]:
        busy_probability += 0.4
    if row["hour"] in [12, 13]:
        busy_probability += 0.2
    if row["is_weekend"] == 0:
        busy_probability += 0.2
    if row["capacity"] == 1:
        busy_probability += 0.2
    busy_probability = min(busy_probability, 0.95)
    return np.random.binomial(1, busy_probability)

df["is_busy"] = df.apply(simulate_occupancy, axis=1)

print(f"Busy chargers: {df['is_busy'].sum()}")
print(f"Free chargers: {(df['is_busy']==0).sum()}")

# Features
features = ["hour", "day_of_week", "is_weekend", "capacity", "lat", "lon"]
X = df[features]
y = df["is_busy"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Training AI model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy * 100:.2f}%")
print("AI model trained successfully!")

with open("charger_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved!")