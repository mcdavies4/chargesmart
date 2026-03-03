import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import os

if not os.path.exists("enriched_data.csv"):
    print("No enriched_data.csv found!")
    exit()

print("Loading data...")
df = pd.read_csv("enriched_data.csv")
print(f"Total records: {len(df):,}")

# Clean capacity
df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(1)
df["is_weekend"] = df["is_weekend"].astype(int)

# Add country as numeric feature
country_map = {'UK': 0, 'US': 1, 'EU': 2}
if 'country' in df.columns:
    df['country_code'] = df['country'].map(country_map).fillna(0).astype(int)
else:
    df['country_code'] = 0

# Encode location type
if 'location_type' in df.columns:
    loc_map = {
        'motorway': 3, 'supermarket': 2, 'retail': 2,
        'council': 1, 'tesla': 2, 'other': 1
    }
    df['location_code'] = df['location_type'].map(loc_map).fillna(1).astype(int)
else:
    df['location_code'] = 1

# Better occupancy simulation - tuned for each country
np.random.seed(42)

def simulate_occupancy(row):
    hour = row["hour"]
    capacity = row["capacity"]
    is_weekend = row["is_weekend"]
    country = row.get("country_code", 0)
    loc = row.get("location_code", 1)

    # Base by hour - strong pattern
    hour_probs = {
        0:0.05, 1:0.04, 2:0.03, 3:0.03, 4:0.04, 5:0.08,
        6:0.15, 7:0.35, 8:0.55, 9:0.50, 10:0.35, 11:0.40,
        12:0.50, 13:0.45, 14:0.35, 15:0.40, 16:0.50,
        17:0.65, 18:0.70, 19:0.60, 20:0.40, 21:0.25,
        22:0.15, 23:0.08
    }
    busy_probability = hour_probs.get(hour, 0.2)

    # Weekend shifts peak to midday
    if is_weekend:
        if hour in [10, 11, 12, 13, 14, 15]:
            busy_probability += 0.20
        if hour in [7, 8, 9, 17, 18, 19]:
            busy_probability -= 0.15

    # Capacity - strong effect
    if capacity == 1:
        busy_probability += 0.30
    elif capacity == 2:
        busy_probability += 0.15
    elif capacity >= 8:
        busy_probability -= 0.20
    elif capacity >= 4:
        busy_probability -= 0.10

    # Location type - distinct patterns
    if loc == 3:  # motorway - busy all day
        busy_probability += 0.20
    elif loc == 2:  # supermarket/tesla - busy lunch and evenings
        if hour in [11, 12, 13, 14, 17, 18]:
            busy_probability += 0.25
    elif loc == 1:  # council - busy office hours
        if hour in [8, 9, 10, 16, 17, 18]:
            busy_probability += 0.15

    # Country patterns - distinct EV adoption rates
    if country == 0:  # UK - moderate adoption
        busy_probability += 0.05
    elif country == 1:  # US - varies hugely by city
        busy_probability += 0.08
    elif country == 2:  # EU - Norway/Netherlands very high adoption
        busy_probability += 0.15
        if hour in [7, 8, 9, 16, 17, 18]:
            busy_probability += 0.10

    busy_probability = min(max(busy_probability, 0.02), 0.97)
    return np.random.binomial(1, busy_probability)

print("Simulating occupancy patterns...")
df["is_busy"] = df.apply(simulate_occupancy, axis=1)

busy_count = df['is_busy'].sum()
free_count = (df['is_busy'] == 0).sum()
print(f"Busy: {busy_count:,} ({busy_count/len(df)*100:.1f}%)")
print(f"Free: {free_count:,} ({free_count/len(df)*100:.1f}%)")

# Features - now includes country and location
features = ["hour", "day_of_week", "is_weekend", "capacity", "lat", "lon", "country_code", "location_code"]
X = df[features]
y = df["is_busy"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining on {len(X_train):,} records...")
print("This may take a few minutes with 4 million records...\n")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Feature importance
importances = model.feature_importances_
print("\nFeature importance:")
for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")

with open("charger_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature list so app.py knows what to feed the model
with open("model_features.txt", "w") as f:
    f.write(",".join(features))

print("\nModel saved!")
print("Done!")
