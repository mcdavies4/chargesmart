import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import os

if not os.path.exists("enriched_data.csv"):
    print("No enriched_data.csv found!")
    exit()

print("Loading data...")
df = pd.read_csv("enriched_data.csv", usecols=[
    'hour', 'day_of_week', 'is_weekend', 'capacity',
    'lat', 'lon', 'location_type', 'country'
])
print(f"Total records: {len(df):,}")

# ── CLEAN ────────────────────────────────────────────────────
df["capacity"]   = pd.to_numeric(df["capacity"], errors="coerce").fillna(1).clip(1, 20)
df["is_weekend"] = df["is_weekend"].astype(int)

country_map = {'UK': 0, 'US': 1, 'EU': 2}
df['country_code'] = df['country'].map(country_map).fillna(0).astype(int)

loc_map = {'motorway': 3, 'supermarket': 2, 'retail': 2, 'council': 1, 'tesla': 2, 'other': 1}
df['location_code'] = df['location_type'].map(loc_map).fillna(1).astype(int)

# ── OCCUPANCY SIMULATION ─────────────────────────────────────
np.random.seed(42)

def simulate_occupancy(row):
    p = 0.15
    hour    = row["hour"]
    country = row["country_code"]
    loc     = row["location_code"]
    cap     = row["capacity"]

    # Time of day
    if hour in [7, 8, 9]:    p += 0.35
    if hour in [17, 18, 19]: p += 0.40
    if hour in [12, 13]:     p += 0.20
    if hour in [0, 1, 2, 3]: p -= 0.12

    # Weekend
    if row["is_weekend"]:
        if hour in [10, 11, 12, 13, 14, 15]: p += 0.25
        else:                                 p -= 0.05
    else:
        p += 0.10

    # Capacity
    if cap == 1:   p += 0.25
    elif cap == 2: p += 0.10
    elif cap >= 6: p -= 0.15

    # Location
    if loc == 3:   p += 0.20
    elif loc == 2: p += 0.10

    # Country
    if country == 1: p += 0.05
    elif country == 2: p += 0.08

    return np.random.binomial(1, float(np.clip(p, 0.02, 0.95)))

print("Simulating occupancy patterns...")
df["is_busy"] = df.apply(simulate_occupancy, axis=1)

busy = df['is_busy'].sum()
free = len(df) - busy
print(f"Busy: {busy:,} ({busy/len(df)*100:.1f}%)")
print(f"Free: {free:,} ({free/len(df)*100:.1f}%)")

# ── TRAIN ────────────────────────────────────────────────────
features = [
    "hour", "day_of_week", "is_weekend",
    "capacity", "lat", "lon",
    "country_code", "location_code",
]

X = df[features]
y = df["is_busy"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining on {len(X_train):,} records with {len(features)} features...")
print("This may take 5-10 minutes...\n")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=14,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy * 100:.2f}%")

print("\nFeature importance:")
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 100)
    print(f"  {feat:<20} {imp:.3f}  {bar}")

with open("charger_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model_features.txt", "w") as f:
    f.write(",".join(features))

print("\nModel saved to charger_model.pkl")
print("Done!")
