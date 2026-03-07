import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import os

if not os.path.exists("enriched_data.csv"):
    print("No enriched_data.csv found!")
    exit()

print("Loading data...")
df = pd.read_csv("enriched_data.csv")
print(f"Total records: {len(df):,}")

# ── CLEAN ────────────────────────────────────────────────────
df["capacity"]          = pd.to_numeric(df["capacity"], errors="coerce").fillna(1).clip(1, 20)
df["hour"]              = pd.to_numeric(df["hour"], errors="coerce").fillna(12).astype(int)
df["day_of_week"]       = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0).astype(int)
df["is_weekend"]        = df["is_weekend"].astype(int) if "is_weekend" in df.columns else ((df["day_of_week"] >= 5).astype(int))
df["is_bank_holiday"]   = df["is_bank_holiday"].astype(int)   if "is_bank_holiday"   in df.columns else 0
df["is_school_holiday"] = df["is_school_holiday"].astype(int) if "is_school_holiday" in df.columns else 0
df["is_summer"]         = df["is_summer"].astype(int)         if "is_summer"         in df.columns else 0
df["temperature"]       = pd.to_numeric(df.get("temperature", 15), errors="coerce").fillna(15)
df["precipitation"]     = pd.to_numeric(df.get("precipitation", 0), errors="coerce").fillna(0)

# Encode categoricals
country_map  = {'UK': 0, 'US': 1, 'EU': 2, 'ZA': 3, 'KE': 4, 'NG': 5,
                'EG': 6, 'ET': 7, 'MA': 8, 'GH': 9, 'RW': 10, 'TZ': 11,
                'AE': 12, 'SA': 13, 'IL': 14, 'IN': 15, 'ID': 16,
                'TH': 17, 'VN': 18, 'BR': 19, 'CL': 20, 'CO': 21}
df['country_code'] = df['country'].map(country_map).fillna(0).astype(int)

loc_map = {'motorway': 3, 'supermarket': 2, 'retail': 2, 'tesla': 2, 'council': 1, 'other': 1}
df['location_code'] = df['location_type'].map(loc_map).fillna(1).astype(int) if 'location_type' in df.columns else 1

# Engineered features
df['hour_sin']    = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos']    = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin']     = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['is_peak']     = df['hour'].apply(lambda h: 1 if h in [7,8,9,17,18,19] else 0)
df['is_morning']  = df['hour'].apply(lambda h: 1 if 6 <= h <= 12 else 0)
df['cap_log']     = np.log1p(df['capacity'])
df['is_hot']      = (df['temperature'] > 25).astype(int)
df['is_raining']  = (df['precipitation'] > 0.5).astype(int)

# ── OCCUPANCY SIMULATION ─────────────────────────────────────
np.random.seed(42)

def simulate_occupancy(row):
    p = 0.15
    hour = int(row["hour"])
    cap  = float(row["capacity"])
    loc  = int(row["location_code"])
    country = int(row["country_code"])

    # Time of day patterns
    if hour in [7, 8, 9]:        p += 0.38
    elif hour in [17, 18, 19]:   p += 0.42
    elif hour in [12, 13]:       p += 0.22
    elif hour in [0, 1, 2, 3]:   p -= 0.13

    # Weekend
    if row["is_weekend"]:
        if hour in [10,11,12,13,14,15]: p += 0.28
        else:                           p -= 0.05
    else:
        p += 0.10

    # Capacity
    if cap == 1:    p += 0.28
    elif cap == 2:  p += 0.12
    elif cap >= 6:  p -= 0.18

    # Location type
    if loc == 3:    p += 0.22
    elif loc == 2:  p += 0.12

    # Holidays
    if row.get("is_bank_holiday", 0):   p += 0.15
    if row.get("is_school_holiday", 0): p += 0.10
    if row.get("is_summer", 0):         p += 0.08

    # Weather
    temp = float(row.get("temperature", 15))
    if 15 <= temp <= 25: p += 0.08
    if float(row.get("precipitation", 0)) > 0.5: p -= 0.10

    # Country demand
    if country == 1:   p += 0.06   # US
    elif country == 2: p += 0.09   # EU
    elif country == 12: p += 0.12  # UAE

    return np.random.binomial(1, float(np.clip(p, 0.02, 0.97)))

print("Simulating occupancy patterns...")
df["is_busy"] = df.apply(simulate_occupancy, axis=1)

busy = df['is_busy'].sum()
print(f"Busy: {busy:,} ({busy/len(df)*100:.1f}%)  Free: {len(df)-busy:,}")

# ── TRAIN ────────────────────────────────────────────────────
features = [
    "hour", "day_of_week", "is_weekend", "is_peak", "is_morning",
    "capacity", "cap_log", "lat", "lon",
    "country_code", "location_code",
    "hour_sin", "hour_cos", "day_sin",
    "is_bank_holiday", "is_school_holiday", "is_summer",
    "temperature", "precipitation", "is_hot", "is_raining",
]
# Only keep features that exist
features = [f for f in features if f in df.columns]

X = df[features]
y = df["is_busy"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining on {len(X_train):,} records with {len(features)} features...")
print("Using GradientBoosting for higher accuracy...")

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    min_samples_split=20,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\n✅ Model accuracy: {accuracy * 100:.2f}%")

print("\nFeature importance:")
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])[:10]:
    bar = "█" * int(imp * 100)
    print(f"  {feat:<22} {imp:.3f}  {bar}")

with open("charger_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model_features.txt", "w") as f:
    f.write(",".join(features))

print("\nModel saved to charger_model.pkl")
print("Done!")
