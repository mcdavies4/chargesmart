from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os
import requests
from geopy.distance import geodesic
import pgeocode

app = Flask(__name__, static_folder='static')

OCM_API_KEY = 'd29bb079-0234-40d8-b0af-a2bb55a7d399'
OCM_URL = 'https://api.openchargemap.io/v3/poi'

def load_chargers():
    if os.path.exists('enriched_data.csv'):
        df = pd.read_csv('enriched_data.csv')
        chargers = df.groupby('id').agg({
            'lat': 'first', 'lon': 'first', 'capacity': 'first',
            'operator': 'first', 'location_type': 'first', 'country': 'first'
        }).reset_index()
        print(f"Loaded {len(chargers)} unique chargers")
        return chargers
    else:
        return generate_synthetic_chargers()

def generate_synthetic_chargers():
    np.random.seed(42)
    uk = pd.DataFrame({
        'id': range(1, 101),
        'lat': np.random.uniform(51.0, 53.0, 100),
        'lon': np.random.uniform(-2.0, 0.5, 100),
        'capacity': np.random.choice([1, 2, 4, 6], 100),
        'operator': np.random.choice(['BP Pulse', 'Pod Point', 'Osprey', 'Unknown'], 100),
        'location_type': np.random.choice(['motorway', 'supermarket', 'council', 'other'], 100),
        'country': 'UK'
    })
    us = pd.DataFrame({
        'id': range(101, 201),
        'lat': np.random.uniform(34.0, 47.0, 100),
        'lon': np.random.uniform(-122.5, -70.0, 100),
        'capacity': np.random.choice([2, 4, 8, 12], 100),
        'operator': np.random.choice(['Tesla', 'ChargePoint', 'EVgo', 'Blink', 'Electrify America'], 100),
        'location_type': np.random.choice(['motorway', 'supermarket', 'tesla', 'other'], 100),
        'country': 'US'
    })
    return pd.concat([uk, us], ignore_index=True)

def train_model(chargers):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    np.random.seed(42)
    n = len(chargers) * 20
    hours = np.random.randint(0, 24, n)
    days = np.random.randint(0, 7, n)
    is_weekend = (days >= 5).astype(int)
    capacities = np.random.choice(chargers['capacity'].values, n)
    lats = np.random.choice(chargers['lat'].values, n)
    lons = np.random.choice(chargers['lon'].values, n)
    score = np.zeros(n)
    score += np.where((hours >= 8) & (hours <= 9), 3, 0)
    score += np.where((hours >= 17) & (hours <= 19), 3, 0)
    score += np.where((hours >= 12) & (hours <= 13), 1, 0)
    score += is_weekend * 2
    score += np.where(capacities == 1, 2, 0)
    score += np.random.randint(0, 4, n)
    busy = (score >= 5).astype(int)
    X = np.column_stack([hours, days, is_weekend, capacities, lats, lons])
    X_train, X_test, y_train, y_test = train_test_split(X, busy, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Model trained! Accuracy: {model.score(X_test, y_test):.2%}")
    with open('charger_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

def get_live_chargers(lat, lon, radius_miles, country):
    try:
        radius_km = radius_miles * 1.60934
        country_code = 'GB' if country == 'UK' else 'US'
        params = {
            'key': OCM_API_KEY,
            'latitude': lat,
            'longitude': lon,
            'distance': radius_km,
            'distanceunit': 'KM',
            'countrycode': country_code,
            'maxresults': 30,
            'compact': True,
            'verbose': False,
        }
        response = requests.get(OCM_URL, params=params, timeout=8)
        if response.status_code != 200:
            return []
        data = response.json()
        chargers = []
        for poi in data:
            try:
                addr = poi.get('AddressInfo', {})
                c_lat = addr.get('Latitude')
                c_lon = addr.get('Longitude')
                if not c_lat or not c_lon:
                    continue
                operator = 'Unknown'
                op_info = poi.get('OperatorInfo')
                if op_info and op_info.get('Title'):
                    operator = op_info['Title'][:30]
                connections = poi.get('Connections', [])
                capacity = max(len(connections), 1)
                status_type = poi.get('StatusType')
                live_status = None
                if status_type:
                    status_id = status_type.get('ID', 0)
                    if status_id == 50:
                        live_status = 'free'
                    elif status_id​​​​​​​​​​​​​​​​
