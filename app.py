from flask import Flask, request, jsonify, render_template, redirect
import pandas as pd
import numpy as np
import pickle
import os
import requests
import json
import stripe
from geopy.distance import geodesic
import pgeocode

app = Flask(__name__, static_folder='static')

stripe.api_key = os.environ.get('STRIPE_SECRET_KEY', '')
STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY', '')
PRO_PRICE_ID = os.environ.get('STRIPE_PRO_PRICE_ID', '')
FLEET_PRICE_ID = os.environ.get('STRIPE_FLEET_PRICE_ID', '')

OCM_API_KEY = 'd29bb079-0234-40d8-b0af-a2bb55a7d399'
OCM_URL = 'https://api.openchargemap.io/v3/poi'

SUBSCRIBERS = {}

def load_subscribers():
    global SUBSCRIBERS
    if os.path.exists('subscribers.json'):
        with open('subscribers.json') as f:
            SUBSCRIBERS = json.load(f)

def save_subscribers():
    with open('subscribers.json', 'w') as f:
        json.dump(SUBSCRIBERS, f)

def get_plan(email):
    if not email:
        return 'free'
    return SUBSCRIBERS.get(email.lower(), 'free')

load_subscribers()

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
        country_code = 'GB' if country == 'UK' else ('US' if country == 'US' else None)
        params = {
            'key': OCM_API_KEY,
            'latitude': lat,
            'longitude': lon,
            'distance': radius_km,
            'distanceunit': 'KM',
            'maxresults': 30,
            'compact': True,
            'verbose': False,
        }
        if country_code:
            params['countrycode'] = country_code
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
                elif addr.get('Title'):
                    operator = addr['Title'][:30]
                elif addr.get('AddressLine1'):
                    operator = addr['AddressLine1'][:30]
                connections = poi.get('Connections', [])
                capacity = max(len(connections), 1)
                status_type = poi.get('StatusType')
                live_status = None
                if status_type:
                    status_id = status_type.get('ID', 0)
                    if status_id == 50:
                        live_status = 'free'
                    elif status_id == 210:
                        live_status = 'busy'
                dist = geodesic((lat, lon), (c_lat, c_lon)).miles
                chargers.append({
                    'id': poi.get('ID', 0),
                    'lat': float(c_lat),
                    'lon': float(c_lon),
                    'operator': operator,
                    'capacity': capacity,
                    'distance_miles': round(dist, 2),
                    'live_status': live_status,
                    'country': country,
                })
            except:
                continue
        return sorted(chargers, key=lambda x: x['distance_miles'])
    except Exception as e:
        print(f"OCM API error: {e}")
        return []

chargers = load_chargers()

if os.path.exists('charger_model.pkl'):
    with open('charger_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded from file")
else:
    print("Training model...")
    model = train_model(chargers)

nomi_uk = pgeocode.Nominatim('GB')
nomi_us = pgeocode.Nominatim('US')
nomi_de = pgeocode.Nominatim('DE')
nomi_fr = pgeocode.Nominatim('FR')
nomi_nl = pgeocode.Nominatim('NL')
nomi_no = pgeocode.Nominatim('NO')
nomi_se = pgeocode.Nominatim('SE')

def lookup_location(query):
    query = query.strip().upper()
    for nomi, country in [
        (nomi_uk, 'UK'),
        (nomi_us, 'US'),
        (nomi_de, 'EU'),
        (nomi_fr, 'EU'),
        (nomi_nl, 'EU'),
        (nomi_no, 'EU'),
        (nomi_se, 'EU'),
    ]:
        try:
            result = nomi.query_postal_code(query)
            if result is not None and not pd.isna(result.latitude):
                return float(result.latitude), float(result.longitude), country
        except:
            continue
    return None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    try:
        hour = int(request.args.get('hour', 12))
        day_of_week = int(request.args.get('day_of_week', 0))
        capacity = int(request.args.get('capacity', 2))
        lat = float(request.args.get('lat', 51.5))
        lon = float(request.args.get('lon', -0.1))
        features = np.array([[hour, day_of_week, 1 if day_of_week >= 5 else 0, capacity, lat, lon]])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        return jsonify({
            'prediction': 'busy' if prediction == 1 else 'free',
            'probability_busy': round(float(proba[1]) * 100, 1),
            'probability_free': round(float(proba[0]) * 100, 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/nearby')
def nearby():
    try:
        hour = int(request.args.get('hour', 12))
        day_of_week = int(request.args.get('day_of_week', 0))
        radius_miles = float(request.args.get('radius', 1.0))
        if request.args.get('lat') and request.args.get('lon'):
            user_lat = float(request.args.get('lat'))
            user_lon = float(request.args.get('lon'))
            country = request.args.get('country', 'UK')
        elif request.args.get('postcode'):
            query = request.args.get('postcode')
            user_lat, user_lon, country = lookup_location(query)
            if user_lat is None:
                return jsonify({'error': f'Could not find location: {query}. Please check and try again.'}), 400
        else:
            return jsonify({'error': 'Please provide a postcode, ZIP code, or coordinates'}), 400
        is_weekend = 1 if day_of_week >= 5 else 0
        results = []
        live_chargers = get_live_chargers(user_lat, user_lon, radius_miles, country)
        if live_chargers:
            for c in live_chargers[:20]:
                features = np.array([[hour, day_of_week, is_weekend, c['capacity'], c['lat'], c['lon']]])
                prediction = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                if c['live_status']:
                    final_prediction = c['live_status']
                    prob_free = 95.0 if c['live_status'] == 'free' else 5.0
                    source = 'live'
                else:
                    final_prediction = 'busy' if prediction == 1 else 'free'
                    prob_free = round(float(proba[0]) * 100, 1)
                    source = 'ai'
                results.append({
                    'id': c['id'],
                    'lat': c['lat'],
                    'lon': c['lon'],
                    'operator': c['operator'],
                    'capacity': c['capacity'],
                    'distance_miles': c['distance_miles'],
                    'prediction': final_prediction,
                    'probability_free': prob_free,
                    'probability_busy': round(100 - prob_free, 1),
                    'country': c['country'],
                    'source': source
                })
        else:
            local_chargers = chargers.copy()
            if country and 'country' in local_chargers.columns:
                local_chargers = local_chargers[local_chargers['country'] == country]
            nearby_list = []
            for _, c in local_chargers.iterrows():
                dist = geodesic((user_lat, user_lon), (c['lat'], c['lon'])).miles
                if dist <= radius_miles:
                    nearby_list.append((dist, c))
            for dist, c in sorted(nearby_list)[:20]:
                features = np.array([[hour, day_of_week, is_weekend, c['capacity'], c['lat'], c['lon']]])
                prediction = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                results.append({
                    'id': int(c['id']),
                    'lat': round(float(c['lat']), 6),
                    'lon': round(float(c['lon']), 6),
                    'operator': str(c['operator']),
                    'capacity': int(c['capacity']),
                    'distance_miles': round(dist, 2),
                    'prediction': 'busy' if prediction == 1 else 'free',
                    'probability_free': round(float(proba[0]) * 100, 1),
                    'probability_busy': round(float(proba[1]) * 100, 1),
                    'country': str(c.get('country', 'UK')),
                    'source': 'ai'
                })
        return jsonify({'total': len(results), 'chargers': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session():
    try:
        data = request.get_json()
        plan = data.get('plan', 'pro')
        email = data.get('email', '')
        price_id = FLEET_PRICE_ID if plan == 'fleet' else PRO_PRICE_ID
        base_url = request.host_url.rstrip('/')
        # 14-day free trial for all plans
        subscription_data = {'trial_period_days': 14}

        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            mode='subscription',
            customer_email=email if email else None,
            line_items=[{'price': price_id, 'quantity': 1}],
            success_url=f'{base_url}/payment-success?session_id={{CHECKOUT_SESSION_ID}}&plan={plan}&email={email}',
            cancel_url=f'{base_url}/?cancelled=true',
            metadata={'plan': plan, 'email': email},
            subscription_data=subscription_data if subscription_data else None
        )
        return jsonify({'url': session.url})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/payment-success')
def payment_success():
    session_id = request.args.get('session_id')
    plan = request.args.get('plan', 'pro')
    email = request.args.get('email', '')
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        if session.payment_status == 'paid' or session.status == 'complete':
            customer_email = session.customer_email or email
            if customer_email:
                SUBSCRIBERS[customer_email.lower()] = plan
                save_subscribers()
            return redirect(f'/?success=true&plan={plan}&email={customer_email}')
    except Exception as e:
        print(f"Payment verification error: {e}")
    return redirect('/?success=true&plan=' + plan)

@app.route('/check-subscription', methods=['POST'])
def check_subscription():
    data = request.get_json()
    email = data.get('email', '').lower()
    plan = get_plan(email)
    return jsonify({'plan': plan, 'email': email})

@app.route('/stripe-key')
def stripe_key():
    key = os.environ.get('STRIPE_SECRET_KEY', 'NOT FOUND')
    pub = os.environ.get('STRIPE_PUBLISHABLE_KEY', 'NOT FOUND')
    pro = os.environ.get('STRIPE_PRO_PRICE_ID', 'NOT FOUND')
    fleet = os.environ.get('STRIPE_FLEET_PRICE_ID', 'NOT FOUND')
    return jsonify({
        'secret_key_found': key != 'NOT FOUND',
        'secret_key_prefix': key[:12] if key != 'NOT FOUND' else 'NOT FOUND',
        'publishable_key_found': pub != 'NOT FOUND',
        'pro_price_found': pro != 'NOT FOUND',
        'fleet_price_found': fleet != 'NOT FOUND',
    })

@app.route('/static/manifest.json')
def manifest():
    return app.send_static_file('manifest.json')

@app.route('/static/service-worker.js')
def service_worker():
    response = app.send_static_file('service-worker.js')
    response.headers['Service-Worker-Allowed'] = '/'
    return response

@app.route('/sitemap.xml')
def sitemap():
    return app.send_static_file('sitemap.xml')

@app.route('/robots.txt')
def robots():
    return app.send_static_file('robots.txt')


# ── REVIEWS ─────────────────────────────────────────────────
import datetime

REVIEWS_FILE = 'reviews.json'

def load_reviews():
    if os.path.exists(REVIEWS_FILE):
        with open(REVIEWS_FILE) as f:
            return json.load(f)
    return {}

def save_reviews(reviews):
    with open(REVIEWS_FILE, 'w') as f:
        json.dump(reviews, f)

@app.route('/submit-review', methods=['POST'])
def submit_review():
    try:
        data = request.get_json()
        charger_id = str(data.get('charger_id', ''))
        rating = int(data.get('rating', 0))
        comment = str(data.get('comment', ''))[:200]
        operator = str(data.get('operator', 'Unknown'))[:40]
        if not charger_id or rating < 1 or rating > 5:
            return jsonify({'error': 'Invalid review'}), 400
        reviews = load_reviews()
        if charger_id not in reviews:
            reviews[charger_id] = {'operator': operator, 'ratings': [], 'comments': []}
        reviews[charger_id]['ratings'].append(rating)
        reviews[charger_id]['comments'].append({
            'rating': rating,
            'comment': comment,
            'date': datetime.datetime.now().strftime('%Y-%m-%d')
        })
        save_reviews(reviews)
        avg = sum(reviews[charger_id]['ratings']) / len(reviews[charger_id]['ratings'])
        return jsonify({'success': True, 'avg_rating': round(avg, 1), 'total': len(reviews[charger_id]['ratings'])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get-reviews/<charger_id>')
def get_reviews(charger_id):
    reviews = load_reviews()
    data = reviews.get(str(charger_id), {})
    if not data:
        return jsonify({'avg_rating': None, 'total': 0, 'comments': []})
    ratings = data.get('ratings', [])
    avg = sum(ratings) / len(ratings) if ratings else 0
    return jsonify({
        'avg_rating': round(avg, 1),
        'total': len(ratings),
        'comments': data.get('comments', [])[-5:]
    })

# ── FAULT REPORTER ───────────────────────────────────────────
FAULTS_FILE = 'faults.json'

def load_faults():
    if os.path.exists(FAULTS_FILE):
        with open(FAULTS_FILE) as f:
            return json.load(f)
    return {}

@app.route('/report-fault', methods=['POST'])
def report_fault():
    try:
        data = request.get_json()
        charger_id = str(data.get('charger_id', ''))
        fault_type = str(data.get('fault_type', 'other'))
        operator = str(data.get('operator', 'Unknown'))[:40]
        if not charger_id:
            return jsonify({'error': 'Invalid'}), 400
        faults = load_faults()
        if charger_id not in faults:
            faults[charger_id] = {'operator': operator, 'reports': []}
        faults[charger_id]['reports'].append({
            'fault_type': fault_type,
            'date': datetime.datetime.now().strftime('%Y-%m-%d')
        })
        with open(FAULTS_FILE, 'w') as f:
            json.dump(faults, f)
        return jsonify({'success': True, 'total_reports': len(faults[charger_id]['reports'])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── JOURNEY COST CALCULATOR ──────────────────────────────────
@app.route('/journey-cost', methods=['POST'])
def journey_cost():
    try:
        data = request.get_json()
        miles = float(data.get('miles', 0))
        car_model = data.get('car_model', 'average')
        currency = data.get('currency', 'gbp')
        EV_EFFICIENCY = {'nissan_leaf': 3.5, 'tesla_model3': 4.0, 'tesla_modelx': 2.8, 'vw_id3': 3.8, 'average': 3.5}
        ELEC_COST = {'gbp': 0.28, 'usd': 0.16, 'eur': 0.22}
        PETROL_COST = {'gbp': 0.155, 'usd': 0.12, 'eur': 0.14}
        PETROL_MPG = 35
        sym = {'gbp': '£', 'usd': '$', 'eur': '€'}[currency]
        mpkwh = EV_EFFICIENCY.get(car_model, 3.5)
        kwh_needed = miles / mpkwh
        ev_cost = kwh_needed * ELEC_COST[currency]
        petrol_cost = (miles / PETROL_MPG) * PETROL_COST[currency] * 4.546
        saving = petrol_cost - ev_cost
        co2_saved = (0.404 - 0.069) * miles
        return jsonify({
            'ev_cost': round(ev_cost, 2),
            'petrol_cost': round(petrol_cost, 2),
            'saving': round(saving, 2),
            'kwh_needed': round(kwh_needed, 1),
            'co2_saved': round(co2_saved, 1),
            'symbol': sym
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ── CHARGER DESERT MAP ──────────────────────────────────────
@app.route('/charger-deserts')
def charger_deserts():
    return render_template('index.html')

@app.route('/api/charger-deserts')
def api_charger_deserts():
    try:
        region = request.args.get('region', 'uk')

        # Grid definitions per region
        regions = {
            'uk': {
                'lat_min': 49.9, 'lat_max': 58.7,
                'lon_min': -6.0, 'lon_max': 1.8,
                'grid_step': 0.3,
                'name': 'United Kingdom',
                'country': 'UK'
            },
            'us_east': {
                'lat_min': 24.5, 'lat_max': 47.5,
                'lon_min': -85.0, 'lon_max': -66.0,
                'grid_step': 0.5,
                'name': 'US East Coast',
                'country': 'US'
            },
            'us_west': {
                'lat_min': 32.0, 'lat_max': 49.0,
                'lon_min': -124.5, 'lon_max': -104.0,
                'grid_step': 0.5,
                'name': 'US West Coast',
                'country': 'US'
            },
            'germany': {
                'lat_min': 47.2, 'lat_max': 55.1,
                'lon_min': 5.8, 'lon_max': 15.1,
                'grid_step': 0.3,
                'name': 'Germany',
                'country': 'EU'
            },
            'france': {
                'lat_min': 41.3, 'lat_max': 51.1,
                'lon_min': -5.2, 'lon_max': 9.6,
                'grid_step': 0.3,
                'name': 'France',
                'country': 'EU'
            },
            'netherlands': {
                'lat_min': 50.7, 'lat_max': 53.6,
                'lon_min': 3.3, 'lon_max': 7.3,
                'grid_step': 0.2,
                'name': 'Netherlands',
                'country': 'EU'
            }
        }

        if region not in regions:
            region = 'uk'

        reg = regions[region]

        # Load charger data
        if os.path.exists('enriched_data.csv'):
            import pandas as pd
            df = pd.read_csv('enriched_data.csv')
            if reg['country'] != 'ALL':
                df = df[df['country'] == reg['country']]
            charger_coords = list(zip(df['lat'].values, df['lon'].values))
        else:
            charger_coords = []

        # Build grid and find deserts
        deserts = []
        covered = []
        DESERT_RADIUS = 0.08  # ~5 miles in degrees

        lat = reg['lat_min']
        while lat <= reg['lat_max']:
            lon = reg['lon_min']
            while lon <= reg['lon_max']:
                # Check if any charger is within radius
                has_charger = False
                for clat, clon in charger_coords:
                    if abs(clat - lat) < DESERT_RADIUS and abs(clon - lon) < DESERT_RADIUS:
                        has_charger = True
                        break

                if not has_charger:
                    deserts.append({'lat': round(lat, 3), 'lon': round(lon, 3)})
                else:
                    covered.append({'lat': round(lat, 3), 'lon': round(lon, 3)})

                lon += reg['grid_step']
            lat += reg['grid_step']

        total = len(deserts) + len(covered)
        desert_pct = round(len(deserts) / total * 100, 1) if total > 0 else 0

        # Build sub-region breakdown (quadrants)
        lat_mid = (reg['lat_min'] + reg['lat_max']) / 2
        lon_mid = (reg['lon_min'] + reg['lon_max']) / 2

        quadrants = {
            'North West': {'d': 0, 'c': 0},
            'North East': {'d': 0, 'c': 0},
            'South West': {'d': 0, 'c': 0},
            'South East': {'d': 0, 'c': 0},
        }
        for pt in deserts:
            ns = 'North' if pt['lat'] >= lat_mid else 'South'
            ew = 'West' if pt['lon'] < lon_mid else 'East'
            quadrants[f'{ns} {ew}']['d'] += 1
        for pt in covered:
            ns = 'North' if pt['lat'] >= lat_mid else 'South'
            ew = 'West' if pt['lon'] < lon_mid else 'East'
            quadrants[f'{ns} {ew}']['c'] += 1

        breakdown = []
        for name, vals in quadrants.items():
            total_q = vals['d'] + vals['c']
            pct_q = round(vals['d'] / total_q * 100, 1) if total_q > 0 else 0
            breakdown.append({
                'name': name,
                'desert_zones': vals['d'],
                'covered_zones': vals['c'],
                'desert_pct': pct_q
            })
        breakdown.sort(key=lambda x: -x['desert_pct'])

        return jsonify({
            'region': reg['name'],
            'deserts': deserts[:2000],
            'total_desert_zones': len(deserts),
            'total_covered_zones': len(covered),
            'desert_percentage': desert_pct,
            'charger_count': len(charger_coords),
            'breakdown': breakdown
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ═══════════════════════════════════════════════════════════════
# CHARGESMART PUBLIC API v1
# ═══════════════════════════════════════════════════════════════
from api_system import create_api_key, validate_key, record_usage, get_key_stats, TIER_LIMITS, TIER_PRICES
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get('Authorization', '')
        api_key = auth.replace('Bearer ', '').strip()
        if not api_key:
            api_key = request.args.get('key', '')
        valid, tier, remaining, error = validate_key(api_key)
        if not valid:
            return jsonify({
                'error': error,
                'docs': 'https://chargesmart.online/developers'
            }), 401
        record_usage(api_key)
        request.api_tier = tier
        request.api_remaining = remaining
        return f(*args, **kwargs)
    return decorated

def api_response(data, api_key=''):
    """Wrap response with metadata"""
    auth = request.headers.get('Authorization', '')
    key = auth.replace('Bearer ', '').strip() or request.args.get('key', '')
    resp = {
        'success': True,
        'data': data,
        'meta': {
            'version': 'v1',
            'calls_remaining_today': getattr(request, 'api_remaining', None),
            'docs': 'https://chargesmart.online/developers'
        }
    }
    return jsonify(resp)

# ── GET API KEY ──────────────────────────────────────────────
@app.route('/api/v1/keys/register', methods=['POST'])
def register_api_key():
    try:
        data = request.get_json() or {}
        email = data.get('email', '').strip().lower()
        if not email or '@' not in email:
            return jsonify({'error': 'Valid email required'}), 400
        result = create_api_key(email, tier='free')
        if result['existing']:
            msg = 'Your existing API key has been returned'
        else:
            msg = 'API key created successfully. Welcome to ChargeSmart API!'
        return jsonify({
            'success': True,
            'message': msg,
            'api_key': result['key'],
            'tier': result['tier'],
            'daily_limit': TIER_LIMITS['free'],
            'docs': 'https://chargesmart.online/developers'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── USAGE STATS ──────────────────────────────────────────────
@app.route('/api/v1/keys/stats')
def api_key_stats():
    auth = request.headers.get('Authorization', '')
    api_key = auth.replace('Bearer ', '').strip() or request.args.get('key', '')
    if not api_key:
        return jsonify({'error': 'API key required'}), 401
    stats = get_key_stats(api_key)
    if not stats:
        return jsonify({'error': 'Invalid API key'}), 401
    return jsonify({'success': True, 'data': stats})

# ── 1. PREDICT CHARGER AVAILABILITY ─────────────────────────
@app.route('/api/v1/predict')
@require_api_key
def api_predict():
    try:
        lat  = float(request.args.get('lat', 51.5))
        lon  = float(request.args.get('lon', -0.1))
        hour = int(request.args.get('hour', datetime.now().hour))
        day  = int(request.args.get('day', datetime.now().weekday()))
        cap  = int(request.args.get('capacity', 2))
        country = request.args.get('country', 'UK').upper()

        country_map = {'UK': 0, 'US': 1, 'EU': 2}
        loc_map_r = {'motorway': 3, 'supermarket': 2, 'council': 1, 'tesla': 2, 'other': 1}
        loc_type = request.args.get('location_type', 'other')

        features_vals = [[
            hour, day,
            1 if day >= 5 else 0,
            min(cap, 20),
            lat, lon,
            country_map.get(country, 0),
            loc_map_r.get(loc_type, 1)
        ]]

        import pickle, os
        if os.path.exists('charger_model.pkl'):
            model = pickle.load(open('charger_model.pkl', 'rb'))
            proba = model.predict_proba(features_vals)[0]
            prob_free = round(float(proba[0]) * 100, 1)
            prediction = 'free' if prob_free >= 50 else 'busy'
        else:
            prob_free = 50.0
            prediction = 'unknown'

        # Find best time today
        best_hour = hour
        best_prob = prob_free
        for h in range(24):
            fv = [[h, day, 1 if day >= 5 else 0, min(cap,20), lat, lon, country_map.get(country,0), loc_map_r.get(loc_type,1)]]
            if os.path.exists('charger_model.pkl'):
                p = model.predict_proba(fv)[0]
                pf = round(float(p[0]) * 100, 1)
                if pf > best_prob:
                    best_prob = pf
                    best_hour = h

        return api_response({
            'prediction':      prediction,
            'probability_free': prob_free,
            'probability_busy': round(100 - prob_free, 1),
            'best_time_today': f"{best_hour:02d}:00",
            'best_time_probability': best_prob,
            'inputs': {
                'lat': lat, 'lon': lon,
                'hour': hour, 'day_of_week': day,
                'capacity': cap, 'country': country,
                'location_type': loc_type
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 2. CHARGER DESERTS ───────────────────────────────────────
@app.route('/api/v1/deserts')
@require_api_key
def api_deserts():
    try:
        region = request.args.get('region', 'uk')
        # Reuse existing desert logic
        with app.test_request_context(f'/api/charger-deserts?region={region}'):
            pass
        resp = api_charger_deserts.__wrapped__() if hasattr(api_charger_deserts, '__wrapped__') else api_charger_deserts()
        data = resp.get_json()
        return api_response(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 3. JOURNEY COST ──────────────────────────────────────────
@app.route('/api/v1/journey-cost')
@require_api_key
def api_journey_cost():
    try:
        miles    = float(request.args.get('miles', 0))
        car      = request.args.get('car', 'average')
        currency = request.args.get('currency', 'gbp').lower()
        if miles <= 0:
            return jsonify({'error': 'miles must be greater than 0'}), 400

        EV_EFF  = {'average':3.5,'nissan_leaf':3.5,'tesla_model3':4.0,'tesla_modelx':2.8,'vw_id3':3.8}
        ELEC    = {'gbp':0.28,'usd':0.16,'eur':0.22}
        PETROL  = {'gbp':0.155,'usd':0.12,'eur':0.14}
        SYMS    = {'gbp':'£','usd':'$','eur':'€'}

        kwh       = miles / (EV_EFF.get(car, 3.5))
        ev_cost   = kwh * ELEC.get(currency, 0.28)
        pet_cost  = (miles / 35) * PETROL.get(currency, 0.155) * 4.546
        saving    = pet_cost - ev_cost
        co2_saved = 0.335 * miles

        return api_response({
            'miles':          miles,
            'car_model':      car,
            'currency':       currency,
            'symbol':         SYMS.get(currency, '£'),
            'ev_cost':        round(ev_cost, 2),
            'petrol_cost':    round(pet_cost, 2),
            'saving':         round(saving, 2),
            'saving_pct':     round((saving / pet_cost) * 100, 1),
            'kwh_needed':     round(kwh, 1),
            'co2_saved_kg':   round(co2_saved, 1),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 4. CARBON SAVINGS ────────────────────────────────────────
@app.route('/api/v1/carbon')
@require_api_key
def api_carbon():
    try:
        miles          = float(request.args.get('miles', 0))
        vehicle        = request.args.get('vehicle', 'medium')
        trips_per_week = int(request.args.get('trips_per_week', 1))
        currency       = request.args.get('currency', 'gbp').lower()
        if miles <= 0:
            return jsonify({'error': 'miles must be greater than 0'}), 400

        EV_CO2   = {'small':0.053,'medium':0.069,'large':0.098,'van':0.135}
        PETROL_CO2 = 0.404
        FUEL_COST  = {'gbp':0.155,'usd':0.193,'eur':0.175}
        EV_COST    = {'gbp':0.052,'usd':0.065,'eur':0.058}
        SYMS       = {'gbp':'£','usd':'$','eur':'€'}

        co2_per_trip   = (PETROL_CO2 - EV_CO2.get(vehicle, 0.069)) * miles
        money_per_trip = (FUEL_COST.get(currency,0.155) - EV_COST.get(currency,0.052)) * miles

        return api_response({
            'miles_per_trip':        miles,
            'vehicle_type':          vehicle,
            'trips_per_week':        trips_per_week,
            'currency':              currency,
            'symbol':                SYMS.get(currency,'£'),
            'co2_saved_per_trip_kg': round(co2_per_trip, 2),
            'co2_saved_weekly_kg':   round(co2_per_trip * trips_per_week, 2),
            'co2_saved_annual_kg':   round(co2_per_trip * trips_per_week * 52, 1),
            'trees_equivalent':      round(co2_per_trip * trips_per_week * 52 / 21.77, 1),
            'money_saved_per_trip':  round(money_per_trip, 2),
            'money_saved_weekly':    round(money_per_trip * trips_per_week, 2),
            'money_saved_annual':    round(money_per_trip * trips_per_week * 52, 2),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 5. CHARGER REVIEWS ───────────────────────────────────────
@app.route('/api/v1/reviews')
@require_api_key
def api_reviews():
    try:
        lat    = float(request.args.get('lat', 51.5))
        lon    = float(request.args.get('lon', -0.1))
        radius = float(request.args.get('radius', 5))

        reviews = load_reviews()
        faults  = load_faults()
        results = []
        for charger_id, data in reviews.items():
            ratings = data.get('ratings', [])
            if not ratings:
                continue
            avg = round(sum(ratings) / len(ratings), 1)
            fault_count = len(faults.get(charger_id, {}).get('reports', []))
            results.append({
                'charger_id':    charger_id,
                'operator':      data.get('operator', 'Unknown'),
                'avg_rating':    avg,
                'total_reviews': len(ratings),
                'recent_faults': fault_count,
                'comments':      data.get('comments', [])[-3:],
            })
        results.sort(key=lambda x: -x['total_reviews'])
        return api_response({
            'total_chargers_with_reviews': len(results),
            'chargers': results[:50]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── DEVELOPER DASHBOARD PAGE ─────────────────────────────────
@app.route('/developers')
def developers_page():
    return render_template('developers.html')


# ── NETWORK UPTIME LEADERBOARD ───────────────────────────────
@app.route('/leaderboard')
def leaderboard_page():
    return render_template('leaderboard.html')

@app.route('/api/leaderboard')
def api_leaderboard():
    try:
        faults  = load_faults()
        reviews = load_reviews()

        # Known networks with base reliability scores
        networks = {
            'Tesla Supercharger': {'base': 96, 'country': 'UK/US/EU', 'type': 'DC Fast'},
            'Pod Point':          {'base': 88, 'country': 'UK',       'type': 'AC/DC'},
            'BP Pulse':           {'base': 71, 'country': 'UK',       'type': 'AC/DC'},
            'Osprey':             {'base': 85, 'country': 'UK',       'type': 'DC Fast'},
            'InstaVolt':          {'base': 89, 'country': 'UK',       'type': 'DC Fast'},
            'Gridserve':          {'base': 92, 'country': 'UK',       'type': 'DC Fast'},
            'ChargePoint':        {'base': 84, 'country': 'US',       'type': 'AC/DC'},
            'EVgo':               {'base': 79, 'country': 'US',       'type': 'DC Fast'},
            'Electrify America':  {'base': 76, 'country': 'US',       'type': 'DC Fast'},
            'Blink':              {'base': 68, 'country': 'US',       'type': 'AC/DC'},
            'Ionity':             {'base': 91, 'country': 'EU',       'type': 'DC Fast'},
            'Allego':             {'base': 83, 'country': 'EU',       'type': 'AC/DC'},
            'Fastned':            {'base': 94, 'country': 'EU',       'type': 'DC Fast'},
            'Vattenfall':         {'base': 87, 'country': 'EU',       'type': 'AC/DC'},
        }

        results = []
        for name, info in networks.items():
            # Count community fault reports for this network
            fault_count = 0
            review_count = 0
            total_rating = 0
            for cid, fdata in faults.items():
                op = fdata.get('operator', '').lower()
                if any(n.lower() in op for n in name.lower().split()):
                    fault_count += len(fdata.get('reports', []))
            for cid, rdata in reviews.items():
                op = rdata.get('operator', '').lower()
                if any(n.lower() in op for n in name.lower().split()):
                    ratings = rdata.get('ratings', [])
                    review_count += len(ratings)
                    total_rating += sum(ratings)

            # Adjust uptime based on faults
            uptime = info['base']
            uptime -= min(fault_count * 2, 15)
            uptime = max(min(uptime, 99), 40)

            avg_rating = round(total_rating / review_count, 1) if review_count > 0 else None

            results.append({
                'name':         name,
                'uptime':       uptime,
                'country':      info['country'],
                'type':         info['type'],
                'fault_reports': fault_count,
                'review_count': review_count,
                'avg_rating':   avg_rating,
                'trend':        'up' if uptime >= info['base'] else 'down',
            })

        results.sort(key=lambda x: -x['uptime'])
        for i, r in enumerate(results):
            r['rank'] = i + 1

        return jsonify({
            'success': True,
            'networks': results,
            'total_networks': len(results),
            'last_updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── FLEET PDF REPORT ─────────────────────────────────────────
@app.route('/fleet-report-pdf', methods=['POST'])
def fleet_report_pdf():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        import io

        data = request.get_json() or {}
        fleet_name    = data.get('fleet_name', 'My Fleet')
        vehicles      = data.get('vehicles', [])
        month         = data.get('month', datetime.datetime.now().strftime('%B %Y'))
        total_miles   = sum(v.get('miles', 0) for v in vehicles)
        total_sessions= sum(v.get('sessions', 0) for v in vehicles)

        # Calculations
        CO2_PER_MILE  = 0.335
        COST_PER_MILE_EV     = 0.052
        COST_PER_MILE_PETROL = 0.155
        total_co2     = round(total_miles * CO2_PER_MILE, 1)
        total_saving  = round(total_miles * (COST_PER_MILE_PETROL - COST_PER_MILE_EV), 2)
        trees_eq      = round(total_co2 / 21.77, 1)

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=20*mm, leftMargin=20*mm,
                                topMargin=20*mm, bottomMargin=20*mm)

        styles = getSampleStyleSheet()
        GREEN  = colors.HexColor('#00C851')
        DARK   = colors.HexColor('#0a0a0f')
        GREY   = colors.HexColor('#6b6b80')

        title_style = ParagraphStyle('Title', fontSize=24, fontName='Helvetica-Bold',
                                     textColor=DARK, spaceAfter=4)
        sub_style   = ParagraphStyle('Sub',   fontSize=12, fontName='Helvetica',
                                     textColor=GREY, spaceAfter=20)
        label_style = ParagraphStyle('Label', fontSize=9,  fontName='Helvetica',
                                     textColor=GREY)
        val_style   = ParagraphStyle('Val',   fontSize=22, fontName='Helvetica-Bold',
                                     textColor=GREEN)
        section_style = ParagraphStyle('Section', fontSize=13, fontName='Helvetica-Bold',
                                        textColor=DARK, spaceBefore=16, spaceAfter=8)

        story = []

        # Header
        story.append(Paragraph(f'⚡ ChargeSmart Fleet Report', title_style))
        story.append(Paragraph(f'{fleet_name} · {month}', sub_style))
        story.append(Spacer(1, 8))

        # Summary stats table
        stats_data = [
            [Paragraph('TOTAL MILES', label_style), Paragraph('CHARGING SESSIONS', label_style),
             Paragraph('CO2 SAVED', label_style), Paragraph('MONEY SAVED', label_style)],
            [Paragraph(f'{total_miles:,}', val_style), Paragraph(f'{total_sessions}', val_style),
             Paragraph(f'{total_co2}kg', val_style), Paragraph(f'£{total_saving:,.2f}', val_style)],
        ]
        stats_table = Table(stats_data, colWidths=[42*mm, 50*mm, 42*mm, 42*mm])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fff8')),
            ('BOX',        (0,0), (-1,-1), 1, colors.HexColor('#e0ffe0')),
            ('INNERGRID',  (0,0), (-1,-1), 0.5, colors.HexColor('#e0ffe0')),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#f0fff0'), colors.white]),
            ('TOPPADDING',    (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 8))

        # Trees equivalent callout
        story.append(Paragraph(f'🌳  Equivalent to planting {trees_eq} trees this month', sub_style))

        # Vehicle breakdown
        if vehicles:
            story.append(Paragraph('Vehicle Breakdown', section_style))
            veh_data = [['Vehicle', 'Driver', 'Miles', 'Sessions', 'CO2 Saved', 'Cost Saved']]
            for v in vehicles:
                m = v.get('miles', 0)
                s = v.get('sessions', 0)
                co2 = round(m * CO2_PER_MILE, 1)
                sav = round(m * (COST_PER_MILE_PETROL - COST_PER_MILE_EV), 2)
                veh_data.append([
                    v.get('plate', 'Unknown'),
                    v.get('driver', 'Unknown'),
                    f'{m:,}',
                    str(s),
                    f'{co2}kg',
                    f'£{sav:.2f}',
                ])
            veh_table = Table(veh_data, colWidths=[30*mm, 35*mm, 25*mm, 25*mm, 28*mm, 28*mm])
            veh_table.setStyle(TableStyle([
                ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor('#00C851')),
                ('TEXTCOLOR',     (0,0), (-1,0),  colors.white),
                ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
                ('FONTSIZE',      (0,0), (-1,-1), 9),
                ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
                ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, colors.HexColor('#f9f9f9')]),
                ('BOX',           (0,0), (-1,-1), 0.5, colors.HexColor('#dddddd')),
                ('INNERGRID',     (0,0), (-1,-1), 0.25, colors.HexColor('#eeeeee')),
                ('TOPPADDING',    (0,0), (-1,-1), 6),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(veh_table)

        # Footer
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            f'Generated by ChargeSmart · chargesmart.online · {datetime.datetime.now().strftime("%d %b %Y")}',
            ParagraphStyle('Footer', fontSize=8, textColor=GREY, alignment=TA_CENTER)
        ))

        doc.build(story)
        buf.seek(0)

        from flask import send_file
        return send_file(
            buf,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'ChargeSmart_Fleet_Report_{month.replace(" ","_")}.pdf'
        )
    except ImportError:
        return jsonify({'error': 'PDF generation requires reportlab. Run: pip install reportlab'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ═══════════════════════════════════════════════════════════════
# SIGNUP TRACKING
# ═══════════════════════════════════════════════════════════════
SIGNUPS_FILE = 'signups.json'

def load_signups():
    if os.path.exists(SIGNUPS_FILE):
        with open(SIGNUPS_FILE) as f:
            return json.load(f)
    return []

def save_signups(signups):
    with open(SIGNUPS_FILE, 'w') as f:
        json.dump(signups, f, indent=2)

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json() or {}
        name   = data.get('name', '').strip()
        email  = data.get('email', '').strip().lower()
        source = data.get('source', '').strip()

        if not email or '@' not in email:
            return jsonify({'error': 'Valid email required'}), 400
        if not name:
            return jsonify({'error': 'Name required'}), 400

        signups = load_signups()

        # Check duplicate
        for s in signups:
            if s.get('email') == email:
                return jsonify({'success': True, 'message': 'You are already signed up!', 'duplicate': True})

        signups.append({
            'id':        len(signups) + 1,
            'name':      name,
            'email':     email,
            'source':    source or 'Not specified',
            'date':      datetime.datetime.now().strftime('%Y-%m-%d'),
            'time':      datetime.datetime.now().strftime('%H:%M'),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
        save_signups(signups)

        return jsonify({
            'success': True,
            'message': f'Welcome aboard {name}! You are user #{len(signups)}.',
            'user_number': len(signups)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/admin/signups')
def admin_signups():
    # Simple password check via query param
    pw = request.args.get('pw', '')
    if pw != os.environ.get('ADMIN_PW', 'chargesmart2026'):
        return jsonify({'error': 'Unauthorised'}), 401
    signups = load_signups()
    # Analytics
    from collections import Counter
    sources = Counter(s.get('source','Unknown') for s in signups)
    dates   = Counter(s.get('date','') for s in signups)
    return jsonify({
        'total_signups':  len(signups),
        'signups':        signups,
        'by_source':      dict(sources.most_common()),
        'by_date':        dict(sorted(dates.items())),
        'today':          sum(1 for s in signups if s.get('date') == datetime.datetime.now().strftime('%Y-%m-%d')),
        'this_week':      sum(1 for s in signups if s.get('date', '') >= (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')),
    })

@app.route('/admin')
def admin_dashboard():
    pw = request.args.get('pw', '')
    if pw != os.environ.get('ADMIN_PW', 'chargesmart2026'):
        return '''<!DOCTYPE html><html><body style="background:#0a0a0f;color:#fff;font-family:monospace;display:flex;align-items:center;justify-content:center;height:100vh;flex-direction:column;gap:16px;">
        <div style="font-size:32px;">⚡</div>
        <form method="GET">
            <input name="pw" type="password" placeholder="Admin password" style="padding:10px;background:#1a1a2a;border:1px solid #333;color:#fff;border-radius:6px;font-family:monospace;">
            <button type="submit" style="padding:10px 20px;background:#00ff87;border:none;border-radius:6px;color:#000;font-family:monospace;font-weight:700;cursor:pointer;margin-left:8px;">ENTER</button>
        </form></body></html>''', 401
    return render_template('admin.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

# ═══════════════════════════════════════════════════════════════
# CHARGESMART API v1 — EXTENDED ENDPOINTS
# ═══════════════════════════════════════════════════════════════

def load_charger_data(country=None):
    """Load enriched_data.csv once, optionally filtered by country."""
    import pandas as pd
    if not os.path.exists('enriched_data.csv'):
        return None
    cols = ['lat','lon','hour','day_of_week','capacity','location_type','operator','country']
    df = pd.read_csv('enriched_data.csv', usecols=cols)
    if country and country != 'ALL':
        df = df[df['country'] == country.upper()]
    return df

# ── 1. PEAK HOURS ────────────────────────────────────────────
@app.route('/api/v1/peak-hours')
@require_api_key
def api_peak_hours():
    try:
        lat     = float(request.args.get('lat', 51.5))
        lon     = float(request.args.get('lon', -0.1))
        country = request.args.get('country', 'UK').upper()
        radius  = float(request.args.get('radius', 0.5))

        import pandas as pd
        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        # Filter to nearby chargers
        nearby = df[
            (abs(df['lat'] - lat) < radius) &
            (abs(df['lon'] - lon) < radius)
        ]
        if len(nearby) == 0:
            nearby = df  # fallback to country-wide patterns

        hour_counts = nearby.groupby('hour').size()
        total = hour_counts.sum()
        hours_data = []
        for h in range(24):
            count = int(hour_counts.get(h, 0))
            pct   = round(count / total * 100, 1) if total > 0 else 0
            hours_data.append({
                'hour':       h,
                'label':      f'{h:02d}:00',
                'demand_pct': pct,
                'demand':     'high' if pct > 6 else 'medium' if pct > 3 else 'low'
            })

        sorted_hours = sorted(hours_data, key=lambda x: -x['demand_pct'])
        return api_response({
            'lat': lat, 'lon': lon, 'country': country,
            'peak_hours':    sorted_hours[:3],
            'quietest_hours':list(reversed(sorted_hours))[:3],
            'best_time_to_charge': sorted_hours[-1]['label'],
            'worst_time_to_charge': sorted_hours[0]['label'],
            'hourly_breakdown': hours_data,
            'data_points': len(nearby)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 2. OPERATOR INTELLIGENCE ─────────────────────────────────
@app.route('/api/v1/operators')
@require_api_key
def api_operators():
    try:
        country = request.args.get('country', 'UK').upper()
        import pandas as pd
        df = load_charger_data(country if country != 'ALL' else None)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        ops = []
        for op_name, grp in df.groupby('operator'):
            if op_name.lower() in ['unknown','']:
                continue
            loc_dist = grp['location_type'].value_counts().to_dict()
            ops.append({
                'operator':        op_name,
                'total_chargers':  len(grp['lat'].drop_duplicates()),
                'avg_capacity':    round(float(grp['capacity'].mean()), 1),
                'max_capacity':    int(grp['capacity'].max()),
                'location_types':  loc_dist,
                'primary_location':grp['location_type'].mode()[0] if len(grp) > 0 else 'unknown',
                'countries':       list(grp['country'].unique()) if 'country' in grp.columns else [country],
                'lat_range':       [round(float(grp['lat'].min()),2), round(float(grp['lat'].max()),2)],
                'lon_range':       [round(float(grp['lon'].min()),2), round(float(grp['lon'].max()),2)],
            })

        ops.sort(key=lambda x: -x['total_chargers'])
        return api_response({
            'country':        country,
            'total_operators': len(ops),
            'operators':      ops[:50],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 3. COVERAGE SCORE ────────────────────────────────────────
@app.route('/api/v1/coverage')
@require_api_key
def api_coverage():
    try:
        lat     = float(request.args.get('lat', 51.5))
        lon     = float(request.args.get('lon', -0.1))
        radius  = float(request.args.get('radius', 10))
        country = request.args.get('country', 'UK').upper()

        import pandas as pd
        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        # Convert radius miles to degrees (~0.014 deg per mile)
        deg = radius * 0.014
        nearby = df[
            (abs(df['lat'] - lat) < deg) &
            (abs(df['lon'] - lon) < deg)
        ]

        unique_chargers = len(nearby[['lat','lon']].drop_duplicates())
        avg_capacity    = float(nearby['capacity'].mean()) if len(nearby) > 0 else 0
        operators       = nearby['operator'].nunique() if len(nearby) > 0 else 0
        has_fast        = 'motorway' in nearby['location_type'].values or avg_capacity >= 4

        # Score 0-100
        score = 0
        score += min(unique_chargers * 3, 40)  # up to 40 pts for charger count
        score += min(avg_capacity * 5, 20)      # up to 20 pts for capacity
        score += min(operators * 5, 20)         # up to 20 pts for network diversity
        score += 20 if has_fast else 0          # 20 pts for fast charging
        score = min(int(score), 100)

        grade = 'A' if score >= 80 else 'B' if score >= 60 else 'C' if score >= 40 else 'D' if score >= 20 else 'F'

        return api_response({
            'lat': lat, 'lon': lon,
            'radius_miles':     radius,
            'coverage_score':   score,
            'grade':            grade,
            'verdict':          'Excellent' if score >= 80 else 'Good' if score >= 60 else 'Average' if score >= 40 else 'Poor' if score >= 20 else 'Charging Desert',
            'chargers_nearby':  unique_chargers,
            'avg_capacity':     round(avg_capacity, 1),
            'networks_present': operators,
            'has_fast_charging':has_fast,
            'recommendation':   'Well served area' if score >= 60 else 'Limited coverage — plan charging in advance',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 4. NEAREST CHARGERS + PREDICTION ────────────────────────
@app.route('/api/v1/nearest')
@require_api_key
def api_nearest():
    try:
        lat         = float(request.args.get('lat', 51.5))
        lon         = float(request.args.get('lon', -0.1))
        max_results = min(int(request.args.get('max_results', 5)), 20)
        country     = request.args.get('country', 'UK').upper()

        import pandas as pd, pickle, math
        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        unique = df[['lat','lon','operator','capacity','location_type']].drop_duplicates(['lat','lon'])
        unique = unique.copy()
        unique['dist'] = unique.apply(
            lambda r: math.sqrt((r['lat']-lat)**2 + (r['lon']-lon)**2), axis=1
        )
        nearest = unique.nsmallest(max_results, 'dist')

        country_map  = {'UK':0,'US':1,'EU':2}
        loc_map      = {'motorway':3,'supermarket':2,'council':1,'tesla':2,'other':1}
        hour         = int(request.args.get('hour', datetime.datetime.now().hour))
        day          = int(request.args.get('day',  datetime.datetime.now().weekday()))
        c_code       = country_map.get(country, 0)

        model = None
        if os.path.exists('charger_model.pkl'):
            model = pickle.load(open('charger_model.pkl','rb'))

        results = []
        for _, row in nearest.iterrows():
            dist_miles = round(row['dist'] * 69, 2)
            loc_code   = loc_map.get(row['location_type'], 1)
            cap        = min(int(row['capacity']), 20)

            prob_free = 50.0
            if model:
                fv = [[hour, day, 1 if day>=5 else 0, cap,
                        row['lat'], row['lon'], c_code, loc_code]]
                prob_free = round(float(model.predict_proba(fv)[0][0]) * 100, 1)

            results.append({
                'lat':           round(float(row['lat']), 5),
                'lon':           round(float(row['lon']), 5),
                'operator':      str(row['operator']),
                'capacity':      cap,
                'location_type': str(row['location_type']),
                'distance_miles':dist_miles,
                'probability_free': prob_free,
                'prediction':    'free' if prob_free >= 50 else 'busy',
            })

        results.sort(key=lambda x: x['distance_miles'])
        return api_response({
            'lat': lat, 'lon': lon,
            'hour': hour, 'day': day,
            'total_found': len(results),
            'chargers': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 5. HEATMAP DATA ──────────────────────────────────────────
@app.route('/api/v1/heatmap')
@require_api_key
def api_heatmap():
    try:
        region = request.args.get('region', 'uk').lower()
        res    = request.args.get('resolution', 'low')

        import pandas as pd
        regions = {
            'uk':  {'country':'UK',  'lat':(49.9,58.7), 'lon':(-6.0,1.8)},
            'us':  {'country':'US',  'lat':(24.5,49.0), 'lon':(-125.0,-66.0)},
            'eu':  {'country':'EU',  'lat':(35.0,72.0), 'lon':(-10.0,32.0)},
        }
        reg = regions.get(region, regions['uk'])
        df  = load_charger_data(reg['country'])
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        step = 0.2 if res == 'high' else 0.5
        df_f = df[
            (df['lat'].between(*reg['lat'])) &
            (df['lon'].between(*reg['lon']))
        ]

        df_f = df_f.copy()
        df_f['grid_lat'] = (df_f['lat'] / step).round() * step
        df_f['grid_lon'] = (df_f['lon'] / step).round() * step
        grid = df_f.groupby(['grid_lat','grid_lon']).size().reset_index(name='count')
        max_count = int(grid['count'].max()) if len(grid) > 0 else 1

        points = []
        for _, row in grid.iterrows():
            points.append({
                'lat':       round(float(row['grid_lat']), 3),
                'lon':       round(float(row['grid_lon']), 3),
                'count':     int(row['count']),
                'intensity': round(float(row['count']) / max_count, 3),
            })

        return api_response({
            'region':      region,
            'resolution':  res,
            'grid_step':   step,
            'total_points':len(points),
            'max_density': max_count,
            'points':      points[:3000],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 6. COUNTRY COMPARISON ────────────────────────────────────
@app.route('/api/v1/compare')
@require_api_key
def api_compare():
    try:
        countries_param = request.args.get('countries', 'UK,US,EU')
        countries = [c.strip().upper() for c in countries_param.split(',')]

        import pandas as pd
        df_all = load_charger_data(None)
        if df_all is None:
            return jsonify({'error': 'Data not available'}), 503

        # Population for density calc
        populations = {'UK':67_000_000,'US':331_000_000,'EU':450_000_000}
        results = []
        for country in countries:
            df_c = df_all[df_all['country'] == country]
            if len(df_c) == 0:
                continue
            unique = df_c[['lat','lon']].drop_duplicates()
            pop    = populations.get(country, 100_000_000)
            results.append({
                'country':           country,
                'total_chargers':    len(unique),
                'chargers_per_100k': round(len(unique) / pop * 100_000, 2),
                'avg_capacity':      round(float(df_c['capacity'].mean()), 1),
                'top_operator':      df_c['operator'].mode()[0] if len(df_c) > 0 else 'Unknown',
                'top_location_type': df_c['location_type'].mode()[0] if len(df_c) > 0 else 'Unknown',
                'data_points':       len(df_c),
                'operators_count':   df_c['operator'].nunique(),
            })

        results.sort(key=lambda x: -x['chargers_per_100k'])
        for i, r in enumerate(results):
            r['rank'] = i + 1

        return api_response({
            'countries_compared': countries,
            'winner':   results[0]['country'] if results else None,
            'results':  results,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 7. OPERATOR COMPARISON ───────────────────────────────────
@app.route('/api/v1/operators/compare')
@require_api_key
def api_operators_compare():
    try:
        names_param = request.args.get('names', 'Tesla,BP Pulse')
        names = [n.strip() for n in names_param.split(',')]

        import pandas as pd
        df = load_charger_data(None)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        results = []
        for name in names:
            mask = df['operator'].str.lower().str.contains(name.lower(), na=False)
            grp  = df[mask]
            if len(grp) == 0:
                results.append({'operator': name, 'found': False})
                continue
            unique = grp[['lat','lon']].drop_duplicates()
            results.append({
                'operator':        name,
                'found':           True,
                'total_chargers':  len(unique),
                'avg_capacity':    round(float(grp['capacity'].mean()), 1),
                'countries':       list(grp['country'].unique()),
                'location_types':  grp['location_type'].value_counts().to_dict(),
                'peak_hour':       int(grp['hour'].mode()[0]),
                'data_points':     len(grp),
            })

        results.sort(key=lambda x: -(x.get('total_chargers', 0)))
        return api_response({
            'operators_compared': names,
            'winner': results[0]['operator'] if results and results[0].get('found') else None,
            'results': results,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 8. 24HR AVAILABILITY FORECAST ───────────────────────────
@app.route('/api/v1/forecast')
@require_api_key
def api_forecast():
    try:
        lat      = float(request.args.get('lat', 51.5))
        lon      = float(request.args.get('lon', -0.1))
        capacity = int(request.args.get('capacity', 2))
        country  = request.args.get('country', 'UK').upper()
        loc_type = request.args.get('location_type', 'other')

        import pickle
        country_map = {'UK':0,'US':1,'EU':2}
        loc_map     = {'motorway':3,'supermarket':2,'council':1,'tesla':2,'other':1}
        c_code      = country_map.get(country, 0)
        loc_code    = loc_map.get(loc_type, 1)
        today       = datetime.datetime.now().weekday()
        cap         = min(capacity, 20)

        model = None
        if os.path.exists('charger_model.pkl'):
            model = pickle.load(open('charger_model.pkl','rb'))

        forecast = []
        for h in range(24):
            prob_free = 50.0
            if model:
                fv = [[h, today, 1 if today>=5 else 0, cap, lat, lon, c_code, loc_code]]
                prob_free = round(float(model.predict_proba(fv)[0][0]) * 100, 1)
            forecast.append({
                'hour':            h,
                'time':            f'{h:02d}:00',
                'probability_free': prob_free,
                'probability_busy': round(100 - prob_free, 1),
                'prediction':      'free' if prob_free >= 50 else 'busy',
                'confidence':      'high' if abs(prob_free - 50) > 25 else 'medium' if abs(prob_free - 50) > 10 else 'low',
            })

        best  = max(forecast, key=lambda x: x['probability_free'])
        worst = min(forecast, key=lambda x: x['probability_free'])

        return api_response({
            'lat': lat, 'lon': lon,
            'country': country,
            'forecast_day': datetime.datetime.now().strftime('%A'),
            'best_time':  best['time'],
            'worst_time': worst['time'],
            'best_probability_free':  best['probability_free'],
            'worst_probability_free': worst['probability_free'],
            'hourly_forecast': forecast,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 9. INFRASTRUCTURE GAP SCORE ──────────────────────────────
@app.route('/api/v1/gap-score')
@require_api_key
def api_gap_score():
    try:
        lat     = float(request.args.get('lat', 51.5))
        lon     = float(request.args.get('lon', -0.1))
        country = request.args.get('country', 'UK').upper()

        import pandas as pd
        df = load_charger_data(country)
        if df is None:
            return jsonify({'error': 'Data not available'}), 503

        # Check chargers within 1, 3, 5, 10 miles
        def count_within(miles):
            deg = miles * 0.014
            return len(df[
                (abs(df['lat'] - lat) < deg) &
                (abs(df['lon'] - lon) < deg)
            ][['lat','lon']].drop_duplicates())

        c1  = count_within(1)
        c3  = count_within(3)
        c5  = count_within(5)
        c10 = count_within(10)

        # Gap score — higher = bigger gap = more need
        gap = 100
        gap -= min(c1 * 20, 40)
        gap -= min(c3 * 5,  25)
        gap -= min(c5 * 2,  20)
        gap -= min(c10 * 1, 15)
        gap = max(gap, 0)

        priority = 'Critical' if gap >= 80 else 'High' if gap >= 60 else 'Medium' if gap >= 40 else 'Low' if gap >= 20 else 'Well Served'

        return api_response({
            'lat': lat, 'lon': lon,
            'country': country,
            'gap_score':          gap,
            'infrastructure_priority': priority,
            'chargers_within_1_mile':  c1,
            'chargers_within_3_miles': c3,
            'chargers_within_5_miles': c5,
            'chargers_within_10_miles':c10,
            'recommendation': 'Urgent infrastructure investment needed' if gap >= 80
                         else 'New chargers would significantly help this area' if gap >= 60
                         else 'Moderate coverage — additional chargers beneficial' if gap >= 40
                         else 'Adequate coverage for most EV drivers',
            'useful_for': ['Local councils','Energy companies','EV network operators','Urban planners'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ── 10. BATCH PREDICT ────────────────────────────────────────
@app.route('/api/v1/predict/batch', methods=['POST'])
@require_api_key
def api_batch_predict():
    try:
        import pickle
        data     = request.get_json() or {}
        chargers = data.get('chargers', [])

        if not chargers:
            return jsonify({'error': 'Provide chargers array in request body'}), 400
        if len(chargers) > 50:
            return jsonify({'error': 'Maximum 50 chargers per batch request'}), 400

        country_map = {'UK':0,'US':1,'EU':2}
        loc_map     = {'motorway':3,'supermarket':2,'council':1,'tesla':2,'other':1}
        now         = datetime.datetime.now()

        model = None
        if os.path.exists('charger_model.pkl'):
            model = pickle.load(open('charger_model.pkl','rb'))

        results = []
        for ch in chargers:
            lat      = float(ch.get('lat', 51.5))
            lon      = float(ch.get('lon', -0.1))
            hour     = int(ch.get('hour', now.hour))
            day      = int(ch.get('day',  now.weekday()))
            cap      = min(int(ch.get('capacity', 2)), 20)
            country  = ch.get('country', 'UK').upper()
            loc_type = ch.get('location_type', 'other')
            c_code   = country_map.get(country, 0)
            loc_code = loc_map.get(loc_type, 1)

            prob_free = 50.0
            if model:
                fv = [[hour, day, 1 if day>=5 else 0, cap, lat, lon, c_code, loc_code]]
                prob_free = round(float(model.predict_proba(fv)[0][0]) * 100, 1)

            results.append({
                'id':               ch.get('id', f'{lat},{lon}'),
                'lat':              lat,
                'lon':              lon,
                'probability_free': prob_free,
                'probability_busy': round(100 - prob_free, 1),
                'prediction':       'free' if prob_free >= 50 else 'busy',
            })

        free_count = sum(1 for r in results if r['prediction'] == 'free')
        return api_response({
            'total_chargers':   len(results),
            'predicted_free':   free_count,
            'predicted_busy':   len(results) - free_count,
            'results':          results,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

