"""
ChargeSmart API Key System
Handles key generation, validation, rate limiting and usage tracking
"""
import json
import os
import secrets
import string
from datetime import datetime, date

KEYS_FILE = 'api_keys.json'

TIER_LIMITS = {
    'free':       100,
    'developer':  10000,
    'business':   100000,
    'enterprise': 999999999,
}

TIER_PRICES = {
    'free':       0,
    'developer':  49,
    'business':   199,
    'enterprise': 999,
}

def load_keys():
    if os.path.exists(KEYS_FILE):
        with open(KEYS_FILE) as f:
            return json.load(f)
    return {}

def save_keys(keys):
    with open(KEYS_FILE, 'w') as f:
        json.dump(keys, f, indent=2)

def generate_key():
    chars = string.ascii_letters + string.digits
    token = ''.join(secrets.choice(chars) for _ in range(32))
    return f"cs_live_{token}"

def create_api_key(email, tier='free'):
    keys = load_keys()
    # Check if email already has a key
    for key, data in keys.items():
        if data.get('email') == email:
            return {'key': key, 'tier': data['tier'], 'existing': True}
    key = generate_key()
    keys[key] = {
        'email':      email,
        'tier':       tier,
        'created':    datetime.now().strftime('%Y-%m-%d'),
        'usage':      {},
        'total_calls': 0,
    }
    save_keys(keys)
    return {'key': key, 'tier': tier, 'existing': False}

def validate_key(api_key):
    """Returns (valid, tier, remaining_calls, error_message)"""
    if not api_key:
        return False, None, 0, 'Missing API key. Add Authorization: Bearer YOUR_KEY header'
    keys = load_keys()
    if api_key not in keys:
        return False, None, 0, 'Invalid API key. Get yours free at chargesmart.online/developers'
    data = keys[api_key]
    tier  = data.get('tier', 'free')
    limit = TIER_LIMITS.get(tier, 100)
    today = str(date.today())
    usage_today = data.get('usage', {}).get(today, 0)
    if usage_today >= limit:
        return False, tier, 0, f'Daily limit reached ({limit} calls/day on {tier} tier). Upgrade at chargesmart.online/developers'
    return True, tier, limit - usage_today, None

def record_usage(api_key):
    keys = load_keys()
    if api_key not in keys:
        return
    today = str(date.today())
    if 'usage' not in keys[api_key]:
        keys[api_key]['usage'] = {}
    keys[api_key]['usage'][today] = keys[api_key]['usage'].get(today, 0) + 1
    keys[api_key]['total_calls'] = keys[api_key].get('total_calls', 0) + 1
    save_keys(keys)

def get_key_stats(api_key):
    keys = load_keys()
    if api_key not in keys:
        return None
    data = keys[api_key]
    tier  = data.get('tier', 'free')
    limit = TIER_LIMITS[tier]
    today = str(date.today())
    today_usage = data.get('usage', {}).get(today, 0)
    # Last 7 days usage
    from datetime import timedelta
    weekly = []
    for i in range(7):
        d = str(date.today() - timedelta(days=i))
        weekly.append({'date': d, 'calls': data.get('usage', {}).get(d, 0)})
    return {
        'email':         data.get('email'),
        'tier':          tier,
        'limit_per_day': limit,
        'used_today':    today_usage,
        'remaining':     max(0, limit - today_usage),
        'total_calls':   data.get('total_calls', 0),
        'created':       data.get('created'),
        'weekly_usage':  list(reversed(weekly)),
    }
