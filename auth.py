"""
ChargeSmart Authentication System
Magic link passwordless login + session management
"""
import json, os, secrets, hashlib
from datetime import datetime, timedelta

import os as _os
_data_dir     = '/tmp' if not _os.access('.', _os.W_OK) else '.'
USERS_FILE    = _os.path.join(_data_dir, 'users.json')
TOKENS_FILE   = _os.path.join(_data_dir, 'magic_tokens.json')
SESSIONS_FILE = _os.path.join(_data_dir, 'sessions.json')
for _f in ['users.json', 'magic_tokens.json', 'sessions.json']:
    _src = _f; _dst = _os.path.join(_data_dir, _f)
    if _os.path.exists(_src) and not _os.path.exists(_dst):
        import shutil as _sh; _sh.copy(_src, _dst)

# ── FILE HELPERS ─────────────────────────────────────────────
def load_users():
    return json.load(open(USERS_FILE)) if os.path.exists(USERS_FILE) else {}

def save_users(u):
    json.dump(u, open(USERS_FILE,'w'), indent=2)

def load_tokens():
    return json.load(open(TOKENS_FILE)) if os.path.exists(TOKENS_FILE) else {}

def save_tokens(t):
    json.dump(t, open(TOKENS_FILE,'w'), indent=2)

def load_sessions():
    return json.load(open(SESSIONS_FILE)) if os.path.exists(SESSIONS_FILE) else {}

def save_sessions(s):
    json.dump(s, open(SESSIONS_FILE,'w'), indent=2)

# ── USER MANAGEMENT ──────────────────────────────────────────
def get_or_create_user(email, name=''):
    users = load_users()
    email = email.lower().strip()
    uid   = hashlib.md5(email.encode()).hexdigest()[:16]
    if uid not in users:
        users[uid] = {
            'uid':        uid,
            'email':      email,
            'name':       name or email.split('@')[0].title(),
            'created':    datetime.now().strftime('%Y-%m-%d'),
            'plan':       'free',
            'api_key':    None,
            'favourites': [],
            'login_count':0,
            'last_login': None,
        }
        save_users(users)
    return users[uid]

def get_user_by_uid(uid):
    return load_users().get(uid)

def get_user_by_email(email):
    email = email.lower().strip()
    uid   = hashlib.md5(email.encode()).hexdigest()[:16]
    return load_users().get(uid)

def update_user(uid, updates):
    users = load_users()
    if uid in users:
        users[uid].update(updates)
        save_users(users)
    return users.get(uid)

# ── MAGIC LINKS ──────────────────────────────────────────────
def create_magic_token(email):
    """Generate a one-time login token valid for 15 minutes"""
    tokens = load_tokens()
    # Clean expired tokens
    now = datetime.now()
    tokens = {k:v for k,v in tokens.items()
              if datetime.fromisoformat(v['expires']) > now}
    token  = secrets.token_urlsafe(32)
    tokens[token] = {
        'email':   email.lower().strip(),
        'expires': (now + timedelta(minutes=15)).isoformat(),
        'used':    False,
    }
    save_tokens(tokens)
    return token

def verify_magic_token(token):
    """Returns (valid, email, error) tuple"""
    tokens = load_tokens()
    if token not in tokens:
        return False, None, 'Invalid or expired link'
    t = tokens[token]
    if t['used']:
        return False, None, 'This link has already been used'
    if datetime.fromisoformat(t['expires']) < datetime.now():
        return False, None, 'This link has expired — request a new one'
    tokens[token]['used'] = True
    save_tokens(tokens)
    return True, t['email'], None

# ── SESSIONS ─────────────────────────────────────────────────
def create_session(uid):
    """Create a 30-day session for a user"""
    sessions = load_sessions()
    # Clean old sessions
    now = datetime.now()
    sessions = {k:v for k,v in sessions.items()
                if datetime.fromisoformat(v['expires']) > now}
    session_id = secrets.token_urlsafe(48)
    sessions[session_id] = {
        'uid':     uid,
        'created': now.isoformat(),
        'expires': (now + timedelta(days=30)).isoformat(),
    }
    save_sessions(sessions)
    return session_id

def get_session_user(session_id):
    """Returns user dict if session valid, None otherwise"""
    if not session_id:
        return None
    sessions = load_sessions()
    s = sessions.get(session_id)
    if not s:
        return None
    if datetime.fromisoformat(s['expires']) < datetime.now():
        return None
    return get_user_by_uid(s['uid'])

def delete_session(session_id):
    sessions = load_sessions()
    sessions.pop(session_id, None)
    save_sessions(sessions)

# ── EMAIL ─────────────────────────────────────────────────────
def send_magic_link(email, token, base_url):
    """Send magic link via Brevo (formerly Sendinblue) SMTP API"""
    link = f"{base_url}auth/verify?token={token}"

    brevo_key  = os.environ.get('BREVO_API_KEY', '')
    from_email = os.environ.get('FROM_EMAIL', 'azubuikedavies@gmail.com')
    from_name  = os.environ.get('FROM_NAME', 'ChargeSmart')

    html_body = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:0 auto;padding:32px;background:#0d1117;color:#e0e0e0;border-radius:16px;">
        <div style="font-size:32px;margin-bottom:12px;">⚡</div>
        <h2 style="margin-bottom:8px;color:#ffffff;font-size:22px;">Your login link</h2>
        <p style="color:#9999aa;margin-bottom:28px;line-height:1.6;">
            Click the button below to sign in to ChargeSmart.<br>
            This link expires in <strong style="color:#ffffff;">15 minutes</strong> and can only be used once.
        </p>
        <a href="{link}"
           style="display:inline-block;background:#00ff87;color:#0a0a0f;padding:14px 32px;
                  border-radius:8px;text-decoration:none;font-weight:700;
                  font-family:monospace;letter-spacing:1px;font-size:14px;">
            SIGN IN TO CHARGESMART →
        </a>
        <p style="color:#666;font-size:12px;margin-top:28px;">
            Or copy this link:<br>
            <a href="{link}" style="color:#00ff87;word-break:break-all;">{link}</a>
        </p>
        <hr style="border:none;border-top:1px solid #1e1e2e;margin:24px 0;">
        <p style="color:#555;font-size:11px;">
            If you didn't request this email, you can safely ignore it.<br>
            © ChargeSmart · chargesmart.online
        </p>
    </div>
    """

    text_body = f"Your ChargeSmart login link:\n{link}\n\nExpires in 15 minutes. Single use only."

    if brevo_key:
        try:
            import urllib.request, json as _json
            payload = _json.dumps({
                "sender":  {"name": from_name, "email": from_email},
                "to":      [{"email": email}],
                "subject": "⚡ Your ChargeSmart login link",
                "htmlContent": html_body,
                "textContent": text_body,
            }).encode('utf-8')

            req = urllib.request.Request(
                'https://api.brevo.com/v3/smtp/email',
                data=payload,
                headers={
                    'api-key':      brevo_key,
                    'Content-Type': 'application/json',
                    'Accept':       'application/json',
                },
                method='POST'
            )
            resp = urllib.request.urlopen(req, timeout=10)
            resp_body = resp.read().decode('utf-8')
            print(f"✅ Brevo sent to {email} — status {resp.status}: {resp_body[:100]}")
            return True, 'Email sent via Brevo'
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            print(f"❌ Brevo HTTP {e.code}: {err_body}")
            print(f"\n{'='*60}")
            print(f"MAGIC LINK FALLBACK for {email}:")
            print(f"{link}")
            print(f"{'='*60}\n")
            return False, f"Brevo {e.code}: {err_body}"
        except Exception as e:
            print(f"❌ Brevo error: {e}")
            print(f"\n{'='*60}")
            print(f"MAGIC LINK FALLBACK for {email}:")
            print(f"{link}")
            print(f"{'='*60}\n")
            return False, str(e)
    else:
        # Dev mode — no key set, just log the link
        print(f"\n{'='*60}")
        print(f"DEV MODE — MAGIC LINK for {email}:")
        print(f"{link}")
        print(f"{'='*60}\n")
        return True, 'link_logged'
