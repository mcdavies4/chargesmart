"""
ChargeSmart Data Audit
Run this in your project folder to see exactly what data you have.
Usage: python check_data.py
"""

import os, json, glob
from datetime import datetime

print("""
╔══════════════════════════════════════════════════════╗
║         ChargeSmart — Data Audit                     ║
╚══════════════════════════════════════════════════════╝
""")

# ── 1. RAW COLLECTOR FILES ─────────────────────────────────────
print("=" * 55)
print("RAW DATA FILES (data_*.json)")
print("=" * 55)

data_files = sorted(glob.glob("data_*.json"))
total_raw = 0
region_counts = {}

if not data_files:
    print("  ❌ No data_*.json files found")
else:
    for f in data_files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            count = len(data) if isinstance(data, list) else len(data.get('chargeDevices', data.get('results', [])))
            size_kb = os.path.getsize(f) / 1024
            region = f.replace('data_', '').replace('.json', '')
            region_counts[region] = count
            total_raw += count
            print(f"  ✅ {f:<35} {count:>8,} chargers  ({size_kb:.0f}KB)")
        except Exception as e:
            print(f"  ⚠️  {f} — error reading: {e}")

print(f"\n  📦 Total raw files:     {len(data_files)}")
print(f"  ⚡ Total raw chargers:  {total_raw:,}")

# ── 2. ENRICHED CSV ────────────────────────────────────────────
print("\n" + "=" * 55)
print("ENRICHED DATA (enriched_data.csv)")
print("=" * 55)

if os.path.exists("enriched_data.csv"):
    try:
        import pandas as pd
        df = pd.read_csv("enriched_data.csv")
        size_mb = os.path.getsize("enriched_data.csv") / 1024 / 1024
        print(f"  ✅ Rows:        {len(df):,}")
        print(f"  ✅ Columns:     {len(df.columns)} — {list(df.columns)}")
        print(f"  ✅ File size:   {size_mb:.1f} MB")

        if 'country' in df.columns:
            print(f"\n  By country:")
            for country, count in df['country'].value_counts().items():
                pct = count / len(df) * 100
                bar = "█" * int(pct / 2)
                print(f"    {country:<6} {count:>10,}  ({pct:.1f}%)  {bar}")

        if 'location_type' in df.columns:
            print(f"\n  By location type:")
            for loc, count in df['location_type'].value_counts().head(8).items():
                print(f"    {loc:<20} {count:>10,}")

        if 'hour' in df.columns:
            print(f"\n  Time samples:  {df['hour'].nunique()} unique hours")

        print(f"\n  Unique charger IDs: {df['id'].nunique():,}" if 'id' in df.columns else "")
        print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}" if 'timestamp' in df.columns else "")

    except ImportError:
        print("  ⚠️  pandas not installed — run: pip install pandas")
    except Exception as e:
        print(f"  ❌ Error reading CSV: {e}")
else:
    print("  ❌ enriched_data.csv not found")
    print("     Run: python enricher.py")

# ── 3. MODEL ───────────────────────────────────────────────────
print("\n" + "=" * 55)
print("TRAINED MODEL (charger_model.pkl)")
print("=" * 55)

if os.path.exists("charger_model.pkl"):
    import pickle
    size_mb = os.path.getsize("charger_model.pkl") / 1024 / 1024
    with open("charger_model.pkl", "rb") as f:
        model = pickle.load(f)
    print(f"  ✅ Model type:    {type(model).__name__}")
    print(f"  ✅ Features:      {model.n_features_in_}")
    print(f"  ✅ File size:     {size_mb:.1f} MB")
    if os.path.exists("model_features.txt"):
        feats = open("model_features.txt").read().split(",")
        print(f"  ✅ Feature list: {feats}")
else:
    print("  ❌ charger_model.pkl not found")
    print("     Run: python model.py")

# ── 4. SUMMARY ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)

enriched_rows = 0
if os.path.exists("enriched_data.csv"):
    try:
        import pandas as pd
        enriched_rows = len(pd.read_csv("enriched_data.csv"))
    except: pass

print(f"""
  Raw collector files:   {len(data_files)} regions
  Raw charger records:   {total_raw:,}
  Enriched ML rows:      {enriched_rows:,}
  Model trained:         {'✅ Yes' if os.path.exists('charger_model.pkl') else '❌ No'}

  Next steps:
  {'→ Run: python enricher.py    (create enriched_data.csv)' if not os.path.exists('enriched_data.csv') else '✅ enriched_data.csv ready'}
  {'→ Run: python model.py       (train model)' if not os.path.exists('charger_model.pkl') else '✅ model ready'}
  {'→ Run: python pipeline.py all  (collect all 30 regions)' if len(data_files) < 10 else '✅ good data coverage'}
""")
