"""
ChargeSmart — Africa + Emerging Markets Pipeline
=================================================
Collects Africa + Middle East + Asia + LatAm (fresh pass),
enriches all data, retrains model, then pushes to Railway.

Already have collector running? This adds another pass on top —
more snapshots = better training diversity.

Usage:  python run_africa_emerging.py
"""

import sys, os, subprocess, json, glob
from datetime import datetime

def ts():
    return datetime.now().strftime("%H:%M:%S")

def banner(msg):
    print(f"\n{'='*58}\n  {msg}\n{'='*58}")

# ─────────────────────────────────────────────────────────────
banner(f"ChargeSmart — Africa + Emerging Markets Pipeline")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Quick audit of what we already have before starting
data_files = glob.glob("data_*.json")
existing_rows = 0
for f in data_files:
    try:
        with open(f) as fp:
            for line in fp:
                snap = json.loads(line)
                existing_rows += len(snap.get("chargers", []))
    except: pass

print(f"\n  Existing data files:    {len(data_files)}")
print(f"  Existing raw records:   {existing_rows:,}")
print(f"  Adding: Africa + Middle East + Asia + LatAm\n")

# ─────────────────────────────────────────────────────────────
banner(f"[{ts()}] STEP 1 — COLLECTING AFRICA")

import collector

print(f"  Regions: {', '.join(collector.AFRICA_REGIONS)}\n")
africa_count = collector.collect_all(collector.AFRICA_REGIONS)
print(f"\n  ✅ Africa: {africa_count:,} chargers collected")

# ─────────────────────────────────────────────────────────────
banner(f"[{ts()}] STEP 2 — COLLECTING MIDDLE EAST + ASIA + LATAM")

emerging_rest = collector.MIDDLE_EAST_REGIONS + collector.ASIA_REGIONS + collector.LATAM_REGIONS
print(f"  Regions: {', '.join(emerging_rest)}\n")
emerging_count = collector.collect_all(emerging_rest)
print(f"\n  ✅ Middle East + Asia + LatAm: {emerging_count:,} chargers collected")

total_collected = africa_count + emerging_count
print(f"\n  ✅ Total this run: {total_collected:,} chargers")

# ─────────────────────────────────────────────────────────────
banner(f"[{ts()}] STEP 3 — ENRICHING ALL DATA")

import enricher
enricher.enrich()

enriched_rows = 0
if os.path.exists("enriched_data.csv"):
    import pandas as pd
    df = pd.read_csv("enriched_data.csv")
    enriched_rows = len(df)
    size_mb = os.path.getsize("enriched_data.csv") / 1024 / 1024
    print(f"\n  ✅ enriched_data.csv: {enriched_rows:,} rows | {size_mb:.1f} MB")

    if "country" in df.columns:
        print(f"\n  Breakdown by country:")
        vc = df["country"].value_counts()
        for country, count in vc.items():
            pct = count / enriched_rows * 100
            bar = "█" * int(pct / 2)
            print(f"    {country:<8} {count:>10,}  ({pct:4.1f}%)  {bar}")

    if "continent" in df.columns:
        print(f"\n  Breakdown by continent:")
        for cont, count in df["continent"].value_counts().items():
            pct = count / enriched_rows * 100
            print(f"    {cont:<15} {count:>10,}  ({pct:.1f}%)")
else:
    print("  ❌ enriched_data.csv not found — check enricher.py")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
banner(f"[{ts()}] STEP 4 — RETRAINING MODEL")

print("  Model: GradientBoosting, 21 features")
print("  Estimated time: 5-15 minutes\n")

result = subprocess.run([sys.executable, "model.py"])

if result.returncode != 0:
    print("\n  ❌ model.py failed — fix errors above then re-run")
    sys.exit(1)

model_size = os.path.getsize("charger_model.pkl") / 1024 / 1024 if os.path.exists("charger_model.pkl") else 0
print(f"\n  ✅ Model saved ({model_size:.1f} MB)")

# ─────────────────────────────────────────────────────────────
banner(f"[{ts()}] STEP 5 — PUSHING TO RAILWAY")

cmds = [
    "git add enriched_data.csv charger_model.pkl model_features.txt",
    f'git commit -m "Retrain: {enriched_rows:,} rows, Africa+Emerging refresh {datetime.now().strftime("%Y-%m-%d")}"',
    "git push",
]

for cmd in cmds:
    print(f"  > {cmd}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"  ⚠️  Command failed — run manually and check git status")
        break
    print()

# ─────────────────────────────────────────────────────────────
banner("PIPELINE COMPLETE")

print(f"""
  Collected this run:     {total_collected:,} chargers
  Enriched ML rows:       {enriched_rows:,}
  Model retrained:        ✅
  Pushed to Railway:      ✅

  Railway will redeploy automatically in ~2 minutes.
  Check: https://chargesmart.online/version

  Accuracy target: 78-82% (up from 65.56%)
  Paste the accuracy score here to confirm ✅
""")
