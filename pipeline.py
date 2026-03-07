"""
ChargeSmart Auto Pipeline
Runs: Collect → Enrich → Train → (optionally push to git)

Usage:
    python pipeline.py                  # collect existing regions, enrich, train
    python pipeline.py emerging         # collect emerging markets only
    python pipeline.py all              # collect all 30 regions
    python pipeline.py enrich-only      # just enrich + train existing data files
"""

import subprocess, sys, os, time
from datetime import datetime

def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ {label} failed with code {result.returncode}")
        sys.exit(1)
    print(f"✅ {label} complete")

mode = sys.argv[1] if len(sys.argv) > 1 else 'existing'

print(f"""
╔══════════════════════════════════════╗
║   ChargeSmart Auto Pipeline          ║
║   Mode: {mode:<28}║
║   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}       ║
╚══════════════════════════════════════╝
""")

# Step 1: Collect (skip if enrich-only)
if mode != 'enrich-only':
    import glob
    before = len(glob.glob('data_*.json'))
    
    # Run collector once (not the infinite loop)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 1: Collecting data (mode={mode})...")
    
    import collector
    modes = {
        'all':      collector.ALL_REGIONS,
        'existing': collector.EXISTING_REGIONS,
        'emerging': collector.EMERGING_MARKETS,
        'africa':   collector.AFRICA_REGIONS,
        'uk':       ['uk'],
    }
    regions = modes.get(mode, collector.EXISTING_REGIONS)
    count = collector.collect_all(regions)
    print(f"✅ Collected {count:,} chargers")
    
    after = len(glob.glob('data_*.json'))
    print(f"   Data files: {before} → {after}")
else:
    print("Skipping collection (enrich-only mode)")

# Step 2: Enrich
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: Enriching data...")
import enricher
enricher.enrich()

# Check enriched output
if os.path.exists('enriched_data.csv'):
    import pandas as pd
    df = pd.read_csv('enriched_data.csv')
    print(f"✅ enriched_data.csv: {len(df):,} rows")
    if 'country' in df.columns:
        print(f"   Countries: {dict(df['country'].value_counts().head(10))}")
else:
    print("❌ enriched_data.csv not created!")
    sys.exit(1)

# Step 3: Train model
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: Training model...")
run("python model.py", "Model training")

# Step 4: Summary
print(f"""
╔══════════════════════════════════════╗
║   Pipeline Complete!                 ║
╚══════════════════════════════════════╝

Next step - push to Railway:
  git add enriched_data.csv charger_model.pkl model_features.txt
  git commit -m "Update dataset and retrain model"
  git push
""")
