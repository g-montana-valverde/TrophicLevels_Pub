import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from pathlib import Path
import json
from statsmodels.stats import multitest

from functions_pipeline import (
    harmonize_data,
)

# -------------------------------------------------------
# 1. Parse arguments
# -------------------------------------------------------
parser = argparse.ArgumentParser(description="Run TL and Directedness Group Comparisons")
parser.add_argument("--data-dir", required=True, help="Folder with MAT/CSV files")
parser.add_argument("--out-dir", required=True, help="Where results will be saved")
args = parser.parse_args()

data_dir = Path(args.data_dir)
derivatives_dir = Path(args.derivatives_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(exist_ok=True, parents=True)

group_class = 'HCn-HCp-MCIp-ADp'

parts = [p.strip() for p in group_class.split('-') if p.strip()]
if not parts:
	raise ValueError(f"Invalid group_class: {group_class!r}")
for i, part in enumerate(parts, start=1):
	globals()[f"g{i}"] = part

# -------------------------------------------------------
# 2. Load data
# -------------------------------------------------------
region_labels = pd.read_csv(data_dir + 'dbs80_labels.txt', sep="\t", header=None)[1].values
net_names = ['VN', 'SMN', 'DAN', 'SAN', 'LN', 'CN', 'DMN']
with open(data_dir + 'reg2net.json', 'r') as f_json:
	reg2net = json.load(f_json)
  
print("Loading data...")
groups = [globals()[k] for k in sorted(globals().keys()) if k.startswith('g') and k[1:].isdigit()]
dfs, df_dirs = [], []
for grp in groups:
  data_tl = loadmat(data_dir / f"TL_ADNI3_{grp}_dbs80.mat")["TL"]
  data_directedness = loadmat(data_dir / f"DIR_ADNI3_{grp}_dbs80.mat")["directedness"][0]
  deriv_dir = derivatives_dir / grp / 'derivatives'
  subs = sorted(int(p.name.removeprefix('sub-')) for p in deriv_dir.iterdir() if p.is_dir() and p.name.startswith('sub-'))
  df = pd.DataFrame(data_tl, columns=region_labels); df.insert(0, 'ID', subs); df.insert(1, 'Group', grp); dfs.append(df)
  df_dir = pd.DataFrame(data_directedness, columns=['directedness']); df_dir.insert(0, 'ID', subs); df_dir.insert(1, 'Group', grp); df_dirs.append(df_dir)

tl_df = pd.concat(dfs, ignore_index=True)
dir_df = pd.concat(df_dirs, ignore_index=True)

# Age, gender and education for each subject
demo = pd.read_excel(data_dir / "Demographics.xlsx")
# Scan info per subject
sites = pd.read_excel(data_dir / "Sites.xlsx")

# Combine covariates and site info
cov = pd.concat([demo.reset_index(drop=True), sites.reset_index(drop=True)], axis=1)

# -------------------------------------------------------
# 3. Harmonize data
# -------------------------------------------------------
print("Harmonizing...")
TL_h[region_labels] = harmonize_data(tl_df[region_labels].to_numpy(), cov)
dir_h['directedness'] = harmonize_data(dir_df['directedness'].to_numpy().reshape(-1, 1), cov)

# -------------------------------------------------------
# 4. Convert to Network Level
# -------------------------------------------------------
print("Computing TL network-level...")
TL_net_h = TL_h[['ID', 'Group']].copy()
for net in net_names:
  regions_in_net = [region for region, sfn_map in reg2net.items() if sfn in (sfn_map or "") and region in TL_h.columns]
  exclude_subcortical = ['hippocampus', 'amygdala', 'thalamus', 'caudate', 'accumbens', 'putamen', 'gpe', 'gpi', 'stn']
	regions_in_net = [region for region in regions_in_net if not any(keyword in region.lower() for keyword in exclude_subcortical)]
  if regions_in_sfn:
			TL_net_h[net] = TL_h[regions_in_net].mean(axis=1)
	else:
			TL_net_h[net] = np.nan

# -------------------------------------------------------
# 5. Group Comparisons
# -------------------------------------------------------
print("Running group comparisons...")
pairwise = []
n = len(groups)
for delta in range(1, n):
	for i in range(0, n - delta):
		pairwise.append((groups[i], groups[i + delta]))

# 5.1. Directedness
tdir_stats = []
tdir_pvs = []
for a, b in pairwise:
  tdir_stat, td_pv = run_comparisons(dir_h, demo, a, b, test)
  tdir_stats.append(tdir_stat)
  tdir_pvs.append(tdir_pv)
_, tdir_pvs_corr= np.array(multitest.fdrcorrection(tdir_pvs))
plot_directedness(dir_df, tdir_pvs_corr, test, out_path=str(out_dir / "DirectednessGroupComparison.png"))

# -------------------------------------------------------
# 6. Plotting
# -------------------------------------------------------
print("Plotting histogram...")
zTL=zscore(TL_h.mean(axis=0))
plot_histogram(zTL, np.percentile(zTL, 33), np.percentile(zTL, 66), bins=9, , out_path=str(out_dir / "Histogram.png"))

print("Plotting correlations...")
plot_correlation_with_mean(metrics['out_in_degree_ratio_array_GEC'], TL_h, metrics['out_in_degree_ratio_array_GEC'].mean(axis=0), TL_h.mean(axis=0), 'Out-In Degree Ratio', 'Trophic Level', out_path=str(out_dir / "OutInRatio.png"))
plot_correlation_with_mean(metrics['mean_path_len_array_GEC'], TL_h, metrics['mean_path_len_array_GEC'].mean(axis=0), TL_h.mean(axis=0), 'Mean Path Length', 'Trophic Level', out_path=str(out_dir / "MeanPathLength.png"))

print("Pipeline complete.")
