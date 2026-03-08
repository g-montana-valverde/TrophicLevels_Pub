from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat

# Configuration
GROUP_NAME = 'ADp'
ATLAS = 'dbs80'
DERIVATIVES_DIR = Path('/path/to/output/derivatives/')
OUTPUT_DIR = Path('/path/to/output/')
OUTPUT_FILE = OUTPUT_DIR / f'ADNI3_{GROUP_NAME}_{ATLAS}.mat'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Get sorted list of subject IDs
subject_ids = sorted([d for d in DERIVATIVES_DIR.iterdir() if d.is_dir()])

# Load time series data
group_sample = np.empty(len(subject_ids), dtype=object)

for i, subject_dir in enumerate(subject_ids):
	mat_file = subject_dir / f'{subject_dir.name}_{ATLAS}_denoised.mat'


	sample = loadmat(mat_file)
	group_sample[i] = sample['ts']

# Save aggregated group data
savemat(OUTPUT_FILE, {'data': group_sample}, do_compression=True)