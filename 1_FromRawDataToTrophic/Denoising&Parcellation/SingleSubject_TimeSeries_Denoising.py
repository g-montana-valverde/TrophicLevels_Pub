from pathlib import Path
import os
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
import nilearn.image
import json
from nilearn.interfaces.fmriprep import load_confounds
import scipy
import sys
import pandas as pd
import numpy as np
from nilearn.datasets import load_mni152_template
import nibabel as nib
from nilearn.signal import clean
from nilearn import datasets


tk = 'rest'
atlas = 'dbs80'

atlas_file= "/path/to/atlas/dbs80symm_2mm.nii.gz"


# remove lines when finish
preprocessed_data_dir='/path/to/fMRIPrep/outputs/'
SubList = sorted([dir_name for dir_name in os.listdir(preprocessed_data_dir) if os.path.isdir(os.path.join(preprocessed_data_dir, dir_name)) and 'sub-' in dir_name])

for sub in SubList:
	
	# Derivatives directory 
	
	output_timeseries_extraction_dir = '/path/to/output/derivatives/' + sub + '/'
	isExist = os.path.exists(output_timeseries_extraction_dir)
	if not isExist:
		os.makedirs(output_timeseries_extraction_dir)
	
	# Functional NIfTI file path
	func_file = preprocessed_data_dir + sub + '/func/' + sub + '_task-' + tk + '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
	mask_file = preprocessed_data_dir + sub + '/func/' + sub + '_task-' + tk + '_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'

	
	# Obtain TR
	with open(func_file.replace('nii.gz', 'json'), 'r') as file:
		TR=json.load(file)["RepetitionTime"]
		
	confounds_simple, sample_mask = load_confounds(
					func_file,
					strategy = ['high_pass', 'motion', 'wm_csf', 'scrub'],
					motion = 'full', # full= original parameters + quadratic terms + 1st temporal derivatives +  power2 derivatives
					wm_csf = 'full',
					scrub = 5, # further removing segments after accounting for time frames with excessive motion
					fd_threshold = 0.5 # framewise displacement threshold in mm
					)

	# Atlas NIfTI file masker: https://nilearn.github.io/dev/modules/generated/nilearn.maskers.NiftiLabelsMasker.html
	
	masker = NiftiLabelsMasker(
		labels_img = nilearn.image.load_img(atlas_file),
		standardize="zscore_sample", # The signal is z-scored, shifted to zero mean and scaled to unit variance. Uses sample std
		standardize_confounds="zscore_sample", # The same with confounds
		t_r = TR
		)			
	
	
	# Confounds strategy: https://nilearn.github.io/dev/modules/generated/nilearn.interfaces.fmriprep.load_confounds.html, confound tsv and json must be in the same directory as the functional
	# Examples: https://nilearn.github.io/dev/auto_examples/03_connectivity/plot_signal_extraction.html#sphx-glr-auto-examples-03-connectivity-plot-signal-extraction-py
	
	
	if sample_mask is None:
		print(sub, ' (', SubList.index(sub)+1, '/', len(SubList), '): \033[1m' +  'No volumes removed')
	else:
		print(sub, ' (', SubList.index(sub)+1, '/', len(SubList), '): \033[1m', round(len(sample_mask)/confounds_simple.shape[0]*100, 2), '\033[0m% volumes remaining from scrubbing (', len(sample_mask), '/', confounds_simple.shape[0] , ')')
	

	# Applying regressors (and removing volumes) for denoising the time series from functional nifti file on a parcellation
	time_series = masker.fit_transform(func_file, confounds=confounds_simple, sample_mask=sample_mask)
	
	print('	Time Series shape: ', time_series.shape)
	
	
	scipy.io.savemat(output_timeseries_extraction_dir + '/' + sub + '_' + atlas + '_denoised.mat', mdict={'ts': time_series.T, 'TR': TR})
	
