import io
import sys
import os
import scipy
import numpy
from scipy.io import loadmat
from scipy.io import savemat
from scipy import signal
import statistics
from pathlib import Path
from tqdm import trange
import h5py

GroupName='HCn'

atlas= 'dbs80'
ROIs = 80

derivatives = '/path/to/output/derivatives/'
Tau=1

out_dir = '/path/to/output/'

sample_file = out_dir + 'ADNI3_' + GroupName + '_' + atlas + '.mat'


GROUP_IDs = sorted(os.listdir(derivatives))

GROUP_SAMPLE = numpy.empty(len(GROUP_IDs), dtype=object)

for i in trange(0,len(GROUP_SAMPLE)):
	
	sample = loadmat(derivatives + GROUP_IDs[i] + '/' + GROUP_IDs[i] + '_' + atlas + '_denoised.mat')
	
	GROUP_SAMPLE[i] = sample['ts']
	

scipy.io.savemat(sample_file, {'data': GROUP_SAMPLE}, do_compression=True)
