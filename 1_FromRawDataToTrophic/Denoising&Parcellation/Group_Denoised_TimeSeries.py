import os
import numpy
from scipy.io import loadmat, savemat

GroupName='HCn'

atlas= 'dbs80'
ROIs = 80

derivatives = '/path/to/output/derivatives/'

out_dir = '/path/to/output/'

sample_file = out_dir + 'ADNI3_' + GroupName + '_' + atlas + '.mat'

GROUP_IDs = sorted(os.listdir(derivatives))

GROUP_SAMPLE = numpy.empty(len(GROUP_IDs), dtype=object)

for i in range(0,len(GROUP_SAMPLE)):
	
	sample = loadmat(derivatives + GROUP_IDs[i] + '/' + GROUP_IDs[i] + '_' + atlas + '_denoised.mat')
	
	GROUP_SAMPLE[i] = sample['ts']
	

savemat(sample_file, {'data': GROUP_SAMPLE}, do_compression=True)
