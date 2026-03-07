#!/bin/bash

atlas_name='dbs80'
atlas_nifti_file="/path/to/atlas/dbs80symm_2mm.nii.gz"


cd $HOME/ADNI_DWI/
subs=$(ls)


for sub in ${subs[@]}; do
	echo $sub
	cd "/home/gmontana/HCP-EP_DWI/${sub}/${sub}_SC/"

	mrconvert 5tt_coreg_${sub}.mif 5tt_coreg_${sub}.nii.gz

	flirt -in $atlas_nifti_file -ref 5tt_coreg_${sub}.nii.gz -out atlas_aligned_nn_${name}.nii.gz -interp nearestneighbour
	tck2connectome -symmetric -zero_diagonal -tck_weights_in sift_${sub}_5M.txt tracks_${sub}_5M.tck atlas_aligned_nn_${name}.nii.gz ${sub}_connectome_${name}.csv
done
