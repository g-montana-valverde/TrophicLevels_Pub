#!/bin/bash
#https://community.mrtrix.org/t/mtnormalise-nan-values-as-balance-factors/4552/15
################################################################
################ TERMINAL COMMAND LINES ########################
################################################################
NumberOfTracks=5000000
PhaseEncoding=j ## j- == AP and j == PA, i- == RL, i ==LR

datadir=$HOME/ADNI3_DWI/

cd $datadir

sub=$1
sub=$(echo "$sub" | sed 's/\x1b\[[0-9;]*m//g')

cd $sub
RAW_DWI=$(find . -type f -name "*DWI.nii.gz" -print -quit | sed 's|^\./||')
BVEC=$(find . -type f -name "*.bvec" -print -quit | sed 's|^\./||')
BVAL=$(find . -type f -name "*.bval" -print -quit | sed 's|^\./||')
ANAT=$(find . -type f -name "*T1w.nii.gz" -print -quit | sed 's|^\./||')

mkdir ${sub}_SC
cp $RAW_DWI $BVEC $BVAL $ANAT ${sub}_SC
cd ${sub}_SC

################################################################
################# PREPROCESSING ################################
################################################################

#DWI image conversion from Nifty to format that can be read by MRtrix: .mif
echo -e '\e[33mPreprocessing S1: Convertion from nii.gz to .mif...\e[0m'
mrconvert $RAW_DWI DWI_${sub}.mif -fslgrad $BVEC $BVAL

#denoising [Veraart2016]
echo -e '\e[33mPreprocessing S2: Denoising...\e[0m'
dwidenoise DWI_${sub}.mif DWI_den_${sub}.mif

#DWI preprocessing using topup [Andersson2003] (estimating and correcting susceptibility induced distortions) and eddy [Andersson2016] (correcting eddy currents and movements in diffusion data) tools from FSL
echo -e '\e[33mPreprocessing S3: DWI preprocessing...\e[0m'
dwifslpreproc DWI_den_${sub}.mif DWI_den_preproc_${sub}.mif -rpe_none -pe_dir $PhaseEncoding -eddy_options "--slm=linear --data_is_shelled"

#To improve the brain mask estimation. Perform B1 field inhomogeneity correction for a DWI volume series. First, estimating the bias field from DWI b=0 and then applying the field to correct all DW volumes.
# if not N4BiasFieldCorrection get it from pCloudDrive/phD/SOFTWARE/ANTs
echo -e '\e[33mPreprocessing S4: B1 inhomogeneity correction...\e[0m'
dwibiascorrect ants DWI_den_preproc_${sub}.mif DWI_den_preproc_unbiased_${sub}.mif

#Generate a whole brain mask that includes both brain tissue and CSF
echo -e '\e[33mPreprocessing S5: Generate a whole brain mask that includes both brain tissue and CSF...\e[0m'
dwi2mask DWI_den_preproc_unbiased_${sub}.mif mask_${sub}.mif

###############################################################
################# BASIS FUNCTION FOR EACH TISSUE TYPE #########
###############################################################

#Performing various types of response function estimation. The "dhollander" function [Dhollander2016] is best used for multi-shell acquisitions; it will estimate different basis functions for each tissue type (wm, gm and csf).
echo -e '\e[33mResponse function estimation...\e[0m'
dwi2response dhollander DWI_den_preproc_unbiased_${sub}.mif wm_${sub}.txt gm_${sub}.txt csf_${sub}.txt -voxels voxels_${sub}.mif

#Estimation of fiber orientation distributions from diffussion data using spherical deconvolution [Jeurissen2014]
echo -e '\e[33mEstimation of fiber orientation distributions...\e[0m'
dwi2fod msmt_csd DWI_den_preproc_unbiased_${sub}.mif -mask mask_${sub}.mif wm_${sub}.txt wmfod_${sub}.mif gm_${sub}.txt gmfod_${sub}.mif csf_${sub}.txt csffod_${sub}.mif

#Creates an image of the fiber orientation densities overlaid onto the estimated tissues (Blue=WM; Green=GM; Red=CSF)
echo -e '\e[33mCreating fiber orientation densities (FODs) image...\e[0m'
mrconvert -coord 3 0 wmfod_${sub}.mif - | mrcat csffod_${sub}.mif gmfod_${sub}.mif - vf_${sub}.mif

#Normalize the FODs to enable comparison between subjects
echo -e '\e[33mNormalizing FODs for subject comparison...\e[0m'
#mtnormalise wmfod_${sub}.mif wmfod_norm_${sub}.mif gmfod_${sub}.mif gmfod_norm_${sub}.mif csffod_${sub}.mif csffod_norm_${sub}.mif -mask mask_${sub}.mif
# If -nan -nan -nan:
mtnormalise wmfod_${sub}.mif wmfod_norm_${sub}.mif csffod_${sub}.mif csffod_norm_${sub}.mif -mask mask_${sub}.mif

################################################################
################ GM/WM BOUNDARY ################################
################################################################


#Anatomical .mif convertion
echo -e '\e[33mAnatomical .mif convertion...\e[0m'
#mrconvert $ANAT ANAT_${sub}.mif

#Extraction of all five tissue categories (1=GM; 2=Subcortical GM; 3=WM; 4=CSF; 5=Pathological tissue)
echo -e '\e[33mExtraction of all five tissue categories (1=GM; 2=Subcortical GM; 3=WM; 4=CSF; 5=Pathological tissue)\e[0m'
#5ttgen fsl ANAT_${sub}.mif 5tt_nocoreg_${sub}.mif

#Extraction of the b=0 volumes and take the average of b0 volumes (which have the best contrast)
echo -e '\e[33mExtraction of the b=0 volumes and take the average of b0 volumes (which have the best contrast)\e[0m'
dwiextract DWI_den_preproc_unbiased_${sub}.mif - -bzero | mrmath - mean mean_b0_processed_${sub}.mif -axis 3

#Convert previous to coregister
mrconvert mean_b0_processed_${sub}.mif mean_b0_processed_${sub}.nii.gz
mrconvert 5tt_nocoreg_${sub}.mif 5tt_nocoreg_${sub}.nii.gz

#Extraction of the first volume of the 5tt dataset because flirt can only use 3D images, not 4D
fslroi 5tt_nocoreg_${sub}.nii.gz 5tt_vol0_${sub}.nii.gz 0 1

#Creation of a transformation matrix for registration between tissue map and b0 images
flirt -in mean_b0_processed_${sub}.nii.gz -ref 5tt_vol0_${sub}.nii.gz -interp nearestneighbour -dof 6 -omat diff2struct_fsl_${sub}.mat


#Change matrix into a format that can be read by MRtrix
transformconvert diff2struct_fsl_${sub}.mat mean_b0_processed_${sub}.nii.gz 5tt_nocoreg_${sub}.nii.gz flirt_import diff2struct_mrtrix_${sub}.txt

#Coregistering
mrtransform 5tt_nocoreg_${sub}.mif -linear diff2struct_mrtrix_${sub}.txt -inverse 5tt_coreg_${sub}.mif

#Creating a seed region along the GM/WM boundary [Smith2012]
5tt2gmwmi 5tt_coreg_${sub}.mif gmwmSeed_coreg_${sub}.mif


########################################################
################ STREAMLINES ###########################
########################################################

#Perform streamlines tractography (5Million)
tckgen -act 5tt_coreg_${sub}.mif -backtrack -seed_gmwmi gmwmSeed_coreg_${sub}.mif -nthreads 32 -maxlength 250 -cutoff 0.06 -select ${NumberOfTracks} wmfod_norm_${sub}.mif tracks_${sub}_5M.tck

#Optimise per-streamline cross-section multipliers to match a whole-brain tractogram to fixel-wise fibre densities
tcksift2 -act 5tt_coreg_${sub}.mif -nthreads 32 tracks_${sub}_5M.tck wmfod_norm_${sub}.mif sift_${sub}_5M.txt


########################################################
######################	REFERENCES	####################
########################################################

#[Veraart2016] J. Veraart, E. Fieremans, and D.S. Novikov. Diffusion MRI noise mapping using random matrix theory. Magn. Res. Med. 76(5) (2016), pp. 1582–1593.
#[Andersson2016] Jesper L. R. Andersson and Stamatios N. Sotiropoulos. An integrated approach to correction for off-resonance effects and subject movement in diffusion MR imaging. NeuroImage, 125:1063-1078, 2016.
#[Andersson2003] J.L.R. Andersson, S. Skare, J. Ashburner. How to correct susceptibility distortions in spin-echo echo-planar images: application to diffusion tensor imaging. NeuroImage, 20(2):870-888, 2003.
#[Dhollander2016] Dhollander, T.; Raffelt, D. & Connelly, A. Unsupervised 3-tissue response function estimation from single-shell or multi-shell diffusion MR data without a co-registered T1 image. ISMRM Workshop on Breaking the Barriers of Diffusion MRI, 2016, 5
#[Jeurissen2014] Jeurissen, B; Tournier, J-D; Dhollander, T; Connelly, A & Sijbers, J. Multi-tissue constrained spherical deconvolution for improved analysis of multi-shell diffusion MRI data. NeuroImage, 2014, 103, 411-426
#[Smith2012] Smith, R. E.; Tournier, J.-D.; Calamante, F. & Connelly, A. Anatomically-constrained tractography:Improved diffusion MRI streamlines tractography through effective use of anatomical information. NeuroImage, 2012, 62, 1924-1938
