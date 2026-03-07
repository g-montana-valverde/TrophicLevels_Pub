#!/bin/bash
#SBATCH --job-name=ADNI_Preprocessing
#SBATCH -p high
#SBATCH -N 1
#SBATCH --output=/path/%j.out
#SBATCH --error=/path/%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-$(($(ls $HOME/ADNI_dataset | grep sub | wc -l)-1))


task='rest'

dataset_dir=$HOME/ADNI_dataset

iD=${SLURM_ARRAY_TASK_ID}
mapfile -t BIP_list < <(ls ${dataset_dir} | grep sub)
participant=${BIP_list[iD]}


singularity run --cleanenv \
 -B $dataset_dir:/data \
 -B $HOME/ADNI_preprocessed:/output \
 -B $HOME/license.txt:/license.txt \
 /soft/singularity/fmriprep-21.0.1.simg \
 /data /output participant \
 --participant-label $participant \
 -t $task \
 --work-dir $HOME/ADNI_preprocessed/workdir/ \
 --nthreads 8 \
 --omp-nthreads 8 \
 --mem 32 \
 --fs-license-file $HOME/license.txt \
 --skip-bids-validation


