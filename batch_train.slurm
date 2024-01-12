#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=1:00:00
#SBATCH --job-name=ProbeTrain
#SBATCH --error=slurm/err-ProbeTrain-%J.err
#SBATCH --output=slurm/out-ProbeTrain-%J.out

module load anaconda/anaconda3

source /apps/anaconda/anaconda3/etc/profile.d/conda.sh

conda activate base

conda activate chess

which python

hostname

pwd

python train_probes.py --target_layer $target_layer --epochs 1 --batch_size 100
