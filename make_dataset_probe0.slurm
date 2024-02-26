#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --job-name=prb0dat


module load anaconda/anaconda3
source /apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate base
conda activate chess
which python
hostname
pwd
nvidia-smi

curl -d "Dataset Started" https://ntfy.sh/awesomesauceisinteresting
python src/probe_data_generation.py --probe_type COLOR_0 --target_layer 0 --batch_size 50 --tags datagen
curl -d "Dataset complete" https://ntfy.sh/awesomesauceisinteresting
