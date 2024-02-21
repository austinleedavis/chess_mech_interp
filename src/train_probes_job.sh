#!/bin/bash
#SBATCH --job-name=probe0
#SBATCH --mem=100GB  # Adjust memory requirement to match the parameter
#SBATCH --time=25:45:00  # Adjust time requirement
#SBATCH --gres=gpu:1  # Use --gres for GPU request to match the parameter
#SBATCH --exclusive=user  # Ensure exclusive access to GRES at the user level

module load anaconda/anaconda3
source /apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate base
conda activate chess
which python
hostname
pwd

#accelerate config default

curl -d "Training Started" https://ntfy.sh/awesomesauceisinteresting

python src/train_probes.py --target_layer 0 --epochs 30 --batch_size 32 --probe_type COLOR_0 --tags rerun,newton --reload_filename "linear_probes/saved_probes/probe_L0_B12_slice(0,-1,4)_FN_to_color/probe_L0__B12_slice(0, -1, 4)_FN_to_color.pth"

curl -d "Training Complete" https://ntfy.sh/awesomesauceisinteresting