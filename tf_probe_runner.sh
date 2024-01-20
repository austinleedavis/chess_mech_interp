#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --constraint=h100
#SBATCH --error=slurm/19Jan/logs/%J.err
#SBATCH --output=slurm/19Jan/logs/%J.out
#SBATCH --job-name=p0v1x5

module load anaconda/anaconda3
source /apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate base
conda activate chess
which python
hostname
pwd
nvidia-smi

for i in {0..1}
do
   python tf_probing.py --layer $i --lr 0.001 --batch_size 100 --num_epochs 5 --notes "Using mine/theirs/not states"
done


#   -h, --help            show this help message and exit
#   --output_dir OUTPUT_DIR
#                         Output directory (default: linear_probes/)
#   --probe_name PROBE_NAME
#                         Probe file name (default: )
#   --dataset DATASET     Dataset path (default: chess_data/lichess_train.pkl)
#   --layer LAYER         Layer number (default: 6)
#   --batch_size BATCH_SIZE
#                         Batch size (default: 30)
#   --lr LR               Learning rate (default: 0.0001)
#   --wd WD               Weight decay (default: 0.01)
#   --betas BETAS BETAS   Betas for optimizers (default: (0.9, 0.99))
#   --pos_start POS_START
#                         Position start (default: 5)
#   --num_epochs NUM_EPOCHS
#                         Number of epochs (default: 2)
#   --num_games NUM_GAMES
#                         Number of games (default: 100000)
#   --resume              Flag to resume training (default: False)
#   --log_frequency LOG_FREQUENCY
#                         Number of batches between logs (default: 100)
#   --checkpoint_frequency CHECKPOINT_FREQUENCY
#                         Number of batches between logs (default: 100)
