"""Creates a single slurm to process each chunk of the dataset so these can be run in parallel on the compute clusters"""
import os

def get_slurm_script_string(DIR, LAYER):
    script = (
f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=0:25:00
#SBATCH --job-name=Probe{LAYER:02}
#SBATCH --error={DIR}/logs/err-{LAYER:02}-%J.err
#SBATCH --output={DIR}/logs/out-{LAYER:02}-%J.out

module load anaconda/anaconda3

source /apps/anaconda/anaconda3/etc/profile.d/conda.sh

conda activate base

conda activate chess

which python

hostname

pwd

python src/train_probes.py --target_layer {LAYER} --epochs 1 --batch_size 50


""")
    return script

DIR = "slurm/train_probes/"

# Create all subfolders up to and including logs/
if not os.path.exists(DIR+'/logs'):
    os.makedirs(DIR+'/logs')

slurm_files = []
for i in range(12):
    script = get_slurm_script_string(DIR,i)
    
    filename = DIR+f'train-probe-{i:03}.slurm'
    
    slurm_files.append(filename)
    
    with open(filename, 'w') as file:
        file.write(script)
    
