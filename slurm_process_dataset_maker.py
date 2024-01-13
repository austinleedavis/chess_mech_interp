"""Creates a single slurm to process each chunk of the dataset so these can be run in parallel on the compute clusters"""
import os

def get_slurm_script_string(CHUNK_MIN,CHUNK_MAX):
    script = (
f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=15:00
#SBATCH --mem=50G
#SBATCH --job-name=uci{CHUNK_MIN}-{CHUNK_MAX}
#SBATCH --error=slurm/opgn2uci-err-{CHUNK_MIN:03}-{CHUNK_MAX:03}-%J.err
#SBATCH --output=slurm/pgn2uci-out-{CHUNK_MIN:03}-{CHUNK_MAX:03}-%J.out

module load anaconda/anaconda3

source /apps/anaconda/anaconda3/etc/profile.d/conda.sh 2> /dev/null

conda activate base 

conda activate chess

which python

hostname

pwd

python src/prepare_dataset.py --chunk_min {CHUNK_MIN} --chunk_max {CHUNK_MAX}
""")
    return script

DIR = "slurm/data_preprocessing/"

if not os.path.exists(DIR):
    os.makedirs(DIR)

slurm_files = []
for i in range(165):
    script = get_slurm_script_string(i,i+1)
    
    filename = DIR+f'process_lichess_chunk_{i:03}.slurm'
    
    slurm_files.append(filename)
    
    with open(filename, 'w') as file:
        file.write(script)
    
