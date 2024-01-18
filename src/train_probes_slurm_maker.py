"""Creates a single slurm to process each chunk of the dataset so these can be run in parallel on the compute clusters"""
import os

def get_slurm_script_string(
    DIR,
    LAYER,
    EPOCHS,
    BATCH_SIZE,
    PROBE_TYPE,
):
    script = (
f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --constraint=h100
#SBATCH --time=1:30:00
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
nvidia-smi

python src/train_probes.py --target_layer {LAYER} --epochs {EPOCHS} --probe_type {PROBE_TYPE} --batch_size {BATCH_SIZE} --jobid "$SLURM_JOBID"


""")
    return script

DIR = "slurm/train_probes_types/"

# Create all subfolders up to and including logs/
if not os.path.exists(DIR+'/logs'):
    os.makedirs(DIR+'/logs')

slurm_files = []
for probe_type in [
    "PIECE_ANY_0",
    "PIECE_ANY_1",
    "PIECE_BY_COLOR_0",
    "PIECE_BY_COLOR_1",
    "MY_CONTROLLED_0",
    "MY_CONTROLLED_1",
    ]:
    for layer in range(12):
        script = get_slurm_script_string(
            DIR,
            layer,
            EPOCHS:=3,
            BATCH_SIZE=200,
            PROBE_TYPE=probe_type)
            
        filename = DIR+f'train-probe-{layer:03}-type-{probe_type}.slurm'
        
        slurm_files.append(filename)
        
        with open(filename, 'w') as file:
            file.write(script)
    


"""
options:
  -h, --help            show this help message and exit
  --dataset_prefix {lichess_}
                        Prefix of the dataset (Default: 'lichess_')
  --dataset_dir DATASET_DIR
                        Directory where data is stored. (Default 'chess_data/')
  --target_layer TARGET_LAYER
                        Optional. Target layer number (Default: -1)
  --split {train,test}  Optional. Dataset split. Choose from ['train', 'test']) (Default: TRAIN)
  --probe_type {COLOR_0,COLOR_1,COLOR_2,COLOR_3,COLOR_FLIPPING_0,COLOR_FLIPPING_1,PIECE_ANY_0,PIECE_ANY_1,PIECE_BY_COLOR_0,PIECE_BY_COLOR_1,MY_CONTROLLED_0,MY_CONTROLLED_1}
                        Optional. Type of probe. Choose from ['color_0', 'color_1', 'color_2',
                        'color_3', 'color_flipping_0', 'color_flipping_1', 'piece_any_0',
                        'piece_any_1', 'piece_by_color_0', 'piece_by_color_1', 'my_controlled_0',
                        'my_controlled_1']) (Default: COLOR)
  --epochs EPOCHS       Optional. Number of epochs to train (Default: 1)
  --batch_size BATCH_SIZE
                        Optional. batch sizes (Default: 50)
  --slice SLICE         Enter a range in the format start:stop:step
  --jobid JOBID         Job ID if called from SLURM
  """
  

"PIECE_ANY_0",
"PIECE_ANY_1",
"PIECE_BY_COLOR_0",
"PIECE_BY_COLOR_1",
"MY_CONTROLLED_0",
"MY_CONTROLLED_1",
