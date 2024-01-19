"""Creates a single slurm to process each chunk of the dataset so these can be run in parallel on the compute clusters"""
import os


def get_python_line(LAYER, EPOCHS, PROBE_TYPE, BATCH_SIZE):
    return f"""python src/train_probes.py --target_layer {LAYER} --epochs {EPOCHS} --probe_type {PROBE_TYPE} --batch_size {BATCH_SIZE} --jobid "$SLURM_JOBID"

"""


def get_preamble(SLURM_JOB_NAME, DIR, SCRIPT_BATCH_NAME, LAYER, PROBE_TYPE):
    return f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH --time=7:00:00
#SBATCH --job-name={SLURM_JOB_NAME}
#SBATCH --error={LOG_DIR}/err-{LAYER:02}-{PROBE_TYPE}-%J.err
#SBATCH --output={LOG_DIR}/out-{LAYER:02}-{PROBE_TYPE}-%J.out

module load anaconda/anaconda3
source /apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate base
conda activate chess
which python
hostname
pwd
nvidia-smi


"""


### -----------------------
# SCRIPT STARTS HERE
### -----------------------

SCRIPT_BATCH_NAME = "18Jan_colors"
DIR = f"slurm/{SCRIPT_BATCH_NAME}"
SCRIPT_DIR = DIR + "/scripts"
LOG_DIR = DIR + "/logs"
BATCH_SIZE = 50

# Create all subfolders up to and including logs/
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

if not os.path.exists(SCRIPT_DIR):
    os.makedirs(SCRIPT_DIR)

slurm_files = []

for probe_type in [
    "COLOR_0",
    "COLOR_1",
    "COLOR_2",
    "COLOR_3",
    "PIECE_ANY_0",
    "PIECE_ANY_1",
    "PIECE_BY_COLOR_0",
    "PIECE_BY_COLOR_1",
    "MY_CONTROLLED_0",
    "MY_CONTROLLED_1",
]:
    filename = SCRIPT_DIR + f"/train-{probe_type}-all-layers-{BATCH_SIZE}.slurm"

    preamble = get_preamble(
        SCRIPT_BATCH_NAME, probe_type, EPOCHS := 3, BATCH_SIZE, PROBE_TYPE=probe_type
    )

    script = preamble

    for layer in range(11, -1, -1):
        script += get_python_line(layer, EPOCHS, probe_type, BATCH_SIZE)

        # slurm_files.append(filename)

    with open(filename, "w") as file:
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

# S----BATCH --constraint=v100
