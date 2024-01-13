# Dataset Preparation

Run `src/lichess_dataset_maker.py`. This file will download the lichess dataset, unzip the file, break it into 100k chunks, and process the chunks by converting from PGN to UCI notation, map moves to tokens, and map tokens to move indices. Finally, the chunked data will be saved in feather format. If post-processing is needed on only some chunks, the `--chunk_min` and `--chunk_max` options can be given to process a subset of the data chunks. 

> Note: If you have access to a compute cluster, you can use the `chess_data/slurm_process_dataset_maker.py` to generate a slurm script to process each chunk file, and then all the slurm scripts can be run using `chess_data/slurm_process_dataset_runner.sh`

# Model Training



# Training Probes
The `src/train_probes.py` script handles probe training. Probes are simple models trained on the residual stream of the transformer. The residual stream is captured by hooks in the transformer layers using the Transformer Lens library. The script provides a CLI suitable for deployment to a cluster (see the `linear_probes` folder for example slurm scripts.)