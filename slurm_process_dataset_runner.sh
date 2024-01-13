#!/bin/bash

for i in {0..164}
do
   padded_index=$(printf "%03d" $i)
   sbatch slurm/data_preprocessing/process_lichess_chunk_${padded_index}.slurm
done
