#!/bin/bash

for layer in {0..11}
do
    sbatch --export=ALL,target_layer=$layer batch_train.slurm
done

