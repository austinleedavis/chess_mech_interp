#!/bin/bash

# for i in {0..11}
# do
#    padded_index=$(printf "%03d" $i)
#    sbatch slurm/train_probes/train-probe-${padded_index}.slurm
# done

#!/bin/bash

for file in slurm/train_probes_types/train-probe-*; do
   sbatch $file
done
