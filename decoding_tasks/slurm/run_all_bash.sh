#!/bin/bash

if [ $# -lt 1 ];
then
	echo "Usage: $0 [task]"
	exit
fi

task=$1
for region in VISp VISl VISal VISrl VISpm VISam
# for region in VISal VISpm
do
OMP_NUM_THREADS=24 python ../different_task.py task3_linear_all_stim \
 	        --use_zscore --nsample_units_list 1000 --nsamples 3 \
		--regions ${region} --blocks block1_1 block1_2 block2_2 \
	        --logfile="task3_linear_blockall_stimall_${region}"
done

