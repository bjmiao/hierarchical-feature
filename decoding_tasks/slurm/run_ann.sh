#!/bin/bash

if [ $# -lt 1 ];
then
	echo "Usage: $0 [task]"
	exit
fi

task=$1
do
do OMP_NUM_THREADS=15 python different_task_ann.py decoding_ann_task1 --model_name VGG11 --keys result_layer${i} --blocks block1_1 --noise 5	
OMP_NUM_THREADS=24 python ../different_task.py task3_linear_all_stim \
 	        --use_zscore --nsample_units_list 1000 --nsamples 3 \
		--regions ${region} --blocks block1_1 block1_2 block2_2 \
	        --logfile="task3_linear_blockall_stimall_${region}"
done

