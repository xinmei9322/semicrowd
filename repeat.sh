#!/usr/bin/env bash

for i in $(seq 1234 1243); do
    echo $i;
    echo $1;
    echo $2;
    CUDA_VISIBLE_DEVICES=$1 python $2 --seed=$i;
done
