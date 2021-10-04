#!/bin/bash

trials=15
tag="study-3"
shared_args="--wandb --wandb-tag $tag --wandb-entity mlpi --dataset-folder ./datasets/lesions"

experiments=(
    "--sampler TrainingSampler --data-augmentation full"
    "--sampler TrainingSampler --data-augmentation crop-flip"
    "--sampler TrainingSampler --data-augmentation none"
    "--sampler RepeatFactorTrainingSampler --data-augmentation full"
    "--sampler RepeatFactorTrainingSampler --data-augmentation crop-flip"
    "--sampler RepeatFactorTrainingSampler --data-augmentation none"
)

source ./env/bin/activate
for trial in $(seq $trials); do
    for experiment in "${experiments[@]}"; do
        echo "Running $shared_args $experiment (trial: $trial)"
        python train.py $shared_args $experiment
    done
done
