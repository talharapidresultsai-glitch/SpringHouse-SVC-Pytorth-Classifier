#!/bin/sh

#image=$1
image="local/classifier"

mkdir -p test_dir/model
mkdir -p test_dir/output

rm test_dir/model/*
rm test_dir/output/*

sudo docker run -v $(pwd)/test_dir:/opt/ml -e LOG_URL='http://10.0.0.151:6000/log' --rm --gpus all ${image} train



#sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi