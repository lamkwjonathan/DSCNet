#!/usr/bin/env bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python -u DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label TEST_PyramidOffset_AllDir_Entropy --sample_count 16
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python -u DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label TEST_PyramidOffset_AllDir_Entropy --sample_count 16
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python -u DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label TEST_PyramidOffset_AllDir_Entropy --data_dir Data/SynapseMarmoset/ --sample_count 32
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python -u DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label TEST_PyramidOffset_AllDir_Entropy --data_dir Data/SynapseMarmoset/ --sample_count 32