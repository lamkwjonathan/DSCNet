#!/usr/bin/env bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python -u DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label TEST --lr 0.000175
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python -u DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label TEST --lr 0.0002
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python -u DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label TEST --lr 0.0002 --poly_decay_power 2.0
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python -u DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label TEST --lr 0.001
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python -u DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label TEST --lr 0.001 --poly_decay_power 2.0
