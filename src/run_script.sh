#!/bin/bash
#SBATCH --account=deep_rc_mem
#SBATCH --partition=dgx_normal_q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=144:00:00
OMP_NUM_THREADS=8 python execute_train_eval_net.py --dataset=FMNIST --epochs=150 --gpu_id=0 --scnn_arch=5 --batch_size=250 --seed=100
