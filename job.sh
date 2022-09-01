#!/bin/bash
#SBATCH --mail-user=SLRUMReport@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=clusternorm
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=0-08:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

source ~/envs/iic/bin/activate

echo "change TORCH_HOME environment variable"
cd $SLURM_TMPDIR
cp -r ~/scratch/Pytorch_zoo .
export TORCH_HOME=$SLURM_TMPDIR/Pytorch_zoo

echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/TLlib .

echo "Copying the datasets"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/TLlib_Dataset .

date +"%T"
echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cd TLlib
cd examples/domain_adaptation/image_classification

CUDA_VISIBLE_DEVICES=0 python -m code.scripts.cluster.cluster_sobel_twohead --model_ind 640  --arch ClusterNet5gTwoHead --mode IID --dataset CIFAR10 --dataset_root /scratch/local/ssd/xuji/CIFAR --gt_k 10 --output_k_A 70 --output_k_B 10 --lamb 1.0 --lr 0.0001  --num_epochs 2000 --batch_sz 660 --num_dataloaders 3 --num_sub_heads 5 --crop_orig --rand_crop_sz 20 --input_sz 32 --head_A_first --head_B_epochs 2

echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
echo "--------------------------------------<backup the result>-----------------------------------"
date +"%T"
cp -r $SLURM_TMPDIR/TLlib/logs/nprior_20_101 ~/scratch/TLlib/logs/nprior_20_101