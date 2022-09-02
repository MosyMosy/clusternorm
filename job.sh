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

echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code and datasets"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/iic_clusternorm .

echo "Extract the datasets"
cd iic_clusternorm/iic_dataset
tar -xzf cifar-10-python.tar.gz

date +"%T"
echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR/iic_clusternorm/clusternorm

CUDA_VISIBLE_DEVICES=0 python -m code.scripts.cluster.cluster_norm --model_ind 640  --arch ClusterNet5gTwoHead --mode IID --dataset CIFAR10 --dataset_root $SLURM_TMPDIR/iic_clusternorm/iic_dataset --gt_k 10 --output_k_A 70 --output_k_B 10 --lamb 1.0 --lr 0.0001  --num_epochs 2000 --batch_sz 660 --num_dataloaders 3 --num_sub_heads 5 --crop_orig --rand_crop_sz 20 --input_sz 32 --head_A_first --head_B_epochs 2 --out_root $SLURM_TMPDIR/cluster/cluster_norm_result1

echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
echo "--------------------------------------<backup the result>-----------------------------------"
date +"%T"
cp -r $SLURM_TMPDIR/cluster/cluster_norm_result1 ~/scratch/iic_clusternorm/cluster_norm_result1