#!/bin/bash
#SBATCH --mail-user=SLRUMReport@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=clusternorm_pacs_1head
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=3-00:00
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

cd pacs
tar -xzf cartoon.tgz
tar -xzf art_painting.tgz
tar -xzf photo.tgz
tar -xzf sketch.tgz
unzip -q image_list.zip

date +"%T"
echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR/iic_clusternorm/clusternorm

(export CUDA_VISIBLE_DEVICES=0 && nohup python -m code.scripts.cluster.clusternorm_sobel_twohead --model_ind 640  --arch ClusterNormNet5gTwoHead --mode IID --dataset pacs_art --dataset_root $SLURM_TMPDIR/iic_clusternorm/iic_dataset/pacs --gt_k 7 --output_k_A 15 --output_k_B 7 --lamb 1.0 --lr 0.0001  --num_epochs 2000 --batch_sz 660 --num_dataloaders 3 --num_sub_heads 1 --crop_orig --rand_crop_sz 20 --input_sz 32 --head_A_first --head_B_epochs 2 --out_root $SLURM_TMPDIR/cluster/cluster_norm_pacs_art_1head_result > ./cluster_norm_pacs_art_1head_result_out.txt
cp -r $SLURM_TMPDIR/cluster/cluster_norm_pacs_art_1head_result ~/scratch/iic_clusternorm/
cp ./cluster_norm_pacs_art_1head_result_out.txt ~/scratch/iic_clusternorm/cluster_norm_pacs_art_1head_result/
echo "-----------------------------------<End of run the art>---------------------------------") &

(export CUDA_VISIBLE_DEVICES=1 && nohup python -m code.scripts.cluster.clusternorm_sobel_twohead --model_ind 640  --arch ClusterNormNet5gTwoHead --mode IID --dataset pacs_cartoon --dataset_root $SLURM_TMPDIR/iic_clusternorm/iic_dataset/pacs --gt_k 7 --output_k_A 15 --output_k_B 7 --lamb 1.0 --lr 0.0001  --num_epochs 2000 --batch_sz 660 --num_dataloaders 3 --num_sub_heads 1 --crop_orig --rand_crop_sz 20 --input_sz 32 --head_A_first --head_B_epochs 2 --out_root $SLURM_TMPDIR/cluster/cluster_norm_pacs_cartoon_1head_result > ./cluster_norm_pacs_cartoon_1head_result_out.txt
cp -r $SLURM_TMPDIR/cluster/cluster_norm_pacs_cartoon_1head_result ~/scratch/iic_clusternorm/
cp ./cluster_norm_pacs_cartoon_1head_result_out.txt ~/scratch/iic_clusternorm/cluster_norm_pacs_cartoon_1head_result/
echo "-----------------------------------<End of run the cartoon>---------------------------------") &

(export CUDA_VISIBLE_DEVICES=2 && nohup python -m code.scripts.cluster.clusternorm_sobel_twohead --model_ind 640  --arch ClusterNormNet5gTwoHead --mode IID --dataset pacs_photo --dataset_root $SLURM_TMPDIR/iic_clusternorm/iic_dataset/pacs --gt_k 7 --output_k_A 15 --output_k_B 7 --lamb 1.0 --lr 0.0001  --num_epochs 2000 --batch_sz 660 --num_dataloaders 3 --num_sub_heads 1 --crop_orig --rand_crop_sz 20 --input_sz 32 --head_A_first --head_B_epochs 2 --out_root $SLURM_TMPDIR/cluster/cluster_norm_pacs_photo_1head_result > ./cluster_norm_pacs_photo_1head_result_out.txt
cp -r $SLURM_TMPDIR/cluster/cluster_norm_pacs_photo_1head_result ~/scratch/iic_clusternorm/
cp ./cluster_norm_pacs_photo_1head_result_out.txt ~/scratch/iic_clusternorm/cluster_norm_pacs_photo_1head_result/
echo "-----------------------------------<End of run the photo>---------------------------------") &

(export CUDA_VISIBLE_DEVICES=3 && nohup python -m code.scripts.cluster.clusternorm_sobel_twohead --model_ind 640  --arch ClusterNormNet5gTwoHead --mode IID --dataset pacs_sketch --dataset_root $SLURM_TMPDIR/iic_clusternorm/iic_dataset/pacs --gt_k 7 --output_k_A 15 --output_k_B 7 --lamb 1.0 --lr 0.0001  --num_epochs 2000 --batch_sz 660 --num_dataloaders 3 --num_sub_heads 1 --crop_orig --rand_crop_sz 20 --input_sz 32 --head_A_first --head_B_epochs 2 --out_root $SLURM_TMPDIR/cluster/cluster_norm_pacs_sketch_1head_result > ./cluster_norm_pacs_sketch_1head_result_out.txt
cp -r $SLURM_TMPDIR/cluster/cluster_norm_pacs_sketch_1head_result ~/scratch/iic_clusternorm/
cp ./cluster_norm_pacs_sketch_1head_result_out.txt ~/scratch/iic_clusternorm/cluster_norm_pacs_sketch_1head_result/
echo "-----------------------------------<End of run the sketch>---------------------------------") &


wait
echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
