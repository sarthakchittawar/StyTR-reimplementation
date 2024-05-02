#!/bin/bash
#SBATCH -A research
#SBATCH -J "StyTR"
#SBATCH -c 15
#SBATCH --mem-per-cpu=2G
#SBATCH -G 4
#SBATCH -w "gnode054"
#SBATCH -o "train_out.txt"
#SBATCH --time="4-00:00:00"
#SBATCH --mail-type=END

# Entrypoint
cd ~/StyTR-reimplementation
source ~/.bashrc
conda activate mr
python train.py --content_dir /scratch/sarthak/train2017 --style_dir /scratch/sarthak/wikiart_train --batch_size 8 --num_gpus 4 --save_dir /scratch/sarthak/train4 --activation_func gelu
# Exit
echo "Time at exit: " `date`
