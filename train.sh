#!/bin/bash
#SBATCH -A research
#SBATCH -J "StyTR"
#SBATCH -c 20
#SBATCH --mem-per-cpu=2G
#SBATCH -G 3
#SBATCH -w "gnode050"
#SBATCH -o "train_out.txt"
#SBATCH --time="4-00:00:00"
#SBATCH --mail-type=END

# Entrypoint
cd ~/StyTR-reimplementation
source ~/.bashrc
conda activate mr
python train.py --content_dir /scratch/sarthak/train2017 --style_dir /scratch/sarthak/wikiart --batch_size 4 --num_gpus 3 --save_dir /scratch/sarthak/train
# Exit
echo "Time at exit: " `date`
