#!/bin/bash
#SBATCH -A research
#SBATCH -J "StyTR"
#SBATCH -c 20
#SBATCH --mem-per-cpu=2G
#SBATCH -G 3
#SBATCH -w "gnode053"
#SBATCH -o "train_out.txt"
#SBATCH --time="4-00:00:00"
#SBATCH --mail-type=END

# Entrypoint
cd ~/StyTR-reimplementation
source ~/.bashrc
conda activate mr
python train.py --batch_size 6  --content_dir /scratch/sarthak/train2017 --style_dir /scratch/sanika/wikiart

# Exit
echo "Time at exit: " `date`
