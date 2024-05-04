#!/bin/bash

mkdir -p no_cape

total_iters=16
completed_iters=0

while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --content)
        content_img="$2"
        shift
        shift
        ;;
        --style)
        style_img="$2"
        shift
        shift
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

if [ -z "$content_img" ] || [ -z "$style_img" ]; then
    echo "Content and style image paths are required."
    exit 1
fi

for iter in {10000..160000..10000}
do
    python test.py --content "$content_img" --style "$style_img" --output no_cape/$iter --vgg ./vgg_normalised.pth --decoder_path /scratch/sarthak/train_nocape/decoder_iter_$iter.pth --Trans_path /scratch/sarthak/train_nocape/transformer_iter_$iter.pth --embedding_path /scratch/sarthak/train_nocape/embedding_iter_$iter.pth --cape False
    completed_iters=$((completed_iters + 1))
    progress=$((completed_iters * 100 / total_iters))
    printf "\rProgress: [%-20s] %d%%" $(printf "#%.0s" {1..$((progress / 5))}) $progress
    sleep 1  # Adjust this sleep time as per your preference
done

echo "Completed!"
