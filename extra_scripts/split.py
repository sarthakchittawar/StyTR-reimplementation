import os
import random
import shutil
from tqdm import tqdm

def split_data(source_dir, train_dir, val_dir, test_dir, split_ratio=(0.8, 0.1, 0.1)):
    """
    Split images in a source directory into train, validation, and test sets.
    
    Args:
    - source_dir (str): Path to the source directory containing images.
    - train_dir (str): Path to the directory to store train set images.
    - val_dir (str): Path to the directory to store validation set images.
    - test_dir (str): Path to the directory to store test set images.
    - split_ratio (tuple): A tuple of three floats representing the ratio of train, validation, and test sets respectively.
    """
    assert sum(split_ratio) == 1.0, "Split ratio should sum to 1.0"

    # Create destination directories if they don't exist
    for directory in [train_dir, val_dir, test_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            shutil.rmtree(directory)
            os.makedirs(directory)

    # Get list of image files
    image_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Calculate split points based on split ratios
    num_files = len(image_files)
    train_split = int(num_files * split_ratio[0])
    val_split = int(num_files * (split_ratio[0] + split_ratio[1]))

    # Split the image files into train, validation, and test sets
    train_files = image_files[:train_split]
    val_files = image_files[train_split:val_split]
    test_files = image_files[val_split:]

    # Copy images to respective directories
    with tqdm(total = len(train_files) + len(val_files) + len(test_files)) as pbar:
        for filename in train_files:
            shutil.copy(os.path.join(source_dir, filename), os.path.join(train_dir, filename))
            pbar.update(1)
        for filename in val_files:
            shutil.copy(os.path.join(source_dir, filename), os.path.join(val_dir, filename))
            pbar.update(1)
        for filename in test_files:
            shutil.copy(os.path.join(source_dir, filename), os.path.join(test_dir, filename))
            pbar.update(1)

# Example usage:
source_directory = "/scratch/sarthak/wikiart"
train_directory = "/scratch/sarthak/wikiart_train"
val_directory = "/scratch/sarthak/wikiart_val"
test_directory = "/scratch/sarthak/wikiart_test"

split_data(source_directory, train_directory, val_directory, test_directory)
