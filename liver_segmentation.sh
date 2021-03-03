#!/bin/bash

# To run this bash script, type `source ./liver_segmentation.sh` in the command line

# Create environment with pyimagej, proceed with yes
conda create -y -n liver_segmentation -c conda-forge pyimagej openjdk=8

# Activate environment
conda activate liver_segmentation

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Prompt user for images_directory, output_directory, pathologist_estimates, and magnification
echo -e "Enter the relative path to the directory with liver biopsy images: "
read images

echo -e "Enter the relative path to the directory to save output: "
read output

echo -e "Enter the relative path to the CSV file with pathologist fat estimates: "
read estimates

echo -e "Enter the magnification of the images (e.g. 20x): "
read mag

# Run formalin segmentation Python script
python liver_segmentation.py --images_directory $images --output_directory $output --pathologist_estimates $estimates --magnification $mag
