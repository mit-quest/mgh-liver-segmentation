#!/bin/bash

# To run this bash script, type `source liver_segmentation.sh` in the command line
# Use `source liver_segmentation.sh use_config` to use values set in config.file

# Create environment with pyimagej, proceed with yes
if ! { conda env list | grep 'liver_segmentation'; } >/dev/null 2>&1; then
	conda create -y -n liver_segmentation -c conda-forge pyimagej openjdk=8
	# Activate environment
	conda activate liver_segmentation
	# Install dependencies from requirements.txt
	pip3 install -r requirements.txt
else
	# Activate environment
	conda activate liver_segmentation
fi

use_config=False
if [[ $# -eq 1 && $1 = "use_config" ]]; then
	use_config=True
fi

if $use_config; then
	source config.file
else
	# Prompt user for images_directory, output_directory, 
	# pathologist_estimates, magnification, and preservation type
	echo -e "Enter the relative path to the directory with liver biopsy images: "
	read images

	echo -e "Enter the relative path to the directory to save output: "
	read output

	echo -e "Enter the magnification of the images ('10x', '20x', or '40x'): "
	read mag

	echo -e "Enter the biopsy preservation type ('frozen' or 'formalin'): "
	read pres

	echo -e "[OPTIONAL] Enter the relative path to a CSV file with pathologist fat estimates for comparison (press 'Enter' to leave blank): "
	read estimates

	echo -e "[OPTIONAL] Enter new dimensions as 'width'x'height' (e.g. '480x640') to resize images (press 'Enter' to leave blank): "
	read dims
fi

if [ -z "$estimates" ]
then
	estimates='False'
fi
if [ "$dims" ]
then
	# Run Python image resizing script
	resized_image_dir=$images'_resized'
	python3 "scripts/resize_images.py" --images_directory $images --output_directory $resized_image_dir --new_width ${dims%x*} --new_height ${dims#*x}
	images=$resized_image_dir
fi

# Run Python segmentation & estimation script
python3 "scripts/runner.py" --images_directory $images --output_directory $output --pathologist_estimates $estimates --magnification $mag --preservation $pres
