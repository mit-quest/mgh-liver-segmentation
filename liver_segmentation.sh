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
fi

if [ -z "$estimates" ]
then
	estimates='False'
fi

# Run Python script
python3 "scripts/runner.py" --images_directory $images --output_directory $output --pathologist_estimates $estimates --magnification $mag --preservation $pres
