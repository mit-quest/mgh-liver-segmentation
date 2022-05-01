from datetime import timedelta
from typing import List, Dict, Any
import time

import os
import sys
sys.setrecursionlimit(10000000)

import imagej

import helpers
from segment_liver import segment_liver
from estimate_steatosis import estimate_steatosis


def main(**args: Dict[str, Any]) -> None:
    images_directory, output_directory, magnification, preservation, pathologist_estimates = args.values()
    is_frozen = True if preservation == 'frozen' else False
    ij = imagej.init('net.imagej:imagej+net.imagej:imagej-legacy')
    liver_folders = os.listdir(images_directory)

    # Create output folder if it does not already exist
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    for liver_name in liver_folders:

        print("\nProcessing " + liver_name)
        # Create liver output folder if it does not already exist
        liver_output_folder = os.path.join(output_directory, liver_name)
        if not os.path.exists(liver_output_folder):
            os.mkdir(liver_output_folder)

        liver_folder_path = os.path.join(images_directory, liver_name)
        if os.path.isdir(liver_folder_path) and not liver_folder_path.startswith('.'):
            liver_files = os.listdir(liver_folder_path)
            liver_images = []
            for file in liver_files:
                if file.endswith(".tiff") and magnification in file: # TODO: Use regex
                    liver_images.append(file)

            # Remove existing liver fat estimates
            csv_file_path = os.path.join(liver_output_folder,
                                         f'{liver_name}_fat_estimates.csv')
            if os.path.exists(csv_file_path):
                os.remove(csv_file_path)

            for image_name in liver_images:
                start_time = time.monotonic()                
                print(image_name)
                print("Beginning segmentation...")
                segment_liver(images_directory, output_directory,
                                              liver_name, image_name, is_frozen, ij)
                print("Estimating steatosis...")
                estimate_steatosis(images_directory, output_directory,
                                              liver_name, image_name, is_frozen)
                print(image_name + " complete!")
                print("Image processed in " + str(timedelta(seconds=time.monotonic() - start_time)))

            # Create slides for each liver
            if pathologist_estimates:
                powerpoint_save_path = os.path.join(liver_output_folder,
                                                    f'{liver_name}_slides.pptx')
                helpers.make_powerpoint(images_directory, output_directory,
                                pathologist_estimates, liver_name, powerpoint_save_path)
        else:
            print(liver_folder_path + " is either hidden or not a directory. Skipping.")            


if __name__ == "__main__":
    kwargs = helpers.parse_args(sys.argv[1:])
    main(**kwargs)