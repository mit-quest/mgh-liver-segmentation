import ast
from datetime import timedelta
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any
import time

import os
import sys
sys.setrecursionlimit(10000000)

import imagej

import common
import helpers
from segment_image import segment_image
from estimate_steatosis import estimate_steatosis


def _process_image(images_directory, output_directory, liver_name, image_name, is_frozen, magnification): 
    start_time = time.monotonic()

    ij = imagej.init('net.imagej:imagej+net.imagej:imagej-legacy', mode='headless') 

    mag_vars = common.get_mag_vars(magnification)

    print("Beginning segmentation of " + image_name)
    seg_start_time = time.monotonic()
    segment_image(images_directory, output_directory, liver_name, 
                                  image_name, is_frozen, mag_vars, ij)
    print(image_name + " segmentation complete! Time taken: " + str(timedelta(seconds=time.monotonic() - seg_start_time)))
    print("Estimating steatosis of " + image_name)
    est_start_time = time.monotonic()
    estimate_steatosis(images_directory, output_directory, liver_name,
                                   image_name, is_frozen, mag_vars)
    print(image_name + " steatosis estimation complete! Time taken: " + str(timedelta(seconds=time.monotonic() - est_start_time)))

    print(image_name + " complete! Time taken: " + str(timedelta(seconds=time.monotonic() - start_time)))


def main(**args: Dict[str, Any]) -> None:
    images_directory, output_directory, magnification, preservation, pathologist_estimates = args.values()
    is_frozen = True if preservation == 'frozen' else False
    liver_folders = os.listdir(images_directory)

    # Create output folder if it does not already exist
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    for liver_name in liver_folders:

        print("\n\nProcessing " + liver_name)
        start_time = time.monotonic()

        liver_folder_path = os.path.join(images_directory, liver_name)
        if os.path.isdir(liver_folder_path):

            # Create liver output folder if it does not already exist
            liver_output_folder = os.path.join(output_directory, liver_name)
            if not os.path.exists(liver_output_folder):
                os.mkdir(liver_output_folder)

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

            # Init parallel processing for segmentation & fat estimation
            n_procs = cpu_count() if len(liver_images) >= cpu_count() else len(liver_images)
            pool = Pool(processes=n_procs)
            input_list = [[images_directory, output_directory, liver_name, image_name, is_frozen, magnification] for image_name in liver_images]
            pool.starmap(_process_image, input_list) 
            pool.close()

            print(liver_name + " complete! Time taken: " + str(timedelta(seconds=time.monotonic() - start_time)))

            # Create slides for each liver
            if ast.literal_eval(pathologist_estimates):
                powerpoint_save_path = os.path.join(liver_output_folder,
                                                    f'{liver_name}_slides.pptx')
                helpers.make_powerpoint(images_directory, output_directory,
                                pathologist_estimates, liver_name, powerpoint_save_path)
        else:
            print(liver_folder_path + " is not a directory. Skipping.")            

if __name__ == "__main__":
    kwargs = helpers.parse_args(sys.argv[1:])
    main(**kwargs)
    sys.exit(0) # force exit else JVM hangs (known imageJ issue)