import glob
import numpy as np
from PIL import Image
import os
from typing import List, Dict, Any
import sys
import argparse

import helpers


def parse_args(args: List[str]) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Resize images to given dimensions")
    parser.add_argument(
        "--images_directory",
        type=str,
        required=True,
        help="Path to image directory",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Path to output directory to save resized images",
    )
    parser.add_argument(
        "--new_width",
        type=int,
        default=480,
        help="Desired width for resized images",
    )
    parser.add_argument(
        "--new_height",
        type=int,
        default=640,
        help="Desired height for resized images",
    )
    args = vars(parser.parse_args())
    return args

def resize_images(images_directory, new_width, new_height, output_directory):
    """
    Resize images before segmenting and calculating fat estimates for images
    """
    # create output dirs
    image_dir_fps = [fp for fp in glob.glob(images_directory + "/**/**", recursive=True) 
                    if os.path.isdir(fp)]

    image_file_fps = [fp for fp in glob.glob(images_directory + "/**/**", recursive=True) 
                    if os.path.isfile(fp)]

    for fp in image_dir_fps:
        new_dir_fp = fp.replace(images_directory, output_directory) 
        if not os.path.exists(new_dir_fp):
            os.mkdir(new_dir_fp)

    # resize images
    for fp in image_file_fps:
        image = Image.open(fp)
        if image.size != (new_width, new_height):
            new_image = image.resize((new_width, new_height))
        else:
            new_image = image
        save_path = fp.replace(images_directory, output_directory)
        new_image.save(save_path, "tiff")

def main(**args: Dict[str, Any]) -> None:
    images_directory, output_directory, new_width, new_height = args.values()
    resize_images(images_directory, new_width, new_height, output_directory)


if __name__ == "__main__":
    kwargs = parse_args(sys.argv[1:])
    main(**kwargs)
