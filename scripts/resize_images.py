import numpy as np
from PIL import Image
import os
from typing import List, Dict, Any
import sys
import argparse


def resize_images(images_directory, new_width, new_height, output_directory):
    """
    Resize images before segmenting and calculating fat estimates for images
    """
    image_names = os.listdir(images_directory)
    for image_name in image_names:
        image_path = os.path.join(images_directory, image_name)
        image = Image.open(image_path)

        if image.size != (new_width, new_height):
            new_image = image.resize((new_width, new_height))
            save_path = os.path.join(output_directory, image_name)
            new_image.save(save_path, "tiff")


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


def main(**args: Dict[str, Any]) -> None:
    images_directory, output_directory, new_width, new_height = args.values()
    resize_images(images_directory, new_width, new_height, output_directory)


if __name__ == "__main__":
    kwargs = parse_args(sys.argv[1:])
    main(**kwargs)
