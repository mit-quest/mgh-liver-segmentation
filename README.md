# Liver Segmentation

`liver_segmentation` segments images of liver biopsies preserved in formalin to identify areas of fat and calculate the fat estimate.

## Install conda

Before you run the script, you will need to have `conda` installed. The instructions to install `conda` for your operating system are at this website: [https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).

## Usage

To run the script, open the command line. Navigate to the directory with both the bash script and Python script. Then, run the following line in the command line.

```bash
source ./liver_segmentation.sh
```

Once the necessary dependencies are installed, you will be prompted for the following information.

```
Enter the relative path to the directory with liver biopsy images:

Enter the relative path to the directory to save output:

Enter the relative path to the CSV file with pathologist fat estimates:

Enter the magnification of the images (e.g. 20x):
```

## Example

Below is an example directory containing the bash script `liver_segmentation.sh`, Python script `liver_segmentation.py`, directory with liver biopsy images, directory to save output, and CSV file with pathologist estimates.

```
.
├── formalin_images (directory with liver biopsy images)
│   ├── HF-1 (subdirectory names are liver names)
│   │   ├── HF-1 formalin - 3hr - 20x - a - 1.tiff (image name)
│   │   ├── HF-1 formalin - 3hr - 20x - b - 1.tiff
│   │   └── ...
│   ├── HF-2
│   ├── HF-3
│   └── ...
├── formalin_output (directory to save output)
├── liver_segmentation.py
├── liver_segmentation.sh
└── pathologist_estimates_macro.csv
```

* The directory with liver biopsy images should contain subdirectories named as only liver names. The names of images in the subdirectories should contain the liver name and the magnification of the image. The magnification should be written in the format of `20x`.

* The directory to save output should be empty before you run the script. This directory will save the segmented liver images and the fat estimates.

* Each row of the CSV file should begin with the liver name and followed by any pathologist estimates like the following.

```
HF-1,25
HF-2,0
HF-3,40
```

* The magnification of the images to use should be written as `20x` and should match the magnification written in the image names.

The following is an example response to the prompts above.

```
Enter the relative path to the directory with liver biopsy images:
formalin_images

Enter the relative path to the directory to save output:
formalin_output

Enter the relative path to the CSV file with pathologist fat estimates:
pathologist_estimates_macro.csv

Enter the magnification of the images (e.g. 20x):
20x
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
