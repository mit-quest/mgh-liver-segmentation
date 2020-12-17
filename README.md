# mgh-liver-segmentation

## Usage
The code in the `segment_with_ImageJ` folder is used to create the mask and
estimate fat for a liver biopsy image with the watershed algorithm in ImageJ.

1. Use `convert_original_to_binary.py` to create a binary image.

2. Use `filter_nonfat_by_size.py` to filter out very small and very large
islands in the binary image that are unlikely to be fat, and save the
filtering result as an inverted binary image.

3. Apply the watershed algorithm in ImageJ by
[running ImageJ in browser](https://ij.imjoy.io/), opening the inverted binary
image, clicking on `Process -> Binary -> Watershed`, and saving the image.

4. Use `create_new_mask.py` to create the mask using the image after applying
the watershed algorithm. The thresholds used to create the mask depends on the
type of liver image.

5. Use `estimate_fat.py` with the original liver image and created mask to
estimate the total and macro fat.

The code not in the `segment_with_ImageJ` folder was used before trying out
the watershed algorithm in ImageJ to help segment clusters of fat globules.
