bucket = gcs_client.get_bucket("liver-processing-kate")
training_liver_files = []
holdout_liver_files = []
all_liver_files = []


def get_image_files_from_folders(liver_folders, include_path=False):
    liver_files = []
    for liver_folder in liver_folders:
        for blob in bucket.list_blobs(prefix=f"raw-data/{liver_folder}"):
            blob_name_split = blob.name.split("/")
            file = blob_name_split[2]
            if blob_name_split[2] != "" and "20x" in file and f"{liver_folder} " in file:
                if include_path and blob.name not in liver_files:
                    liver_files.append(blob.name)
                else:
                    if blob_name_split[2] not in liver_files:
                        liver_files.append(blob_name_split[2])
    return liver_files


# All liver folders and images
# all_liver_folders = ["FS-1", "HF-1", "HF-2", "HF-3", "HF-4", "HF-5", "HF-6", "HF-7", "HF-8", "HF-9", "HF-10", "HF-11", "HF-12",
#     "HF-13", "HF-14", "HF-15", "HF-16", "HF-17", "HF-18", "HF-19", "HF-20", "HF-21", "HF-22", "HF-23", "TP-2"]
# all_liver_files = get_image_files_from_folders(all_liver_folders, include_path=True)

# Use HF-3 liver images
all_liver_folders=["HF-3"]
all_liver_files = get_image_files_from_folders(all_liver_folders, include_path=True)

print("Number of training images: ", len(training_liver_files), len(set(training_liver_files)))
print("Number of holdout images: ", len(holdout_liver_files), len(set(holdout_liver_files)))
print("Total number of images: ", len(all_liver_files), len(set(all_liver_files)))
