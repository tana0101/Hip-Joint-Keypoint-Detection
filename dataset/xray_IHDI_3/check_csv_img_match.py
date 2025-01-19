import os

# Define the paths to the directories containing CSV and image files
csv_dir = "annotations"
img_dir = "images"

# Supported image extensions
image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

# Get the list of file names without extensions for CSV and images
csv_files = set(os.path.splitext(f)[0] for f in os.listdir(csv_dir) if f.endswith(".csv"))
img_files = set(os.path.splitext(f)[0] for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in image_extensions)

# Find unmatched files
csv_without_img = csv_files - img_files
img_without_csv = img_files - csv_files

# Report the results
if csv_without_img:
    print("CSV files without matching images:")
    for file in csv_without_img:
        print(file)
else:
    print("All CSV files have matching images.")

if img_without_csv:
    print("Image files without matching CSV files:")
    for file in img_without_csv:
        print(file)
else:
    print("All image files have matching CSV files.")
