import os
import cv2
from multiprocessing import Pool


def resize_image(input_data):
    source_file, source_dir, dest_dir, target_width, target_height, postfix = input_data
    try:
        # Add postfix to the filename
        filename, ext = os.path.splitext(source_file)
        filename += postfix
        output_file = filename + ext

        # Check if the destination file already exists
        if os.path.exists(os.path.join(dest_dir, output_file)):
            print(f"Skipping {source_file} as {output_file} already exists")
            return

        # Read the image
        img = cv2.imread(os.path.join(source_dir, source_file))

        # Resize the image to the target dimensions
        resized_img = cv2.resize(img, (target_width, target_height))

        # Save the resized image to the destination directory
        cv2.imwrite(os.path.join(dest_dir, output_file), resized_img)
    except Exception as e:
        print(f"Error processing {source_file}: {e}")


def reduce_resolution_parallel(source_dir, dest_dir, target_width, target_height, postfix):
    print("Resizing process started")

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        print("Creation of new directory")
        os.makedirs(dest_dir)

    # Create a list of input data for each image
    input_data = [(filename, source_dir, dest_dir, target_width, target_height, postfix)
                  for filename in os.listdir(source_dir) if filename.endswith('.png')]

    print(f"Input Data Size: {len(input_data)}")

    # Use multiprocessing Pool to parallelize the resizing process
    with Pool() as pool:
        pool.map(resize_image, input_data)

    print("Resizing process finished")
