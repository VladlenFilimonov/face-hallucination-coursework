import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool


# Function to retrieve size (width and height) of a PNG image
def get_png_size(file_path):
    try:
        img = cv2.imread(file_path)
        height, width, _ = img.shape
        return (os.path.basename(file_path), (width, height))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# Function to retrieve sizes of all PNG images in a directory
def get_png_sizes(directory):
    # Get list of PNG file paths
    png_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.png')]

    # Use multiprocessing Pool to parallelize the processing of PNG files
    with Pool() as pool:
        results = pool.map(get_png_size, png_files)

    # Filter out None results and create a dictionary of file names and sizes
    sizes_dict = {filename: size for filename, size in results if size is not None}
    return sizes_dict


# Function to plot distribution of image counts by width and height
def plot_image_sizes(sizes_dict):
    # Extract widths and heights from the sizes dictionary
    widths = np.array([size[0] for size in sizes_dict.values()])
    heights = np.array([size[1] for size in sizes_dict.values()])

    # Calculate histograms for widths and heights
    width_hist, width_bins = np.histogram(widths, bins=20)
    height_hist, height_bins = np.histogram(heights, bins=20)

    # Calculate bin centers for plotting
    width_centers = (width_bins[:-1] + width_bins[1:]) / 2
    height_centers = (height_bins[:-1] + height_bins[1:]) / 2

    # Plotting
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(width_centers, width_hist, width=np.diff(width_bins), align='center')
    plt.xlabel('Width')
    plt.ylabel('Count')
    plt.title('Distribution of Image Counts by Width')

    plt.subplot(1, 2, 2)
    plt.bar(height_centers, height_hist, width=np.diff(height_bins), align='center')
    plt.xlabel('Height')
    plt.ylabel('Count')
    plt.title('Distribution of Image Counts by Height')

    plt.tight_layout()
    plt.show()


# Function to plot 16 random image examples
def plot_random_images(sizes_dict, directory):
    # Randomly select 16 image filenames
    random_files = np.random.choice(list(sizes_dict.keys()), size=16, replace=False)

    # Plotting
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, filename in enumerate(random_files):
        img = cv2.imread(os.path.join(directory, filename))
        axes[i // 4, i % 4].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i // 4, i % 4].axis('off')
        axes[i // 4, i % 4].set_title(filename)

    plt.tight_layout()
    plt.show()