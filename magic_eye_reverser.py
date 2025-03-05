#!/usr/bin/env python

import os
import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.filters.rank as rank
# import skimage.morphology as morphology


def lerp(n, a, b, x, y, clamp=True):
    m = (n-a)/(b-a)
    m = m * (y-x) + x
    if clamp:
        # ensure within x, y range.
        m = np.clip(m, x, y) 
    return m


def display_img(image, window_name="Image", blocking=False):
    cv2.imshow(window_name, image)
    # Wait for 1 ms to allow window to refresh; continue immediately
    key = cv2.waitKey(1)
    if blocking:
        try:
            # Wait indefinitely until a ESC or 'q' key.
            while key not in [27, ord('q')]:
                key = cv2.waitKey(100)
        except KeyboardInterrupt:
            pass
        cv2.destroyWindow(window_name)


def right_trim_image(image, N=0):
    # Excludes the last N pixel columns.
    return image[:, :-N]


def right_shift_image(image, shift):
    shifted = right_trim_image(image, N=shift)
    # Add zero pixels back to the left side.
    return np.pad(shifted, ((0, 0), (shift, 0)), mode='constant', constant_values=0)


def diff_images(image1, image2):
    return cv2.absdiff(image1, image2)


def normalize_image(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def generate_shift_diffs(image, min_shift, max_shift):
    diffs = []
    for shift in range(min_shift, max_shift):
        shifted_image = right_shift_image(image, shift)
        diffs.append(diff_images(image, shifted_image))
    return diffs


def process_diffs(diff_images, patch_radius=3):
    print('Processing diff images...')
    for index, diff_image in enumerate(diff_images):
        progress = (index+1) / len(diff_images) * 100
        sys.stdout.write(f'\r    Progress: {progress:.2f}%')
        sys.stdout.flush()

        #  Use OpenCV's box filter to sum values in the patch
        kernel_size = (patch_radius*2+1, patch_radius*2+1)
        diff_image = cv2.boxFilter(diff_image, -1, kernel_size, normalize=True)
        diff_images[index] = normalize_image(diff_image)
    print()


def select_diff(diff_images, window_name="Shift Selector", trackbar_name="Shift"):
    """
    Allows the user to select shift values interactively using OpenCV trackbar.
    Uses a mouse click instead of a button to confirm selection.

    Parameters:
    - diff_images: List of diff images.

    Returns:
    - index: Selected shift index.
    """

    # Initial values
    current_index = 0
    confirmed = False  # Flag to exit UI when confirmed

    # Define button area (bottom-right of the image)
    button_x, button_y, button_w, button_h = 20, 20, 150, 50  # Button position

    # Callback function for the trackbar
    def update_index(pos):
        """ Updates the displayed diff image based on min shift selection. """
        nonlocal current_index
        current_index = pos

        selected_diff = diff_images[current_index]
        display_image = cv2.cvtColor(selected_diff, cv2.COLOR_GRAY2BGR)

        # Draw confirmation button
        cv2.rectangle(display_image, (button_x, button_y), (button_x + button_w, button_y + button_h), (0, 0, 255), -1)
        cv2.putText(display_image, "Confirm", (button_x + 20, button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window_name, display_image)

    def mouse_callback(event, x, y, flags, param):
        """ Detects mouse click on the confirm button and exits UI when clicked. """
        nonlocal confirmed
        if event == cv2.EVENT_LBUTTONDOWN:
            if button_x <= x <= button_x + button_w and button_y <= y <= button_y + button_h:
                print("Selection confirmed!")
                confirmed = True

    # Create the OpenCV window
    cv2.namedWindow(window_name)

    # Create trackbars for selecting min and max shift values
    cv2.createTrackbar(trackbar_name, window_name, 0, len(diff_images)-1, update_index)

    # Set the mouse callback to detect button clicks
    cv2.setMouseCallback(window_name, mouse_callback)

    # Initialize display
    update_index(0)

    # Wait for user confirmation
    while not confirmed:
        cv2.waitKey(10)

    cv2.destroyWindow(window_name)

    return current_index


def reverse_magic_eye(input_filename, output_filename):
    # Load the Magic Eye image
    magic_eye_img = cv2.imread(input_filename, cv2.IMREAD_GRAYSCALE)
    if magic_eye_img is None:
        raise FileNotFoundError(input_filename)

    height, width = magic_eye_img.shape

    min_possible_shift = width // 20
    max_possible_shift = width // 2

    mean_diff_threshold = 25

    time.sleep(0.5)

    diff_images = generate_shift_diffs(magic_eye_img, min_possible_shift, max_possible_shift)

    process_diffs(diff_images)

    min_index = select_diff(diff_images, f"{input_filename} - Min Shift Selector", "Min Shift")
    diff_images = diff_images[min_index:]

    max_index = select_diff(diff_images, f"{input_filename} - Max Shift Selector", "Max Shift")
    diff_images = diff_images[:max_index+1]

    print('Building Output Depth Map...')

    min_shift = min_possible_shift + min_index
    field_width = width - min_shift

    # Initialize depth map
    depth_map = np.zeros((height, field_width), dtype=np.uint8)

    window_name = "Depth Map"
    for y in range(height):
        progress = (y+1) / height * 100
        sys.stdout.write(f'\r    Progress: {progress:.2f}%')
        sys.stdout.flush()
        
        for x in range(field_width):
            min_diff_val = float("inf")
            best_index = None

            for index, diff_image in enumerate(diff_images):
                shifted_x = x + min_shift + index
                if shifted_x >= width:
                    break # This pixel was shifted out of bounds.

                diff_val = diff_image[y, shifted_x]

                if diff_val < mean_diff_threshold and diff_val < min_diff_val:
                    min_diff_val = diff_val
                    best_index = index

            if best_index is not None:
                depth_map[y, x] = lerp(len(diff_images) - best_index, 0, len(diff_images), 0, 255)

        if y % 4 == 0:
            display_img(depth_map, window_name=window_name)
    print()
    cv2.destroyWindow(window_name)

    # Normalize depth map for visualization.
    depth_map = normalize_image(depth_map)

    # Save the depth map
    cv2.imwrite(output_filename, depth_map)

    print(f"\nDepth map saved as {output_filename}")


def process_magic_eye_dir(dir_path):
    magic_eye_path = os.path.join(dir_path, "magic_eye.jpg")
    if not os.path.exists(magic_eye_path):
        magic_eye_path = os.path.join(dir_path, "magic_eye.png")
    depth_map_path = os.path.join(dir_path, "depth_map.png")
    
    print(f'\nProcessing {dir_path}...\n')
    reverse_magic_eye(magic_eye_path, depth_map_path)


def process_magic_eye_images(base_directory):
    """
    Processes Magic Eye images by listing all directories in the given base directory
    and generating file paths for "magic_eye.jpg" and "depth_map.png" in each directory.

    Args:
        base_directory (str): The base directory containing subdirectories with Magic Eye images.
    
    Returns:
        list: A list of tuples, each containing the paths to "magic_eye.jpg" and "depth_map.png".
    """
    if not os.path.isdir(base_directory):
        raise ValueError(f"Invalid directory: {base_directory}")

    image_paths = []
    
    # List all subdirectories in the base directory
    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)
        if not os.path.isdir(subdir_path):
            continue
        process_magic_eye_dir(subdir_path)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        process_magic_eye_dir(sys.argv[1])
    else:
        process_magic_eye_images('./puzzles')
