import numpy as np
from utils.simcc import decode_simcc_to_xy

def calculate_nme(preds, targets, points_count, img_size):
    """
    NME in crop space.
    img_size: (crop_width, crop_height)
    """
    preds = preds.reshape(points_count, 2)  # Reshape to (12, 2)
    targets = targets.reshape(points_count, 2)  # Reshape to (12, 2)

    # Calculate the Euclidean distance for each keypoint
    pixel_distances = np.linalg.norm(preds - targets, axis=1)  # Shape: (12,)

    # Calculate the diagonal of the image
    img_diag = np.sqrt(img_size[0]**2 + img_size[1]**2)  # Scalar

    # Normalize distances by the image diagonal
    norm_distances = pixel_distances / img_diag  # Shape: (12,)

    # Return the mean of the normalized distances
    return np.mean(norm_distances)

def calculate_pixel_error(preds, targets, points_count, img_size, input_size):
    """
    Pixel error in crop pixel units.
    img_size: (crop_width, crop_height)
    """
    preds = preds.reshape(points_count, 2)  # Reshape to (12, 2)
    targets = targets.reshape(points_count, 2)  # Reshape to (12, 2)

    # Unpack the original image dimensions
    original_width, original_height = img_size

    # Calculate the scaling factors for width and height
    scale_x = original_width / input_size  # Assuming input_size is the model input size
    scale_y = original_height / input_size

    # Scale the predictions and targets
    preds_scaled = preds * np.array([scale_x, scale_y])  # Shape: (12, 2)
    targets_scaled = targets * np.array([scale_x, scale_y])  # Shape: (12, 2)

    # Calculate the Euclidean distance for each keypoint
    pixel_distances = np.linalg.norm(preds_scaled - targets_scaled, axis=1)  # Shape: (12,)

    # Return the mean of the pixel distances
    return np.mean(pixel_distances)