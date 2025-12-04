import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class AugmentedKeypointDataset(Dataset):
    def __init__(self, original_dataset, max_angle=0, max_translate_x=0, max_translate_y=0):
        """
        Args:
            original_dataset: The original dataset to augment.
            max_angle: Maximum rotation angle (degrees) for augmentation.
            max_translate_x: Maximum horizontal translation (pixels).
            max_translate_y: Maximum vertical translation (pixels).
        """
        self.original_dataset = original_dataset
        self.max_angle = max_angle
        self.max_translate_x = max_translate_x
        self.max_translate_y = max_translate_y

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Retrieve the original data
        image, keypoints, crop_size = self.original_dataset[idx]
        crop_width, crop_height = crop_size

        # Generate random augmentation parameters
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        translate_x = np.random.uniform(-self.max_translate_x, self.max_translate_x)
        translate_y = np.random.uniform(-self.max_translate_y, self.max_translate_y)

        # Apply rotation augmentation
        rotated_image = TF.rotate(image, angle)

        # Calculate rotation matrix for keypoints
        angle_rad = np.deg2rad(-angle)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        # Center of rotation (image center)
        img_width, img_height = image.shape[1], image.shape[2]
        center_x, center_y = img_width / 2, img_height / 2

        # Apply rotation to keypoints
        keypoints_rotated = []
        for i in range(0, len(keypoints), 2):
            x = keypoints[i]
            y = keypoints[i + 1]
            x_new = cos_theta * (x - center_x) - sin_theta * (y - center_y) + center_x
            y_new = sin_theta * (x - center_x) + cos_theta * (y - center_y) + center_y
            keypoints_rotated.extend([x_new, y_new])

        # Apply translation to keypoints
        keypoints_translated = [
            coord + translate_x if i % 2 == 0 else coord + translate_y
            for i, coord in enumerate(keypoints_rotated)
        ]

        # Apply translation to the image
        translated_image = TF.affine(
            rotated_image, angle=0, translate=(translate_x, translate_y), scale=1, shear=0
        )

        return translated_image, torch.tensor(keypoints_translated, dtype=torch.float32), (crop_width, crop_height)
