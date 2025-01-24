import numpy as np
import cv2
from scipy.ndimage import label, mean




def find_max_intensity(input_image, x, y, r, search_vec, fg_inc_pixels=1):
    max_intensity = -np.inf
    best_coords = (x, y, r)

    # Create a base mask
    base_mask = np.zeros_like(input_image, dtype=np.uint8)

    for dx, dy in np.ndindex(len(search_vec), len(search_vec)):
        nx = x + search_vec[dx]
        ny = y + search_vec[dy]
        for dr in [-1, 0, 1]:
            nr = r + dr + fg_inc_pixels

            # Create the circle mask
            mask = base_mask.copy()
            cv2.circle(mask, (nx, ny), nr, 1, thickness=-1)

            # Label connected regions and calculate the mean intensity
            fg_label, _ = label(mask)
            fg_mean = mean(input_image, fg_label, index=np.arange(1, fg_label.max() + 1))

            # Check for maximum intensity
            if fg_mean > max_intensity:
                max_intensity = fg_mean
                best_coords = (nx, ny, nr)

    return best_coords, max_intensity


import numpy as np
from scipy import ndimage
import cv2


def optimize_circle_intensity(input_image, x, y, r, search_vec, fg_inc_pixels=1):
    max_intensity = -np.inf
    best_coords = (x, y, r)

    # Generate base mask for circle with radius `r`
    for dx in search_vec:
        for dy in search_vec:
            for dr in [-1, 0, 1]:
                nx = x + dx
                ny = y + dy
                nr = r + dr + fg_inc_pixels

                # Ensure coordinates are within bounds
                if nr < 1 or nx < 0 or ny < 0 or nx >= input_image.shape[1] or ny >= input_image.shape[0]:
                    continue

                # Create a binary mask for the circle
                spot_fg = np.zeros_like(input_image, dtype=np.uint8)
                cv2.circle(spot_fg, (nx, ny), nr, 1, thickness=-1)

                # Label the connected components
                fg_label, _ = ndimage.label(spot_fg)

                # Compute the mean intensity
                fg_mean = ndimage.mean(input_image, labels=fg_label, index=1)

                # Check for maximum intensity
                if fg_mean > max_intensity:
                    max_intensity = fg_mean
                    best_coords = (nx, ny, nr)

    return best_coords, max_intensity


