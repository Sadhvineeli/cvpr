import cv2
import numpy as np

def adjust_contrast(image, alpha):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def adjust_brightness(image, beta):
    return cv2.convertScaleAbs(image, alpha=1, beta=beta)

def intensity_level_slicing(image, min_thresh, max_thresh, new_value=255):
    mask = (image >= min_thresh) & (image <= max_thresh)
    result = np.zeros_like(image)
    result[mask] = new_value
    return result

def bit_level_slicing(image, bit_plane):
    return (image & (1 << bit_plane)) * 255

image_path = "download.jpeg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Image could not be loaded.")
    exit()

contrast_enhanced = adjust_contrast(image, 1.5)
brightness_enhanced = adjust_brightness(image, 50)
intensity_sliced = intensity_level_slicing(image, 100, 200)
bit_sliced = bit_level_slicing(image, 3)

cv2.imshow("Original Image", image)
cv2.imshow("Contrast Enhanced", contrast_enhanced)
cv2.imshow("Brightness Enhanced", brightness_enhanced)
cv2.imshow("Intensity Level Slicing", intensity_sliced)
cv2.imshow("Bit Level Slicing", bit_sliced)

cv2.waitKey(0)
cv2.destroyAllWindows()
