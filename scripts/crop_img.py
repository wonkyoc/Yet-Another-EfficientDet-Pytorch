"""
A script for cropping images
"""

import os
from tqdm import tqdm
import numpy as np
from PIL import Image


input_dir = "datasets/argoverse/input_sid_17"
output_dir = "datasets/argoverse/input_sid_17_3x3_filtered"

def crop_image(img):
    left = 640
    top = 400
    right = 1280
    bottom = 800
    cropped = img.crop((left, top, right, bottom))
    return cropped

def remove_image(img):
    img = img.convert("RGB")
    img_arr = np.array(img)
    img_arr[400:800, 640:1280] = (0, 0, 0)
    return Image.fromarray(img_arr)

def main():
    images = []
    for (a, b, c) in os.walk(input_dir):
        images.extend(c)

    for img_name in tqdm(images):
        img = Image.open(f"{input_dir}/{img_name}")
        #cropped = crop_image(img)
        #cropped.save(f"{output_dir}/{img_name}")

        removed = remove_image(img)
        removed.save(f"{output_dir}/{img_name}")


if __name__ == "__main__":
    main()

