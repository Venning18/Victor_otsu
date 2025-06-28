from src.load_image_pair import load_image_and_gt
from process_image import process_all_methods

img_path = "data/N2DH-GOWT1/img/t01.tif"
gt_path  = "data/N2DH-GOWT1/gt/man_seg01.tif"

image, gt = load_image_and_gt(img_path, gt_path)

results = process_all_methods(image)

for name, mask in results.items():
    print(f"{name}: {mask.shape}, Positiv: {mask.sum()}")
