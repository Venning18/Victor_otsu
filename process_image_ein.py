from src.load_image_pair import load_image_and_gt
from process_image import process_all_methods

img_path = "data/N2DL-HeLa/img/t13.tif"
gt_path  = "data/N2DL-HeLa/gt/man_seg13.tif"

image, gt = load_image_and_gt(img_path, gt_path)

results = process_all_methods(image)

for name, mask in results.items():
    print(f"{name}: {mask.shape}, Positiv: {mask.sum()}")
