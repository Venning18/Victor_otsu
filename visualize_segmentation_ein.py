from src.load_image_pair import load_image_and_gt
from process_image import process_all_methods
from visualize_segmentation import visualize_segmentations

image, gt_mask = load_image_and_gt(
    "data/N2DH-GOWT1/img/t01.tif",
    "data/N2DH-GOWT1/gt/man_seg01.tif"
)

results = process_all_methods(image)

visualize_segmentations(image, gt_mask, results)
