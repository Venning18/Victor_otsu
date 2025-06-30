from src.load_image_pair import load_image_and_gt
from process_image import process_all_methods
from method.evaluate_segmentation.evaluate_segmentation import evaluate_segmentations

image, gt_mask = load_image_and_gt(
    "data/N2DH-GOWT1/img/t01.tif",
    "data/N2DH-GOWT1/gt/man_seg01.tif"
)

results = process_all_methods(image)

df_scores = evaluate_segmentations(gt_mask, results)
print(df_scores)
