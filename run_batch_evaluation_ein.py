from run_batch_evaluation import run_batch_evaluation
import os

if __name__ == "__main__":
    img_dir = "data/N2DH-GOWT1/img"
    gt_dir  = "data/N2DH-GOWT1/gt"

    df = run_batch_evaluation(img_dir, gt_dir)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/dice_scores.csv", index=False)
    print(df)
