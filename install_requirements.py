import subprocess
import sys

packages = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-image",
    "tqdm"
]

for pkg in packages:
    print(f"📦 Installing {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
