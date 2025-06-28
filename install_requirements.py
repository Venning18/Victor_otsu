# install_requirements.py

import subprocess

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
    subprocess.check_call(["pip", "install", pkg])
