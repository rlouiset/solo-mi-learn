"""Download MedMNIST (128x128) on a node with internet access (Jean Zay login node).

Jean Zay compute nodes are offline, so run this once from the login node:

    module load miniforge/24.9.0 && conda activate py39
    python scripts/utils/download_medmnist.py

Each dataset is a single .npz (e.g. pathmnist_128.npz) holding the train/val/test splits.
"""

import argparse
import os

import medmnist

DEFAULT_ROOT = "/lustre/fswork/projects/rech/haj/uik24xv/local/medmnist"
DEFAULT_DATASETS = ["PathMNIST", "BloodMNIST", "DermaMNIST"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    for name in args.datasets:
        DatasetClass = vars(medmnist)[name]
        # downloads (or md5-verifies) the npz once, then checks every split opens offline
        DatasetClass(root=args.root, split="train", download=True, size=args.size)
        for split in ["train", "val", "test"]:
            ds = DatasetClass(root=args.root, split=split, download=False, size=args.size)
            print(f"{name:12s} {split:5s} {len(ds):6d} images")
    print(f"Done. Files in {args.root}: {sorted(os.listdir(args.root))}")


if __name__ == "__main__":
    main()
