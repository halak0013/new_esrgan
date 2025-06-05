import sys

from src.train import train
from src.test import test
from src.utils.config import VERSION

if __name__ == "__main__":
    mode = sys.argv[1] if sys.argv[1] == "test" else "train"
    is_new = sys.argv[2] if sys.argv[2] == "False" else "True"

    if is_new == "True":
        with open("data/version", "w") as f:
            f.write(str(int(VERSION) + 1))

    print(f"Running in {mode}")
    if mode == "train":
        train()
        print("Training model...")
    elif mode == "test":
        test()
    else:
        print("Invalid mode. Use 'train' or 'test'.")

""" 
python main.py train UNet
python main.py test UNet

python main.py train ResUNet
python main.py test ResUNet
"""
