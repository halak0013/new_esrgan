import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2

import src.utils.config as cfg


class NatureFromFolder(Dataset):
    def __init__(self, type_="train", train_ratio=0.7, val_ratio=0.1):
        super(NatureFromFolder, self).__init__()
        self.data = []
        self.root_dir = cfg.DATASET_PATH
        self.class_names = os.listdir(self.root_dir)
        print(self.class_names[:10])

        for index, name in enumerate(self.class_names):
            files = os.path.join(self.root_dir, name)
            self.data.append(files)
        print(self.data[:10])
        # Shuffle data
        np.random.shuffle(self.data)

        # Split data into train, val, and test
        total = len(self.data)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        if type_ == "train":
            self.data = self.data[:train_end]
        elif type_ == "val":
            self.data = self.data[train_end:val_end]
        elif type_ == "test":
            self.data = self.data[val_end:]
        else:
            raise ValueError("type_ must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file = self.data[index]

        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        both_transform = cfg.both_transforms(image=image)["image"]
        low_res = cfg.lowres_transform(image=both_transform)["image"]
        high_res = cfg.highres_transform(image=both_transform)["image"]
        return low_res, high_res


def test():
    dataset_train = NatureFromFolder(root_dir="data/", type_="train")
    dataset_val = NatureFromFolder(root_dir="data/", type_="val")
    dataset_test = NatureFromFolder(root_dir="data/", type_="test")

    print(f"Train dataset size: {len(dataset_train)}")
    print(f"Validation dataset size: {len(dataset_val)}")
    print(f"Test dataset size: {len(dataset_test)}")

    loader = DataLoader(dataset_train, batch_size=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)


if __name__ == "__main__":
    test()