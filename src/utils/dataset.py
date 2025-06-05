import torch
from torch.utils.data import Dataset
import numpy as np
import src.utils.config as cfg
from datasets import load_dataset


class NatureDataset(Dataset):
    def __init__(self, type_="train"):
        super(NatureDataset, self).__init__()
        ds_dict = load_dataset(
            "mertcobanov/nature-dataset", cache_dir=cfg.CACHE_DIR
        )
        train_ratio=0.14 # 7
        val_ratio=0.02  # 1
        full_ds = ds_dict["train"]  # Tüm veri burada
        total = len(full_ds)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        if type_ == "train":
            self.ds = full_ds.select(range(0, train_end))
        elif type_ == "val":
            self.ds = full_ds.select(range(train_end, val_end))
        elif type_ == "test":
            self.ds = full_ds.select(range(val_end, total))
        else:
            raise ValueError("type_ must be 'train', 'val', or 'test'")
        print(len(self.ds))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        # Use Hugging Face dataset instead of local files
        item = self.ds[index]
        image = item["image"]
        image = np.array(image)
        both_transform = cfg.both_transforms(image=image)["image"]
        low_res = cfg.lowres_transform(image=both_transform)["image"]
        high_res = cfg.highres_transform(image=both_transform)["image"]
        return low_res, high_res


def test():
    dataset = NatureDataset()
    # loader = DataLoader(dataset, batch_size=8)

    # for low_res, high_res in loader:
    #     print(low_res.shape)
    #     print(high_res.shape)


if __name__ == "__main__":
    test()
