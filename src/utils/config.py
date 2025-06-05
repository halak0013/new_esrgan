import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = False
SAVE_MODEL = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
LAMBDA_GP = 10
NUM_WORKERS = 12
NUM_EPOCHS = 5

BATCH_SIZE = 32
HIGH_RES = 128
LOW_RES = HIGH_RES // 4
IMG_CHANNELS = 3

with open("data/version", "r") as f:
    VERSION = f.read().strip()

NAME = "normal"
TEST_IMAGE_DIR = "test_images"
SAVED_DIR = "saved"
LOG_DIR = "logs"
DATA_SET_NAME_DIR = "esrgan_dataset"
EXPERIMENT = VERSION + "_" + NAME
def extension(mode_name: str) -> str:
    return mode_name + EXPERIMENT + VERSION + ".pth"
SAVE_PATH = f"{LOG_DIR}/{DATA_SET_NAME_DIR}/{EXPERIMENT}"
DATA = "data"
CACHE_DIR = DATA + "/cache"
CSV_PATH = DATA + "/results.csv"

CHECKPOINT_GEN = SAVE_PATH + "/" + extension("gen_")
CHECKPOINT_DISC = SAVE_PATH + "/" + extension("disc_")

SEED = 41


highres_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

lowres_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)

both_transforms = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
)

test_transform = A.Compose(
    [
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)
