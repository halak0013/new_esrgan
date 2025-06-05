import torch
from torch import optim

from src.esrgan.model import Generator
from src.utils import config
from src.utils.utils import load_checkpoint, plot_examples
from src.utils.utils import seed_torch

def test():
    seed_torch(config.SEED)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.9))
    load_checkpoint(
        config.CHECKPOINT_GEN,
        gen,
        opt_gen,
        config.LEARNING_RATE,
    )
    plot_examples(config.TEST_IMAGE_DIR, gen)
