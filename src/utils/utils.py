import random
import torch
import os
import src.utils.config as cfg
import numpy as np
from PIL import Image
from torchvision.utils import save_image
import src.utils.config as cfg


def gradient_penalty(discriminator, real, fake):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(cfg.DEVICE)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = discriminator(interpolated_images)

    # Interpolated görüntüye göre discriminator'ın gradyanını alır
    # asıl modelin parametreleri güncellenmez
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    # Her görüntüyü vektörleştirir
    gradient = gradient.flatten(start_dim=1) 
    # Her örneğin gradyan uzunluğunu hesaplar
    gradient_norm = gradient.norm(2, dim=1)
    # Normun 1’den sapmasına ceza verir
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=cfg.DEVICE)
    # model.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def plot_examples(low_res_folder, gen, ex):
    files = os.listdir(low_res_folder)

    gen.eval()
    for file in files:
        image = Image.open(cfg.TEST_IMAGE_DIR + "/" + file)
        with torch.no_grad():
            upscaled_img = gen(
                cfg.test_transform(image=np.asarray(image))["image"]
                .unsqueeze(0)
                .to(cfg.DEVICE)
            )
        p = f"{cfg.SAVED_DIR}/{cfg.EXPERIMENT}/{ex}"
        if not os.path.exists(p):
            os.makedirs(p)
        save_image(upscaled_img, f"{p}/{file}")
    gen.train()


def seed_torch(seed=41):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
