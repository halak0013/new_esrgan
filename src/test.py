import torch
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


from src.esrgan.model import Generator
from src.utils import config
from src.utils.utils import load_checkpoint, plot_examples
from src.utils.utils import seed_torch
from src.utils.data_loaders import get_loaders

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def evaluate_and_plot(test_loader, gen, num_samples=3):
    psnr_list = []
    ssim_list = []
    selected_samples = []

    loop = tqdm(test_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for idx, (low_res, high_res) in enumerate(loop):
            low_res = low_res.to(config.DEVICE)
            high_res = high_res.to(config.DEVICE)

            # Generator ile yüksek çözünürlüklü görüntü üret
            fake_high_res = gen(low_res)

            # PSNR ve SSIM hesapla
            for i in range(low_res.shape[0]):
                sr_img = fake_high_res[i].cpu().numpy().transpose(1, 2, 0)
                hr_img = high_res[i].cpu().numpy().transpose(1, 2, 0)
                lr_img = low_res[i].cpu().numpy().transpose(1, 2, 0)

                # Normalize görüntüler (0-1 aralığına)
                sr_img = np.clip(sr_img, 0, 1)
                hr_img = np.clip(hr_img, 0, 1)
                lr_img = np.clip(lr_img, 0, 1)

                # PSNR ve SSIM hesapla
                psnr = peak_signal_noise_ratio(hr_img, sr_img, data_range=1.0)
                ssim = structural_similarity(
                    hr_img, sr_img, channel_axis=2, data_range=1.0
                )
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                # İlk num_samples kadar örneği seç
                if len(selected_samples) < num_samples:
                    selected_samples.append((lr_img, hr_img, sr_img, psnr, ssim))

    # Ortalama PSNR ve SSIM değerlerini yazdır
    print(f"Average PSNR: {np.mean(psnr_list):.4f}")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")

    # Seçilen örnekleri çiz
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    fig.suptitle(
        f"Super-Resolution Evaluation Samples\nPSNR: {np.mean(psnr_list):.4f} SSIM{np.mean(ssim_list):.4f}",
        fontsize=16,
    )
    for i, (lr, hr, sr, psnr, ssim) in enumerate(selected_samples):
        axes[i, 0].imshow(lr)
        axes[i, 0].set_title(f"LR")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(sr)
        axes[i, 1].set_title(f"SR  PSNR: {psnr:.2f} SSIM: {ssim:.4f}")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(hr)
        axes[i, 2].set_title(f"HR")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{config.SAVE_PATH}/evaluation_samples.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def test():
    seed_torch(config.SEED)
    _, _, test_loader = get_loaders()
    gen = Generator(in_channels=3).to(config.DEVICE)
    gen.eval()
    load_checkpoint(config.CHECKPOINT_GEN, gen)
    print("Testing the model...")
    evaluate_and_plot(test_loader, gen, num_samples=12)
