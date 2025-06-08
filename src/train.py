import torch
import numpy as np
from torch import nn
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pandas as pd
import src.utils.config as cfg
from src.utils.utils import (
    gradient_penalty,
    load_checkpoint,
    save_checkpoint,
    plot_examples,
)
from src.utils.loss import VGGLoss
from src.esrgan.model import Generator, Discriminator, initialize_weights
from src.utils.utils import seed_torch
from src.utils.data_loaders import get_loaders
from src.utils import scheduler_select

torch.backends.cudnn.benchmark = True
import time


def train_fn(
    loader,
    disc,
    gen,
    opt_gen,
    opt_disc,
    l1,
    vgg_loss,
    g_scaler,
    d_scaler,
    writer,
    tb_step,
    epoch: int = 0,
):
    loop = tqdm(loader, leave=True)
    len_loop2 = len(loop) // 2 - 1
    print(len_loop2)
    loss_critics = []
    loss_gens = []
    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(cfg.DEVICE)
        low_res = low_res.to(cfg.DEVICE)

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        # amp.autocast hesaplamayı f32/16 karmaşık yapıp optimize
        with torch.amp.autocast(cfg.DEVICE):
            fake = gen(low_res)
            critic_real = disc(high_res)
            critic_fake = disc(fake.detach())

            gp = gradient_penalty(disc, high_res, fake)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + cfg.LAMBDA_GP * gp
            )

        opt_disc.zero_grad()
        d_scaler.scale(loss_critic).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        with torch.amp.autocast(cfg.DEVICE):
            l1_loss = 1e-2 * l1(fake, high_res)
            adversarial_loss = 5e-3 * -torch.mean(disc(fake))
            loss_for_vgg = vgg_loss(fake, high_res)
            gen_loss = l1_loss + loss_for_vgg + adversarial_loss

        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        writer.add_scalar("Critic loss", loss_critic.item(), global_step=tb_step)
        writer.add_scalar("Generator loss", gen_loss.item(), global_step=tb_step)
        writer.add_scalar("L1 loss", l1_loss.item(), global_step=tb_step)
        writer.add_scalar("VGG loss", loss_for_vgg.item(), global_step=tb_step)
        writer.add_scalar(
            "Adversarial loss", adversarial_loss.item(), global_step=tb_step
        )
        writer.add_scalar("Gradient penalty", gp.item(), global_step=tb_step)
        writer.add_scalar(
            "Learning rate", opt_gen.param_groups[0]["lr"], global_step=tb_step
        )
        tb_step += 1

        loop.set_postfix(
            gp=gp.item(),
            critic=loss_critic.item(),
            l1=l1_loss.item(),
            vgg=loss_for_vgg.item(),
            adversarial=adversarial_loss.item(),
        )

        loss_critics.append(loss_critic.item())
        loss_gens.append(gen_loss.item())
        del high_res, low_res, fake, critic_real, critic_fake
        del loss_critic, gen_loss, gp
        # torch.cuda.empty_cache()

        if idx % len_loop2 == 0 and idx > 0:
            plot_examples(
                cfg.TEST_IMAGE_DIR, gen, ex=f"epoch_{epoch}_step_{2 - (len_loop2//idx)}"
            )

    return tb_step, np.mean(loss_critics), np.mean(loss_gens)


def validate(gen, dataloader):
    psnr_list = []
    ssim_list = []
    gen.eval()

    loop = tqdm(dataloader, leave=True)
    with torch.no_grad():
        for lr, hr in loop:
            lr = lr.to(cfg.DEVICE)
            hr = hr.to(cfg.DEVICE)

            sr = gen(lr)

            # Batch içindeki tüm örnekler için
            for i in range(lr.shape[0]):
                sr_img = sr[i].cpu().numpy().transpose(1, 2, 0)
                hr_img = hr[i].cpu().numpy().transpose(1, 2, 0)
                # Küçük görüntüler için win_size ayarlanıyor ve channel_axis kullanılıyor
                min_side = min(sr_img.shape[0], sr_img.shape[1])
                win_size = (
                    7
                    if min_side >= 7
                    else min_side if min_side % 2 == 1 else min_side - 1
                )
                psnr = peak_signal_noise_ratio(hr_img, sr_img, data_range=1.0)
                ssim = structural_similarity(
                    hr_img, sr_img, channel_axis=2, data_range=1.0, win_size=win_size
                )
                psnr_list.append(psnr)
                ssim_list.append(ssim)
    return np.mean(psnr_list), np.mean(ssim_list)


def train():
    initial_time = time.time()
    if cfg.DEVICE.startswith("cuda"):
        torch.cuda.init()
        torch.cuda.empty_cache()

    seed_torch(cfg.SEED)

    train_loader, val_loader, _ = get_loaders()

    # Initialize models, optimizers, and loss functions
    gen = Generator(in_channels=3).to(cfg.DEVICE)
    disc = Discriminator(in_channels=3).to(cfg.DEVICE)
    initialize_weights(gen)

    # Model compilation ekleyin (PyTorch 2.0+)
    # balanced_options = {
    #     "triton.cudagraphs": False,  # Memory safety
    #     "epilogue_fusion": True,  # Performance boost
    #     "shape_padding": True,  # Tensor cores
    #     "memory_planning": True,  # Memory efficiency
    # }
    # if hasattr(torch, "compile"):
    #     gen = torch.compile(gen, options=balanced_options)
    #     disc = torch.compile(disc, options=balanced_options))

    opt_gen = optim.Adam(gen.parameters(), lr=cfg.LEARNING_RATE, betas=(0.0, 0.9))
    opt_disc = optim.Adam(disc.parameters(), lr=cfg.LEARNING_RATE, betas=(0.0, 0.9))
    
    scheduler_params = scheduler_select.scheduler_params_dict[cfg.SCHEDULER_TYPE]

    if cfg.SCHEDULER_TYPE == "onecycle":
        steps_per_epoch = len(train_loader)
        gen_scheduler = scheduler_select.get_scheduler(
            opt_gen, cfg.SCHEDULER_TYPE, scheduler_params, steps_per_epoch=steps_per_epoch
        )
        disc_scheduler = scheduler_select.get_scheduler(
            opt_disc, cfg.SCHEDULER_TYPE, scheduler_params, steps_per_epoch=steps_per_epoch
        )
    else:
        gen_scheduler = scheduler_select.get_scheduler(
            opt_gen, cfg.SCHEDULER_TYPE, scheduler_params
        )
        disc_scheduler = scheduler_select.get_scheduler(
            opt_disc, cfg.SCHEDULER_TYPE, scheduler_params
        )

    writer = SummaryWriter(cfg.SAVE_PATH)
    tb_step = 0
    l1 = nn.L1Loss()
    gen.train()
    disc.train()
    vgg_loss = VGGLoss()

    g_scaler = torch.amp.GradScaler(cfg.DEVICE)
    d_scaler = torch.amp.GradScaler(cfg.DEVICE)

    if cfg.LOAD_MODEL:
        load_checkpoint(
            cfg.CHECKPOINT_GEN,
            gen,
            opt_gen,
            cfg.LEARNING_RATE,
        )
        load_checkpoint(
            cfg.CHECKPOINT_DISC,
            disc,
            opt_disc,
            cfg.LEARNING_RATE,
        )

    best_psnr = 0.0
    best_ssim = 0.0

    for epoch in range(cfg.NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{cfg.NUM_EPOCHS}")
        tb_step, loss_critic, loss_gen = train_fn(
            train_loader,
            disc,
            gen,
            opt_gen,
            opt_disc,
            l1,
            vgg_loss,
            g_scaler,
            d_scaler,
            writer,
            tb_step,
            epoch,
        )

        if cfg.SCHEDULER_TYPE == "reduce_on_plateau":
            gen_scheduler.step(loss_gen)
            disc_scheduler.step(loss_critic)
        else:
            gen_scheduler.step()
            disc_scheduler.step()

        val_psnr, val_ssim = validate(gen, val_loader)

        print(f"Validation PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")
        writer.add_scalar("Validation PSNR", val_psnr, global_step=epoch)
        writer.add_scalar("Validation SSIM", val_ssim, global_step=epoch)

        if cfg.SAVE_MODEL:
            save_checkpoint(gen, opt_gen, filename=cfg.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=cfg.CHECKPOINT_DISC)
        if val_ssim > best_ssim or (val_ssim == best_ssim and val_psnr > best_psnr):
            best_psnr = val_psnr
            best_ssim = val_ssim
            save_checkpoint(
                gen, opt_gen, filename=f"{cfg.CHECKPOINT_GEN}_best.pth"
            )
            save_checkpoint(
                disc, opt_disc, filename=f"{cfg.CHECKPOINT_DISC}_best.pth"
            )
    elapsed = time.time() - initial_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)
    df = pd.read_csv(cfg.CSV_PATH)
    new_row = {
        "experiment": cfg.EXPERIMENT,
        "time": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
        "psnr": val_psnr,
        "ssim": val_ssim,
        "loss_critic": loss_critic,
        "loss_generator": loss_gen,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(cfg.CSV_PATH, index=False)
    print(f"Training completed in {hours:02d}:{minutes:02d}:{seconds:02d}")
    writer.close()
