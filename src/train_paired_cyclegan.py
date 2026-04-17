import os
import sys
import json
import random
import argparse
import psutil, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim_metric
from torchmetrics.functional import peak_signal_noise_ratio as psnr_metric

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import make_dataloader
from src.models  import ResNetGenerator, PatchGANDiscriminator, init_weights, ImageBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Paired CycleGAN")
    parser.add_argument("--anatomy",    type=str, required=True)
    parser.add_argument("--data_root",  type=str, required=True)
    parser.add_argument("--split_dir",  type=str, required=True)
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def load_split(split_dir, anatomy):
    filename = "combined_split.json" if anatomy == "combined" else f"{anatomy.lower()}_split.json"
    with open(os.path.join(split_dir, filename), "r") as f:
        return json.load(f)


def train_step(G, F, D_CT, D_MR, opt_G, opt_D,
               batch, gan_loss, l1_loss, buf_CT, buf_MR, config, device, scaler_G, scaler_D):
    torch.set_grad_enabled(True)

    real_MR = batch["mr"].to(device)
    real_CT = batch["ct"].to(device)
    mask    = batch["mask"].to(device).detach()

    if torch.isnan(real_MR).any() or torch.isnan(real_CT).any():
        print("⚠️  NaN in input — skipping")
        return None

    # ── Generator step ────────────────────────────────────────────────
    opt_G.zero_grad()
    with autocast():
        fake_CT  = G(real_MR); fake_MR  = F(real_CT)
        cycle_MR = F(fake_CT); cycle_CT = G(fake_MR)
        idt_CT   = G(real_CT); idt_MR   = F(real_MR)

        pred_fake_CT  = D_CT(fake_CT)
        pred_fake_MR  = D_MR(fake_MR)

        loss_G = (
            gan_loss(pred_fake_CT, torch.ones_like(pred_fake_CT))
            + gan_loss(pred_fake_MR, torch.ones_like(pred_fake_MR))
            + l1_loss(cycle_MR * mask, real_MR * mask) * config["LAMBDA_CYCLE"]
            + l1_loss(cycle_CT * mask, real_CT * mask) * config["LAMBDA_CYCLE"]
            + l1_loss(idt_CT   * mask, real_CT * mask) * config["LAMBDA_IDENTITY"]
            + l1_loss(idt_MR   * mask, real_MR * mask) * config["LAMBDA_IDENTITY"]
            + l1_loss(fake_CT  * mask, real_CT * mask) * config["LAMBDA_PAIRED"]
            + l1_loss(fake_MR  * mask, real_MR * mask) * config["LAMBDA_PAIRED"]
        )

    if torch.isnan(loss_G) or torch.isinf(loss_G):
        print("⚠️  NaN/Inf G loss — skipping")
        opt_G.zero_grad()
        return None

    scaler_G.scale(loss_G).backward()
    scaler_G.unscale_(opt_G)
    torch.nn.utils.clip_grad_norm_(
        list(G.parameters()) + list(F.parameters()), max_norm=1.0
    )
    scaler_G.step(opt_G)
    scaler_G.update()



    # ── Discriminator step ────────────────────────────────────────────
    opt_D.zero_grad()
    fake_CT_buf = buf_CT.push_and_pop(fake_CT.detach())
    fake_MR_buf = buf_MR.push_and_pop(fake_MR.detach())

    with autocast():
        pred_real_CT  = D_CT(real_CT);  pred_fake_CT2 = D_CT(fake_CT_buf)
        pred_real_MR  = D_MR(real_MR);  pred_fake_MR2 = D_MR(fake_MR_buf)
        loss_D = (
            (gan_loss(pred_real_CT, torch.ones_like(pred_real_CT))
             + gan_loss(pred_fake_CT2, torch.zeros_like(pred_fake_CT2))) * 0.5
          + (gan_loss(pred_real_MR, torch.ones_like(pred_real_MR))
             + gan_loss(pred_fake_MR2, torch.zeros_like(pred_fake_MR2))) * 0.5
        )

    if torch.isnan(loss_D) or torch.isinf(loss_D):
        print("⚠️  NaN/Inf D loss — skipping")
        opt_D.zero_grad()
        return None

    scaler_D.scale(loss_D).backward()
    scaler_D.unscale_(opt_D)
    torch.nn.utils.clip_grad_norm_(
        list(D_CT.parameters()) + list(D_MR.parameters()), max_norm=1.0
    )
    scaler_D.step(opt_D)
    scaler_D.update()

    return {
        "loss_G_total": loss_G.item(),
        "loss_D_total": loss_D.item(),
        "loss_paired":  (l1_loss(fake_CT * mask, real_CT * mask)
                        + l1_loss(fake_MR * mask, real_MR * mask)).item(),
        "loss_cycle":   (l1_loss(cycle_MR * mask, real_MR * mask)
                        + l1_loss(cycle_CT * mask, real_CT * mask)).item(),
    }


def validate(G, F, val_loader, device):
    G.eval(); F.eval()
    mr2ct_ssim, ct2mr_ssim, mr2ct_mae, ct2mr_mae = [], [], [], []

    with torch.no_grad():
        for batch in val_loader:
            real_MR = batch["mr"].to(device); real_CT = batch["ct"].to(device)
            mask    = batch["mask"].to(device)
            fake_CT = G(real_MR); fake_MR = F(real_CT)
            fake_CT_m = fake_CT * mask; real_CT_m = real_CT * mask
            fake_MR_m = fake_MR * mask; real_MR_m = real_MR * mask
            mr2ct_ssim.append(ssim_metric(fake_CT_m, real_CT_m, data_range=2.0).item())
            ct2mr_ssim.append(ssim_metric(fake_MR_m, real_MR_m, data_range=2.0).item())
            mr2ct_mae.append(torch.mean(torch.abs(fake_CT_m - real_CT_m)).item())
            ct2mr_mae.append(torch.mean(torch.abs(fake_MR_m - real_MR_m)).item())

    torch.set_grad_enabled(True)
    G.train(); F.train()

    return {
        "mr2ct_ssim": np.mean(mr2ct_ssim), "ct2mr_ssim": np.mean(ct2mr_ssim),
        "mr2ct_mae":  np.mean(mr2ct_mae),  "ct2mr_mae":  np.mean(ct2mr_mae),
    }


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler_G = GradScaler()
    scaler_D = GradScaler()

    with open(args.config, "r") as f:
        config = json.load(f)
    config["DEVICE"] = device

    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed_all(config["SEED"])
    torch.backends.cudnn.deterministic = True
    torch.set_grad_enabled(True)

    print(f"Device:  {device}")
    print(f"Anatomy: {args.anatomy}")
    print(f"Epochs:  {config['EPOCHS']}")

    ckpt_dir    = os.path.join(args.output_dir, "checkpoints", "paired_cyclegan", args.anatomy)
    results_dir = os.path.join(args.output_dir, "results",     "paired_cyclegan")
    logs_dir    = os.path.join(args.output_dir, "logs",        "paired_cyclegan")
    os.makedirs(ckpt_dir,    exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir,    exist_ok=True)

    split_data   = load_split(args.split_dir, args.anatomy)
    train_loader = make_dataloader(split_data["train"], args.anatomy, args.data_root, "train",
                                   config["BATCH_SIZE"], config["NUM_WORKERS"], config["IMAGE_SIZE"])
    val_loader   = make_dataloader(split_data["val"],   args.anatomy, args.data_root, "val",
                                   config["BATCH_SIZE"], config["NUM_WORKERS"], config["IMAGE_SIZE"])

    G    = ResNetGenerator(in_ch=3, out_ch=3).to(device)
    F    = ResNetGenerator(in_ch=3, out_ch=3).to(device)
    D_CT = PatchGANDiscriminator(in_ch=3).to(device)
    D_MR = PatchGANDiscriminator(in_ch=3).to(device)
    for model in [G, F, D_CT, D_MR]:
        init_weights(model)

    opt_G = optim.Adam(list(G.parameters()) + list(F.parameters()),
                       lr=config["LR"], betas=(config["BETA1"], config["BETA2"]))
    opt_D = optim.Adam(list(D_CT.parameters()) + list(D_MR.parameters()),
                       lr=config["LR"], betas=(config["BETA1"], config["BETA2"]))

    def lambda_rule(epoch):
        if epoch < config["DECAY_EPOCH"]: return 1.0
        return max(0.0, 1.0 - (epoch - config["DECAY_EPOCH"]) / (
            config["EPOCHS"] - config["DECAY_EPOCH"] + 1e-8))

    sched_G  = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lambda_rule)
    sched_D  = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=lambda_rule)
    gan_loss = nn.MSELoss()
    l1_loss  = nn.L1Loss()
    buf_CT   = ImageBuffer(max_size=50)
    buf_MR   = ImageBuffer(max_size=50)

    best_ssim   = -1.0
    start_epoch = 1
    log         = {k: [] for k in ["epoch", "loss_G_total", "loss_D_total",
                                    "loss_paired", "loss_cycle",
                                    "mr2ct_ssim", "ct2mr_ssim",
                                    "mr2ct_mae",  "ct2mr_mae"]}

    existing = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".pth")])
    if existing:
        latest = os.path.join(ckpt_dir, existing[-1])
        ckpt   = torch.load(latest, map_location=device)
        G.load_state_dict(ckpt["G"]); F.load_state_dict(ckpt["F"])
        D_CT.load_state_dict(ckpt["D_CT"]); D_MR.load_state_dict(ckpt["D_MR"])
        opt_G.load_state_dict(ckpt["opt_G"]); opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt["epoch"] + 1
        log_path    = os.path.join(logs_dir, f"paired_cyclegan_{args.anatomy}.csv")
        if os.path.exists(log_path):
            log       = pd.read_csv(log_path).to_dict(orient="list")
            best_ssim = max((v1 + v2) / 2 for v1, v2 in zip(log["mr2ct_ssim"], log["ct2mr_ssim"])
                           ) if log["mr2ct_ssim"] else -1.0
        print(f"Resumed from epoch {start_epoch}. Best mean SSIM: {best_ssim:.4f}")

    for epoch in range(start_epoch, config["EPOCHS"] + 1):
        # Log memory at start of each epoch
        mem = psutil.virtual_memory()
        print(f"[Epoch {epoch}] RAM used: {mem.used / 1e9:.1f}GB / {mem.total / 1e9:.1f}GB")
        gc.collect()
        torch.cuda.empty_cache()


        torch.set_grad_enabled(True)
        G.train(); F.train(); D_CT.train(); D_MR.train()

        epoch_losses = {k: [] for k in ["loss_G_total", "loss_D_total", "loss_paired", "loss_cycle"]}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config['EPOCHS']}"):
            losses = train_step(G, F, D_CT, D_MR, opt_G, opt_D,
                                batch, gan_loss, l1_loss, buf_CT, buf_MR, config, device, scaler_G, scaler_D)
            for k, v in losses.items():
                epoch_losses[k].append(v)

        sched_G.step(); sched_D.step()

        mean_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        val_metrics = validate(G, F, val_loader, device)
        mean_ssim   = (val_metrics["mr2ct_ssim"] + val_metrics["ct2mr_ssim"]) / 2

        log["epoch"].append(epoch)
        for k, v in mean_losses.items():
            log[k].append(v)
        for k, v in val_metrics.items():
            log[k].append(v)

        print(
            f"Epoch {epoch:03d} | "
            f"G: {mean_losses['loss_G_total']:.4f} | "
            f"D: {mean_losses['loss_D_total']:.4f} | "
            f"MR->CT SSIM: {val_metrics['mr2ct_ssim']:.4f} | "
            f"CT->MR SSIM: {val_metrics['ct2mr_ssim']:.4f}"
        )

        if mean_ssim > best_ssim:
            best_ssim = mean_ssim
            torch.save({"G": G.state_dict(), "F": F.state_dict(), "metrics": val_metrics},
                       os.path.join(ckpt_dir, "best_model.pth"))
            print(f"  --> New best model (mean SSIM: {best_ssim:.4f})")

        if epoch % config["CHECKPOINT_FREQ"] == 0:
            torch.save({
                "epoch": epoch,
                "G": G.state_dict(), "F": F.state_dict(),
                "D_CT": D_CT.state_dict(), "D_MR": D_MR.state_dict(),
                "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
                "metrics": val_metrics,
            }, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))
            print(f"  --> Checkpoint saved at epoch {epoch}")

        log_path = os.path.join(logs_dir, f"paired_cyclegan_{args.anatomy}.csv")
        pd.DataFrame(log).to_csv(log_path, index=False)

    print(f"\nTraining complete. Best mean SSIM: {best_ssim:.4f}")


if __name__ == "__main__":
    main()