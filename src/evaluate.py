import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # use non-interactive backend for server
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure as ssim_metric
from torchmetrics.functional import peak_signal_noise_ratio as psnr_metric
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import make_dataloader
from src.models  import UNetGenerator, ResNetGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all trained models")
    parser.add_argument("--data_root",   type=str, required=True)
    parser.add_argument("--split_dir",   type=str, required=True)
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--figures_dir", type=str, required=True)
    return parser.parse_args()


def load_split(split_dir, anatomy):
    filename = "combined_split.json" if anatomy == "combined" else f"{anatomy.lower()}_split.json"
    with open(os.path.join(split_dir, filename), "r") as f:
        return json.load(f)


# ============================================================
# METRIC COMPUTATION
# ============================================================

def compute_metrics_on_loader(model_G, model_F, test_loader, device, model_type="cyclegan"):
    """
    Compute MAE, SSIM, PSNR for both directions on a test loader.
    model_type: 'pix2pix_mr2ct', 'pix2pix_ct2mr', or 'cyclegan'
    Returns a list of per-slice metric dicts.
    """
    if model_type == "cyclegan":
        model_G.eval(); model_F.eval()
    else:
        model_G.eval()

    records = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            real_MR = batch["mr"].to(device)
            real_CT = batch["ct"].to(device)
            mask    = batch["mask"].to(device)

            if model_type == "pix2pix_mr2ct":
                fake_CT   = model_G(real_MR)
                fake_CT_m = fake_CT * mask
                real_CT_m = real_CT * mask
                records.append({
                    "direction": "mr2ct",
                    "patient":   batch["patient"][0],
                    "slice":     batch["slice"][0].item(),
                    "mae":       torch.mean(torch.abs(fake_CT_m - real_CT_m)).item(),
                    "ssim":      ssim_metric(fake_CT_m, real_CT_m, data_range=2.0).item(),
                    "psnr":      psnr_metric(fake_CT_m, real_CT_m, data_range=2.0).item(),
                })

            elif model_type == "pix2pix_ct2mr":
                fake_MR   = model_G(real_CT)
                fake_MR_m = fake_MR * mask
                real_MR_m = real_MR * mask
                records.append({
                    "direction": "ct2mr",
                    "patient":   batch["patient"][0],
                    "slice":     batch["slice"][0].item(),
                    "mae":       torch.mean(torch.abs(fake_MR_m - real_MR_m)).item(),
                    "ssim":      ssim_metric(fake_MR_m, real_MR_m, data_range=2.0).item(),
                    "psnr":      psnr_metric(fake_MR_m, real_MR_m, data_range=2.0).item(),
                })

            else:
                # cyclegan — both directions
                fake_CT   = model_G(real_MR)
                fake_MR   = model_F(real_CT)

                fake_CT_m = fake_CT * mask; real_CT_m = real_CT * mask
                fake_MR_m = fake_MR * mask; real_MR_m = real_MR * mask

                for direction, fake_m, real_m in [
                    ("mr2ct", fake_CT_m, real_CT_m),
                    ("ct2mr", fake_MR_m, real_MR_m),
                ]:
                    records.append({
                        "direction": direction,
                        "patient":   batch["patient"][0],
                        "slice":     batch["slice"][0].item(),
                        "mae":       torch.mean(torch.abs(fake_m - real_m)).item(),
                        "ssim":      ssim_metric(fake_m, real_m, data_range=2.0).item(),
                        "psnr":      psnr_metric(fake_m, real_m, data_range=2.0).item(),
                    })

    torch.set_grad_enabled(True)
    return records


# ============================================================
# FIGURE GENERATION
# ============================================================

def save_qualitative_comparison(
    pix2pix_G_mr2ct, pix2pix_G_ct2mr,
    cyclegan_G, cyclegan_F,
    test_loader, anatomy, figures_dir, device,
    num_samples=3
):
    """
    Generate side-by-side qualitative comparison figure.
    Rows: one sample per patient
    Columns: Real MRI | Pix2Pix sCT | Paired CycleGAN sCT | Real CT
             Real CT  | Pix2Pix sMRI | Paired CycleGAN sMRI | Real MRI
    """
    pix2pix_G_mr2ct.eval(); pix2pix_G_ct2mr.eval()
    cyclegan_G.eval(); cyclegan_F.eval()

    samples = []
    with torch.no_grad():
        for batch in test_loader:
            if len(samples) >= num_samples:
                break
            real_MR = batch["mr"].to(device)
            real_CT = batch["ct"].to(device)
            mask    = batch["mask"].to(device)

            pix_fake_CT = pix2pix_G_mr2ct(real_MR)
            pix_fake_MR = pix2pix_G_ct2mr(real_CT)
            cyc_fake_CT = cyclegan_G(real_MR)
            cyc_fake_MR = cyclegan_F(real_CT)

            samples.append({
                "real_MR":     batch["mr"][0][1].cpu().numpy(),
                "real_CT":     batch["ct"][0][1].cpu().numpy(),
                "mask":        batch["mask"][0][0].cpu().numpy(),
                "pix_fake_CT": pix_fake_CT[0][1].cpu().numpy(),
                "pix_fake_MR": pix_fake_MR[0][1].cpu().numpy(),
                "cyc_fake_CT": cyc_fake_CT[0][1].cpu().numpy(),
                "cyc_fake_MR": cyc_fake_MR[0][1].cpu().numpy(),
                "patient":     batch["patient"][0],
            })

    torch.set_grad_enabled(True)

    # MRI->CT comparison figure
    fig, axes = plt.subplots(num_samples, 5, figsize=(22, 5 * num_samples))
    col_titles = ["Real MRI", "Pix2Pix sCT", "Paired CycleGAN sCT", "Real CT", "Error map"]

    for row, s in enumerate(samples):
        diff_pix = np.abs(s["real_CT"] - s["pix_fake_CT"]) * s["mask"]
        diff_cyc = np.abs(s["real_CT"] - s["cyc_fake_CT"]) * s["mask"]

        axes[row][0].imshow(s["real_MR"],    cmap="gray")
        axes[row][1].imshow(s["pix_fake_CT"], cmap="gray")
        axes[row][2].imshow(s["cyc_fake_CT"], cmap="gray")
        axes[row][3].imshow(s["real_CT"],    cmap="gray")
        # overlay both error maps — pix2pix in red channel, cyclegan in blue
        error_overlay = np.zeros((*diff_pix.shape, 3))
        error_overlay[..., 0] = np.clip(diff_pix / diff_pix.max() if diff_pix.max() > 0 else diff_pix, 0, 1)
        error_overlay[..., 2] = np.clip(diff_cyc / diff_cyc.max() if diff_cyc.max() > 0 else diff_cyc, 0, 1)
        axes[row][4].imshow(error_overlay)

        for col in range(5):
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(col_titles[col], fontsize=11)
        axes[row][0].set_ylabel(f"Patient\n{s['patient']}", fontsize=8, rotation=90)

    plt.suptitle(f"MRI→CT Qualitative Comparison — {anatomy}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(figures_dir, f"qualitative_mr2ct_{anatomy}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # CT->MRI comparison figure
    fig, axes = plt.subplots(num_samples, 5, figsize=(22, 5 * num_samples))
    col_titles = ["Real CT", "Pix2Pix sMRI", "Paired CycleGAN sMRI", "Real MRI", "Error map"]

    for row, s in enumerate(samples):
        diff_pix = np.abs(s["real_MR"] - s["pix_fake_MR"]) * s["mask"]
        diff_cyc = np.abs(s["real_MR"] - s["cyc_fake_MR"]) * s["mask"]

        axes[row][0].imshow(s["real_CT"],    cmap="gray")
        axes[row][1].imshow(s["pix_fake_MR"], cmap="gray")
        axes[row][2].imshow(s["cyc_fake_MR"], cmap="gray")
        axes[row][3].imshow(s["real_MR"],    cmap="gray")
        error_overlay = np.zeros((*diff_pix.shape, 3))
        error_overlay[..., 0] = np.clip(diff_pix / diff_pix.max() if diff_pix.max() > 0 else diff_pix, 0, 1)
        error_overlay[..., 2] = np.clip(diff_cyc / diff_cyc.max() if diff_cyc.max() > 0 else diff_cyc, 0, 1)
        axes[row][4].imshow(error_overlay)

        for col in range(5):
            axes[row][col].axis("off")
            if row == 0:
                axes[row][col].set_title(col_titles[col], fontsize=11)

    plt.suptitle(f"CT→MRI Qualitative Comparison — {anatomy}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(figures_dir, f"qualitative_ct2mr_{anatomy}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_training_curves(logs_dir, figures_dir, anatomy):
    """
    Plot training curves for both models side by side.
    """
    pix_mr2ct_path = os.path.join(logs_dir, "pix2pix",        f"pix2pix_{anatomy}_mr2ct.csv")
    pix_ct2mr_path = os.path.join(logs_dir, "pix2pix",        f"pix2pix_{anatomy}_ct2mr.csv")
    cyc_path       = os.path.join(logs_dir, "paired_cyclegan", f"paired_cyclegan_{anatomy}.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if os.path.exists(pix_mr2ct_path):
        df = pd.read_csv(pix_mr2ct_path)
        axes[0][0].plot(df["epoch"], df["val_ssim"],  label="Pix2Pix MRI->CT",  color="steelblue",  linestyle="--")
        axes[1][0].plot(df["epoch"], df["val_mae"],   label="Pix2Pix MRI->CT",  color="steelblue",  linestyle="--")

    if os.path.exists(pix_ct2mr_path):
        df = pd.read_csv(pix_ct2mr_path)
        axes[0][0].plot(df["epoch"], df["val_ssim"],  label="Pix2Pix CT->MRI",  color="steelblue",  linestyle=":")
        axes[1][0].plot(df["epoch"], df["val_mae"],   label="Pix2Pix CT->MRI",  color="steelblue",  linestyle=":")

    if os.path.exists(cyc_path):
        df = pd.read_csv(cyc_path)
        axes[0][0].plot(df["epoch"], df["mr2ct_ssim"], label="Paired CycleGAN MRI->CT", color="darkorange", linestyle="-")
        axes[0][0].plot(df["epoch"], df["ct2mr_ssim"], label="Paired CycleGAN CT->MRI", color="darkorange", linestyle="-.")
        axes[1][0].plot(df["epoch"], df["mr2ct_mae"],  label="Paired CycleGAN MRI->CT", color="darkorange", linestyle="-")
        axes[1][0].plot(df["epoch"], df["ct2mr_mae"],  label="Paired CycleGAN CT->MRI", color="darkorange", linestyle="-.")

        axes[0][1].plot(df["epoch"], df["loss_G_total"], label="Generator loss",     color="seagreen")
        axes[0][1].plot(df["epoch"], df["loss_D_total"], label="Discriminator loss", color="tomato")
        axes[1][1].plot(df["epoch"], df["loss_paired"],  label="Paired L1",          color="purple")
        axes[1][1].plot(df["epoch"], df["loss_cycle"],   label="Cycle consistency",  color="teal")

    axes[0][0].set_title(f"{anatomy} — Validation SSIM"); axes[0][0].set_ylabel("SSIM"); axes[0][0].legend(fontsize=7)
    axes[1][0].set_title(f"{anatomy} — Validation MAE");  axes[1][0].set_ylabel("MAE");  axes[1][0].legend(fontsize=7)
    axes[0][1].set_title(f"{anatomy} — GAN losses");      axes[0][1].set_ylabel("Loss"); axes[0][1].legend(fontsize=7)
    axes[1][1].set_title(f"{anatomy} — Component losses"); axes[1][1].set_ylabel("Loss"); axes[1][1].legend(fontsize=7)

    for ax in axes.flatten():
        ax.set_xlabel("Epoch")

    plt.suptitle(f"Training Curves — {anatomy}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(figures_dir, f"training_curves_{anatomy}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_metrics_barplot(results_df, anatomy, figures_dir):
    """
    Bar chart comparing Pix2Pix vs Paired CycleGAN
    per direction per metric for one anatomy.
    """
    df = results_df[results_df["anatomy"] == anatomy]
    if len(df) == 0:
        print(f"No results found for {anatomy}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics    = ["mae", "ssim", "psnr"]
    directions = ["mr2ct", "ct2mr"]
    models     = ["pix2pix", "paired_cyclegan"]
    colors     = {"pix2pix": "steelblue", "paired_cyclegan": "darkorange"}
    labels     = {"pix2pix": "Pix2Pix", "paired_cyclegan": "Paired CycleGAN"}

    x      = np.arange(len(directions))
    width  = 0.35

    for col, metric in enumerate(metrics):
        ax = axes[col]
        for i, model in enumerate(models):
            means = []
            stds  = []
            for direction in directions:
                subset = df[(df["model"] == model) & (df["direction"] == direction)]
                if len(subset) > 0:
                    means.append(subset[metric].mean())
                    stds.append(subset[metric].std())
                else:
                    means.append(0); stds.append(0)

            offset = (i - 0.5) * width
            bars   = ax.bar(x + offset, means, width,
                            yerr=stds, capsize=4,
                            color=colors[model], label=labels[model],
                            edgecolor="white")

            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (max(stds) * 0.1 if stds else 0.01),
                        f"{mean:.3f}", ha="center", fontsize=7)

        ax.set_title(f"{metric.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(["MRI→CT", "CT→MRI"])
        ax.legend(fontsize=8)
        if metric == "mae":
            ax.set_ylabel("MAE (lower is better)")
        elif metric == "ssim":
            ax.set_ylabel("SSIM (higher is better)")
        else:
            ax.set_ylabel("PSNR dB (higher is better)")

    plt.suptitle(f"Pix2Pix vs Paired CycleGAN — {anatomy} Test Set", fontsize=13)
    plt.tight_layout()
    path = os.path.join(figures_dir, f"metrics_comparison_{anatomy}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def save_statistical_tests(results_df, output_dir):
    """
    Paired t-test between Pix2Pix and Paired CycleGAN
    per anatomy per direction per metric.
    """
    records = []

    for anatomy in results_df["anatomy"].unique():
        for direction in ["mr2ct", "ct2mr"]:
            for metric in ["mae", "ssim", "psnr"]:
                pix_vals = results_df[
                    (results_df["model"]     == "pix2pix") &
                    (results_df["anatomy"]   == anatomy) &
                    (results_df["direction"] == direction)
                ][metric].values

                cyc_vals = results_df[
                    (results_df["model"]     == "paired_cyclegan") &
                    (results_df["anatomy"]   == anatomy) &
                    (results_df["direction"] == direction)
                ][metric].values

                if len(pix_vals) == 0 or len(cyc_vals) == 0:
                    continue

                min_len      = min(len(pix_vals), len(cyc_vals))
                t_stat, p_val = stats.ttest_rel(pix_vals[:min_len], cyc_vals[:min_len])

                records.append({
                    "anatomy":     anatomy,
                    "direction":   direction,
                    "metric":      metric,
                    "pix2pix_mean":    pix_vals.mean(),
                    "cyclegan_mean":   cyc_vals.mean(),
                    "t_statistic": t_stat,
                    "p_value":     p_val,
                    "significant": p_val < 0.05,
                })

    stat_df   = pd.DataFrame(records)
    stat_path = os.path.join(output_dir, "statistical_tests.csv")
    stat_df.to_csv(stat_path, index=False)
    print(f"Saved statistical tests: {stat_path}")
    print(stat_df.to_string(index=False))
    return stat_df


def save_summary_table(results_df, output_dir):
    """
    Save the main results table as CSV and LaTeX.
    """
    summary = results_df.groupby(
        ["model", "anatomy", "direction"]
    )[["mae", "ssim", "psnr"]].agg(["mean", "std"]).round(4)

    csv_path = os.path.join(output_dir, "main_results_table.csv")
    summary.to_csv(csv_path)
    print(f"Saved results table: {csv_path}")

    # latex version
    model_display = {
        "pix2pix":         "Pix2Pix",
        "paired_cyclegan": "Paired CycleGAN (ours)",
    }
    dir_display = {
        "mr2ct": r"MRI$\rightarrow$CT",
        "ct2mr": r"CT$\rightarrow$MRI",
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Quantitative results on the SynthRAD2025 Task 1 test set. Mean $\pm$ std.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Direction & MAE & SSIM & PSNR \\")
    lines.append(r"\midrule")

    for (model, anatomy, direction), row in summary.iterrows():
        model_str = model_display.get(model, model)
        dir_str   = dir_display.get(direction, direction)
        mae_str   = f"${row['mae']['mean']:.4f} \\pm {row['mae']['std']:.4f}$"
        ssim_str  = f"${row['ssim']['mean']:.4f} \\pm {row['ssim']['std']:.4f}$"
        psnr_str  = f"${row['psnr']['mean']:.4f} \\pm {row['psnr']['std']:.4f}$"
        lines.append(f"{model_str} ({anatomy}) & {dir_str} & {mae_str} & {ssim_str} & {psnr_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    tex_path = os.path.join(output_dir, "main_results_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved LaTeX table: {tex_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(args.config, "r") as f:
        config = json.load(f)

    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "evaluation"), exist_ok=True)

    anatomies   = ["AB", "TH", "HN"]
    all_records = []

    for anatomy in anatomies:
        print(f"\n{'='*50}")
        print(f"Evaluating {anatomy}")
        print(f"{'='*50}")

        split_data  = load_split(args.split_dir, anatomy)
        test_loader = make_dataloader(
            split_data["test"], anatomy, args.data_root, "test",
            batch_size=1, num_workers=config["NUM_WORKERS"],
            image_size=config["IMAGE_SIZE"]
        )

        # ---- load Pix2Pix MRI->CT ----
        pix_mr2ct_path = os.path.join(
            args.output_dir, "checkpoints", "pix2pix", anatomy, "mr2ct", "best_model.pth"
        )
        pix_ct2mr_path = os.path.join(
            args.output_dir, "checkpoints", "pix2pix", anatomy, "ct2mr", "best_model.pth"
        )
        cyc_path = os.path.join(
            args.output_dir, "checkpoints", "paired_cyclegan", anatomy, "best_model.pth"
        )

        pix_G_mr2ct = None
        pix_G_ct2mr = None
        cyc_G       = None
        cyc_F       = None

        if os.path.exists(pix_mr2ct_path):
            pix_G_mr2ct = UNetGenerator(in_ch=3, out_ch=3).to(device)
            ckpt        = torch.load(pix_mr2ct_path, map_location=device)
            pix_G_mr2ct.load_state_dict(ckpt["G"])
            print(f"  Loaded Pix2Pix MRI->CT for {anatomy}")

            records = compute_metrics_on_loader(pix_G_mr2ct, None, test_loader, device, "pix2pix_mr2ct")
            for r in records:
                r["model"]   = "pix2pix"
                r["anatomy"] = anatomy
            all_records.extend(records)
        else:
            print(f"  WARNING: Pix2Pix MRI->CT checkpoint not found for {anatomy}")

        if os.path.exists(pix_ct2mr_path):
            pix_G_ct2mr = UNetGenerator(in_ch=3, out_ch=3).to(device)
            ckpt        = torch.load(pix_ct2mr_path, map_location=device)
            pix_G_ct2mr.load_state_dict(ckpt["G"])
            print(f"  Loaded Pix2Pix CT->MRI for {anatomy}")

            records = compute_metrics_on_loader(pix_G_ct2mr, None, test_loader, device, "pix2pix_ct2mr")
            for r in records:
                r["model"]   = "pix2pix"
                r["anatomy"] = anatomy
            all_records.extend(records)
        else:
            print(f"  WARNING: Pix2Pix CT->MRI checkpoint not found for {anatomy}")

        if os.path.exists(cyc_path):
            cyc_G = ResNetGenerator(in_ch=3, out_ch=3).to(device)
            cyc_F = ResNetGenerator(in_ch=3, out_ch=3).to(device)
            ckpt  = torch.load(cyc_path, map_location=device)
            cyc_G.load_state_dict(ckpt["G"])
            cyc_F.load_state_dict(ckpt["F"])
            print(f"  Loaded Paired CycleGAN for {anatomy}")

            records = compute_metrics_on_loader(cyc_G, cyc_F, test_loader, device, "cyclegan")
            for r in records:
                r["model"]   = "paired_cyclegan"
                r["anatomy"] = anatomy
            all_records.extend(records)
        else:
            print(f"  WARNING: Paired CycleGAN checkpoint not found for {anatomy}")

        # generate qualitative figures if both models are available
        if pix_G_mr2ct and pix_G_ct2mr and cyc_G and cyc_F:
            save_qualitative_comparison(
                pix_G_mr2ct, pix_G_ct2mr,
                cyc_G, cyc_F,
                test_loader, anatomy,
                args.figures_dir, device,
                num_samples=3
            )

        save_training_curves(
            os.path.join(args.output_dir, "logs"),
            args.figures_dir,
            anatomy
        )

    if all_records:
        results_df = pd.DataFrame(all_records)

        # save unified results
        unified_path = os.path.join(args.output_dir, "evaluation", "unified_results.csv")
        results_df.to_csv(unified_path, index=False)
        print(f"\nSaved unified results: {unified_path}")

        # print summary
        summary = results_df.groupby(["model", "anatomy", "direction"])[["mae", "ssim", "psnr"]].agg(["mean", "std"])
        print("\nResults Summary:")
        print(summary.round(4))

        # save formatted tables
        save_summary_table(results_df, os.path.join(args.output_dir, "evaluation"))

        # statistical tests
        save_statistical_tests(results_df, os.path.join(args.output_dir, "evaluation"))

        # bar plots per anatomy
        for anatomy in anatomies:
            save_metrics_barplot(results_df, anatomy, args.figures_dir)

        print(f"\nAll figures saved to: {args.figures_dir}")
        print(f"All results saved to: {args.output_dir}/evaluation/")
    else:
        print("No results to evaluate — check that checkpoints exist.")


if __name__ == "__main__":
    main()