"""
Regenerates the tables and figures of the report that image_restoration.py does not produce:
    Table 1  inverse filter PSNR with and without noise (and the degraded PSNR)
    Table 2  PSNR / SSIM of all restorations
    Figure 1 test images, blurs, and a blurred + noisy example
    Figure 2 radially averaged image PSD, noise PSD and the ratio Sn/Sf
    Figure 3 inverse filter: |H(u,0)|, with and without noise
    Figure 4 PSNR and SSIM against lambda (constant-K Wiener) and the Sn/Sf Wiener result
    Figures 5 and 6  restored images of I1 and I2 (constant-K at best PSNR and best SSIM, Sn/Sf Wiener)

The filter definitions and settings are the same as in image_restoration.py
(L = 25 taps, sigma = 0.03, np.random.seed(42) before the noise draw, 21 lambdas from 1e-5 to 1).
Restored and degraded images are clipped to [0, 1] before the metrics are computed.

Run:  python report_tables_figures.py        (img_restoration.mat in the same folder)
Output: ./report_output/  (png figures and csv tables, tables also printed)
"""
import os

import numpy as np
import pandas as pd
import scipy.io as sc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

L = 25
SIGMA = 0.03
LAMBDAS = np.logspace(-5, 0, 21)
DIRECTIONS = ["Horizontal", "Vertical", "Diagonal"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_output")


def kernel_otf(R, C, direction):
    """Normalised L-tap line kernel, centred, and its frequency response H."""
    k = np.zeros((R, C))
    idx = np.arange(-(L // 2), L // 2 + 1)
    if direction == "Horizontal":
        k[R // 2, C // 2 + idx] = 1
    elif direction == "Vertical":
        k[R // 2 + idx, C // 2] = 1
    else:  # Diagonal
        k[R // 2 + idx, C // 2 + idx] = 1
    k /= k.sum()
    return np.fft.fft2(np.fft.ifftshift(k))


def blur(img, H):
    return np.real(np.fft.ifft2(np.fft.fft2(img) * H))


def inverse_filter(g, H):
    return np.real(np.fft.ifft2(np.fft.fft2(g) / (H + 1e-5)))  # same 1e-5 as in image_restoration.py


def wiener(g, H, ratio):
    """H* G / (|H|^2 + ratio); ratio = lambda (constant K) or Sn/Sf (array)."""
    return np.real(np.fft.ifft2(np.conj(H) * np.fft.fft2(g) / (np.abs(H) ** 2 + ratio)))


def periodogram2d(img):
    return np.abs(np.fft.fft2(img)) ** 2 / img.size


def metrics(f, f_hat):
    f_hat = np.clip(f_hat, 0, 1)
    return {
        "MSE": float(np.mean((f - f_hat) ** 2)),
        "PSNR": float(peak_signal_noise_ratio(f, f_hat, data_range=1.0)),
        "SSIM": float(structural_similarity(f, f_hat, data_range=1.0)),
    }


def radial_average(P, nbins=66):
    R, C = P.shape
    r = np.sqrt(np.fft.fftfreq(R)[:, None] ** 2 + np.fft.fftfreq(C)[None, :] ** 2)
    edges = np.linspace(0, 0.5, nbins + 1)
    which = np.digitize(r.ravel(), edges) - 1
    centres, vals = [], []
    for b in range(nbins):
        m = which == b
        if m.any():
            centres.append(r.ravel()[m].mean())
            vals.append(P.ravel()[m].mean())
    return np.array(centres), np.array(vals)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = sc.loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), "img_restoration.mat"))
    images = {"I1": data["I1"].astype(float) / 255.0, "I2": data["I2"].astype(float) / 255.0}

    rows1, rows2, curves = [], [], {}
    examples, restored = {}, {}
    for name, f in images.items():
        R, C = f.shape
        Sf = np.maximum(periodogram2d(f), 1e-12)
        for d in DIRECTIONS:
            H = kernel_otf(R, C, d)
            np.random.seed(42)
            noise = np.random.normal(0, SIGMA, (R, C))
            fb = blur(f, H)
            g = fb + noise

            deg = metrics(f, g)
            inv_clean = metrics(f, inverse_filter(fb, H))
            inv_noisy_img = inverse_filter(g, H)
            inv_noisy = metrics(f, inv_noisy_img)
            rows1.append({
                "Image": name, "Blur": d,
                "Degraded PSNR": deg["PSNR"],
                "Inverse, no noise PSNR": inv_clean["PSNR"],
                "Inverse, sigma=0.03 PSNR": inv_noisy["PSNR"],
            })

            res = pd.DataFrame([{"lambda": lam, **metrics(f, wiener(g, H, lam))} for lam in LAMBDAS])
            best_psnr = res.loc[res["PSNR"].idxmax()]
            best_ssim = res.loc[res["SSIM"].idxmax()]
            w = metrics(f, wiener(g, H, SIGMA ** 2 / Sf))
            rows2.append({
                "Image": name, "Blur": d,
                "Degraded PSNR": deg["PSNR"], "Degraded SSIM": deg["SSIM"],
                "K (best PSNR/MSE)": best_psnr["lambda"], "PSNR @ best PSNR": best_psnr["PSNR"], "SSIM @ best PSNR": best_psnr["SSIM"],
                "K (best SSIM)": best_ssim["lambda"], "PSNR @ best SSIM": best_ssim["PSNR"], "SSIM @ best SSIM": best_ssim["SSIM"],
                "Sn/Sf Wiener PSNR": w["PSNR"], "Sn/Sf Wiener SSIM": w["SSIM"],
            })
            curves[(name, d)] = (res, w)
            restored[(name, d)] = {
                "g": g,
                "lam_p": best_psnr["lambda"], "lam_s": best_ssim["lambda"],
                "r_p": wiener(g, H, best_psnr["lambda"]), "r_s": wiener(g, H, best_ssim["lambda"]),
                "r_w": wiener(g, H, SIGMA ** 2 / Sf),
                "p_p": best_psnr["PSNR"], "p_s": best_ssim["PSNR"], "p_w": w["PSNR"],
            }
            if name == "I1" and d == "Horizontal":
                examples = {"H": H, "fb": fb, "g": g, "inv_clean": inverse_filter(fb, H), "inv_noisy": inv_noisy_img}

    t1, t2 = pd.DataFrame(rows1), pd.DataFrame(rows2)
    t1.to_csv(os.path.join(OUT, "table1_inverse_filter.csv"), index=False)
    t2.to_csv(os.path.join(OUT, "table2_all_restorations.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("Table 1: inverse filter PSNR (dB)")
    print(t1.round(2).to_string(index=False))
    print("\nTable 2: all restorations (sigma = 0.03, L = 25)")
    print(t2.round(3).to_string(index=False))

    # Figure 1: images, blurs, blurred + noise (horizontal)
    fig, axs = plt.subplots(2, 5, figsize=(13, 5.9))
    for i, (name, f) in enumerate(images.items()):
        R, C = f.shape
        axs[i, 0].imshow(f, cmap="gray", vmin=0, vmax=1)
        axs[i, 0].set_title(f"{name} original")
        for j, d in enumerate(DIRECTIONS):
            axs[i, j + 1].imshow(blur(f, kernel_otf(R, C, d)), cmap="gray", vmin=0, vmax=1)
            axs[i, j + 1].set_title(f"{d} blur")
        np.random.seed(42)
        g = blur(f, kernel_otf(R, C, "Horizontal")) + np.random.normal(0, SIGMA, (R, C))
        axs[i, 4].imshow(np.clip(g, 0, 1), cmap="gray", vmin=0, vmax=1)
        axs[i, 4].set_title("Horizontal + noise")
    for a in axs.ravel():
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig1_test_images.png"), dpi=130)
    plt.close()

    # Figure 2: radial PSD and Sn/Sf
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.88))
    for name, f in images.items():
        rc, Pf = radial_average(periodogram2d(f))
        axs[0].semilogy(rc, Pf, label=f"S_f of {name}")
        axs[1].semilogy(rc, SIGMA ** 2 / Pf, label=f"S_n/S_f of {name}")
    axs[0].axhline(SIGMA ** 2, color="k", ls="--", label="S_n = sigma^2")
    axs[0].set_title("Radially averaged image PSD and noise PSD")
    axs[0].set_ylabel("power per bin")
    axs[1].axhline(0.0316, color="r", ls=":", label="best constant K = 0.0316")
    axs[1].axhline(SIGMA ** 2, color="k", ls="--", label="sigma^2 = 9e-4")
    axs[1].set_title("Frequency-dependent ratio used by the Wiener filter")
    axs[1].set_ylabel("S_n / S_f")
    for a in axs:
        a.set_xlabel("radial frequency (cycles/pixel)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig2_spectra.png"), dpi=130)
    plt.close()

    # Figure 3: inverse filter, I1 horizontal blur
    f = images["I1"]
    R, C = f.shape
    H = examples["H"]
    u = np.fft.fftshift(np.fft.fftfreq(C))
    fig, axs = plt.subplots(1, 5, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.4, 1, 1, 1, 1]})
    axs[0].semilogy(u, np.fft.fftshift(np.abs(H[0, :])) + 1e-18)
    axs[0].set_ylim(1e-4, 2)
    axs[0].set_xlabel("normalised frequency u")
    axs[0].set_title("|H(u,0)| (horizontal)")
    axs[0].grid(alpha=0.3)
    for a, img, title in [
        (axs[1], examples["fb"], "Blurred, no noise"),
        (axs[2], examples["inv_clean"], "Inverse filter\n(no noise)"),
        (axs[3], examples["g"], "Blurred + noise"),
        (axs[4], examples["inv_noisy"], "Inverse filter\n(with noise)"),
    ]:
        a.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        a.set_title(title)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig3_inverse_filter.png"), dpi=130)
    plt.close()

    # Figure 4: PSNR and SSIM against lambda
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True)
    colours = {"I1": "tab:blue", "I2": "tab:orange"}
    for j, d in enumerate(DIRECTIONS):
        for name in images:
            res, w = curves[(name, d)]
            for i, metric in enumerate(["PSNR", "SSIM"]):
                axs[i, j].semilogx(res["lambda"], res[metric], color=colours[name], label=f"{name}: constant-K")
                axs[i, j].axhline(w[metric], color=colours[name], ls="--", label=f"{name}: S_n/S_f Wiener")
                axs[i, j].set_ylabel(f"{metric} (dB)" if metric == "PSNR" else metric)
                axs[i, j].grid(alpha=0.3)
        axs[0, j].set_title(f"{d} blur")
        axs[1, j].set_xlabel("lambda")
    axs[0, 0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig4_metrics_vs_lambda.png"), dpi=130)
    plt.close()

    # Figures 5 and 6: restored images (Figure 5 = I1, Figure 6 = I2)
    for fig_no, (name, f) in zip((5, 6), images.items()):
        fig, axs = plt.subplots(3, 5, figsize=(14, 10.5))
        for i, d in enumerate(DIRECTIONS):
            r = restored[(name, d)]
            panels = [
                (f, "Original" if i == 0 else ""),
                (r["g"], f"{d}\nblur + noise"),
                (r["r_p"], f"K={r['lam_p']:.3g} (best PSNR)\n{r['p_p']:.2f} dB"),
                (r["r_s"], f"K={r['lam_s']:.3g} (best SSIM)\n{r['p_s']:.2f} dB"),
                (r["r_w"], f"S_n/S_f Wiener\n{r['p_w']:.2f} dB"),
            ]
            for j, (img, title) in enumerate(panels):
                axs[i, j].imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
                axs[i, j].set_title(title, fontsize=15)
                axs[i, j].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f"fig{fig_no}_{name}_restored.png"), dpi=100)
        plt.close()

    print(f"\nFigures and tables written to {OUT}")


if __name__ == "__main__":
    main()
