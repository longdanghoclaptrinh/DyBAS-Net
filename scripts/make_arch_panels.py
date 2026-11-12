"""Hình con cho sơ đồ kiến trúc DyBAS-Net, vẽ từ dữ liệu và mô hình thật (không minh họa giả).

Chạy checkpoint DyBAS-Net (hai giai đoạn) đã huấn luyện trên bộ Load (10 biến, theo giờ; L = 720, S = 96, seed 2021)
trên một cửa sổ thật của tập test (dự báo bắt đầu 00:00 thứ Hai), giữ lại đại lượng trung gian của từng khối và vẽ mỗi
đại lượng thành một ảnh nhỏ, chữ tiếng Anh, cùng bảng màu với hình Introduction (scripts/make_intro_figure.py).
Ảnh ra ở paper/figures/arch_panels/: từng ảnh PNG (300 dpi) và panel_sheet.png (mọi ảnh, có mã P01…, để đính kèm
khi nhờ công cụ vẽ ghép vào sơ đồ). Ví dụ: python scripts/make_arch_panels.py
"""

import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dytimenet.data.dataset import TimeSeriesDataset  # noqa: E402
from dytimenet.models import build_model  # noqa: E402
from make_intro_figure import load as intro_load, R as INTRO_R  # noqa: E402

CKPT = Path("checkpoints/fix_Load_L720/Load_L720_S96_seed2021_20261027T200708939037Z/best.pt")
CSV = Path("configs/load_new.csv")
OUT = Path("paper/figures/arch_panels")
FORECAST_FROM = "2024-10-07"          # thứ Hai đầu tiên của cửa sổ minh họa (tuần bình thường, ngoài Tết)
VAR = 0                                 # biến minh họa cho các khối theo từng biến: total_consumption_mw
SHORT = ["Total", "Industrial", "Residential", "Per-capita", "Temp.", "Humidity", "Rainfall", "AQI", "Weather",
         "Time"]

BLUE, ORANGE, RED, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#e34948", "#0b0b0b", "#52514e", "#e4e3df"
BASIS = ["#2a78d6", "#eb6834", "#2e9e5b", "#8a5cc2", "#c9a227", "#d0457a"]   # e1 … e6
SEQ = LinearSegmentedColormap.from_list("blue", ["#f4f9ff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f7f7f5", "#f2a07e", "#c43c2c"])

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.edgecolor": MUTED, "axes.linewidth": 0.5,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "xtick.major.size": 2,
                     "ytick.major.size": 2, "mathtext.fontset": "stix", "savefig.dpi": 300})


def run_model():
    """Chạy mô hình trên cửa sổ minh họa; trả dict các đại lượng trung gian (một mẫu, numpy) và thông tin cửa sổ."""
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build_model("dytimebase", **ck["model_kwargs"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    L, S = ck["model_kwargs"]["seq_len"], ck["model_kwargs"]["pred_len"]
    ds = TimeSeriesDataset(CSV, L, S, split="test", dataset_name="Load", scaler_state=ck["scaler"],
                           expected_columns=ck["feature_columns"])
    stamps = pd.to_datetime(pd.read_csv(CSV, usecols=["timestamp"])["timestamp"])
    first = int(np.flatnonzero(stamps.values >= np.datetime64(FORECAST_FROM))[0])   # dòng của bước dự báo đầu tiên
    idx = first - L - ds.start
    x, y = ds[idx]
    x = x[None]
    with torch.no_grad():
        seg_raw = model._segments(x)                                              # [1, C, P, N]
        seg, mean, std = model._normalize(seg_raw, model.period_norm, model.inst_std)
        e = model.basis(seg)                                                      # [1, C, P, R]
        h_enc = model.embed(e.reshape(1, x.shape[1], -1))                         # [1, C, D]
        h_norm = model.cross.norm(h_enc)
        t = h_norm.transpose(1, 2)                                                # [1, D, C]
        hid = model.cross.activation(model.cross.df_linear1(t))                   # [1, D, C']
        x_cross = model.cross.df_linear2(hid).transpose(1, 2)                     # [1, C, D]
        x_intra = x_cross + h_norm                                                # residual_from_norm = True
        x_hat = model.intra(x_intra)
        de = model.out(x_hat).reshape(e.shape)
        e2 = e + de
        y_seg = model.forecast(e2)                                                # [1, C, P, N']
        pred = model._merge(y_seg * std + mean)                                   # [1, C, S]
        assert torch.allclose(pred, model(x), atol=1e-5)
    sc_mean, sc_scale = np.asarray(ck["scaler"]["mean_"]), np.asarray(ck["scaler"]["scale_"])
    inv = lambda a, c: a * sc_scale[c] + sc_mean[c]
    n = lambda tns: tns[0].numpy()
    mse = float(((pred - y[None]) ** 2).mean())
    return dict(x=n(x), y=y.numpy(), seg_raw=n(seg_raw), mean=n(mean), seg=n(seg), e=n(e), h_enc=n(h_enc),
                h_norm=n(h_norm), hid=n(hid), x_cross=n(x_cross), x_intra=n(x_intra), x_hat=n(x_hat), de=n(de),
                e2=n(e2), y_seg=n(y_seg), pred=n(pred), W_B=model.basis.weight.detach().numpy().T,
                W1=model.cross.df_linear1.weight.detach().numpy().T, W2=model.cross.df_linear2.weight.detach().numpy().T,
                W_F=model.forecast.weight.detach().numpy().T, p1=model.cross.df_linear1.p1, inv=inv,
                start=stamps[first - L], fstart=stamps[first], mse=mse, L=L, S=S)


def panel(w=1.55, h=1.05):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def save(fig, name):
    fig.tight_layout(pad=0.15)
    fig.savefig(OUT / f"{name}.png", facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def heat(ax, m, cmap, xlabel, ylabel, center=False, xt=None, yt=None):
    norm = TwoSlopeNorm(0, -np.abs(m).max(), np.abs(m).max()) if center else None
    im = ax.imshow(m, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel(ylabel, labelpad=1)
    if xt is not None:
        ax.set_xticks(*xt)
    if yt is not None:
        ax.set_yticks(*yt)
    for s in ax.spines.values():
        s.set_visible(True)
    return im


def draw_all(r):
    P, N = r["seg"].shape[1:]
    R, Np = r["e"].shape[2], r["y_seg"].shape[2]
    hours = np.arange(1, P + 1)
    seg_t = (np.array([0, 4, 9, 14, 19, 24, 29]), ["1", "5", "10", "15", "20", "25", "30"])
    ph_t = (np.array([0, 5, 11, 17, 23]), ["1", "6", "12", "18", "24"])

    # P01: chuỗi vào, 4 trong 10 biến (đã chuẩn hóa), 30 ngày
    fig, axs = plt.subplots(4, 1, figsize=(1.75, 1.35), sharex=True)
    for a, c, col in zip(axs, (0, 2, 4, 5), (BLUE, ORANGE, RED, "#2e9e5b")):
        a.plot(r["x"][c], color=col, lw=0.45)
        a.set_yticks([])
        a.set_ylabel(SHORT[c], rotation=0, ha="right", va="center", fontsize=6, labelpad=2)
        a.spines[["top", "right", "left"]].set_visible(False)
    axs[-1].set_xticks(np.arange(0, r["L"] + 1, 240), ["0", "240", "480", "720"])
    axs[-1].set_xlabel("time step t (hour)", labelpad=1)
    save(fig, "P01_input")

    # P02: ma trận đoạn X^(i) (P × N)
    fig, ax = panel(1.6, 1.15)
    im = heat(ax, r["seg_raw"][VAR], SEQ, "segment n (day)", "phase p (hour)", xt=seg_t, yt=ph_t)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).ax.tick_params(labelsize=5, length=1.5)
    save(fig, "P02_segment_matrix")

    # P03: N đoạn chồng lên nhau và trung bình theo pha μ_p
    fig, ax = panel()
    for n in range(N):
        ax.plot(hours, r["seg_raw"][VAR][:, n], color=BLUE, lw=0.4, alpha=0.35)
    ax.plot(hours, r["mean"][VAR][:, 0], color=INK, lw=1.3)
    ax.text(0.03, 0.97, r"$\mu_p$", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_xlabel("phase p (hour)", labelpad=1)
    ax.set_yticks([])
    save(fig, "P03_phase_mean")

    # P04: X~ sau khi trừ μ_p
    fig, ax = panel(1.6, 1.15)
    im = heat(ax, r["seg"][VAR], DIV, "segment n (day)", "phase p (hour)", center=True, xt=seg_t, yt=ph_t)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).ax.tick_params(labelsize=5, length=1.5)
    save(fig, "P04_normalized_segments")

    # P05: R thành phần cơ sở e_r (cột của E)
    fig, ax = panel()
    for k in range(R):
        ax.plot(hours, r["e"][VAR][:, k], color=BASIS[k], lw=0.9, label=rf"$e_{k + 1}$")
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_xlabel("phase p (hour)", labelpad=1)
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.16), handlelength=0.8,
              columnspacing=0.5, handletextpad=0.3, borderaxespad=0)
    save(fig, "P05_basis_E")

    # P06: W_B (N × R): trọng số tổ hợp N đoạn thành R cơ sở
    fig, ax = panel(1.05, 1.15)
    im = heat(ax, r["W_B"], DIV, "basis r", "segment n", center=True,
              xt=(np.arange(R), [str(k + 1) for k in range(R)]), yt=seg_t)
    save(fig, "P06_W_B")

    # P07: ma trận Gram G = E^T E (đã chuẩn hóa thành cos), minh họa L_orth
    e = r["e"][VAR]
    g = e.T @ e
    d = np.sqrt(np.diag(g))
    fig, ax = panel(1.15, 1.05)
    im = heat(ax, g / np.outer(d, d), DIV, "basis s", "basis r", center=True,
              xt=(np.arange(R), [str(k + 1) for k in range(R)]), yt=(np.arange(R), [str(k + 1) for k in range(R)]))
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).ax.tick_params(labelsize=5, length=1.5)
    save(fig, "P07_gram_orth")

    # P08: X_enc (D × C) sau Flatten + Embed
    var_t = (np.arange(len(SHORT)), SHORT)
    fig, ax = panel(1.55, 1.25)
    im = heat(ax, r["h_enc"].T, DIV, "", "feature (D = 64)", center=True, xt=var_t,
              yt=(np.array([0, 31, 63]), ["1", "32", "64"]))
    ax.tick_params(axis="x", labelrotation=90, labelsize=5)
    save(fig, "P08_X_enc")

    # P09: W_df,1 (C × C') với một mẫu mặt nạ cạnh M1 thật (p1 của checkpoint): ô bị tắt tô xám
    rng = np.random.default_rng(0)
    m1 = rng.random(r["W1"].shape) >= r["p1"]
    w = np.where(m1, r["W1"], np.nan)
    fig, ax = panel(1.3, 1.3)
    cmap = DIV.copy()
    cmap.set_bad("#bdbcb6")
    lim = np.abs(r["W1"]).max()
    ax.imshow(w, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", interpolation="nearest")
    for i, j in zip(*np.nonzero(~m1)):
        ax.plot([j - 0.4, j + 0.4], [i - 0.4, i + 0.4], color=INK, lw=0.5)
        ax.plot([j - 0.4, j + 0.4], [i + 0.4, i - 0.4], color=INK, lw=0.5)
    ax.set_yticks(*var_t)
    ax.set_xticks(np.arange(r["W1"].shape[1]), [str(k + 1) for k in range(r["W1"].shape[1])])
    ax.set_xlabel("hidden variable (C' = 7)", labelpad=1)
    ax.tick_params(axis="y", labelsize=5)
    save(fig, "P09_W_df1_masked")

    # P10: ánh xạ liên biến hiệu dụng W_df,1 W_df,2 (C × C), bỏ qua GELU: biến nào đóng góp vào biến nào
    fig, ax = panel(1.45, 1.35)
    heat(ax, r["W1"] @ r["W2"], DIV, "to variable", "from variable", center=True, xt=var_t, yt=var_t)
    ax.tick_params(axis="x", labelrotation=90, labelsize=5)
    ax.tick_params(axis="y", labelsize=5)
    save(fig, "P10_cross_mixing_CxC")

    # P11: IntraV: đầu vào X_intra và đầu ra X^_intra (D × C), cùng thang màu
    fig, axs = plt.subplots(1, 2, figsize=(1.9, 1.15), sharey=True)
    lim = max(np.abs(r["x_intra"]).max(), np.abs(r["x_hat"]).max())
    for a, m, t in zip(axs, (r["x_intra"], r["x_hat"]), (r"$\mathcal{X}_{intra}$", r"$\hat{\mathcal{X}}_{intra}$")):
        a.imshow(m.T, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", interpolation="nearest")
        a.set_title(t, pad=2)
        a.set_xticks(*var_t)
        a.tick_params(axis="x", labelrotation=90, labelsize=4.5)
    axs[0].set_yticks([0, 31, 63], ["1", "32", "64"])
    axs[0].set_ylabel("feature", labelpad=1)
    save(fig, "P11_intra_in_out")

    # P12: ΔE (P × R) và E → E' cho cơ sở thay đổi nhiều nhất
    fig, ax = panel()
    for k in range(R):
        ax.plot(hours, r["de"][VAR][:, k], color=BASIS[k], lw=0.9)
    ax.axhline(0, color=MUTED, lw=0.4)
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_xlabel("phase p (hour)", labelpad=1)
    ax.set_yticks([])
    ax.text(0.03, 0.97, r"$\Delta E$", transform=ax.transAxes, va="top", fontsize=8)
    save(fig, "P12_delta_E")

    k = int(np.abs(r["de"][VAR]).sum(0).argmax())
    fig, ax = panel()
    ax.plot(hours, r["e"][VAR][:, k], color=BASIS[k], lw=1.0, ls="--", label=rf"$e_{k + 1}$")
    ax.plot(hours, r["e2"][VAR][:, k], color=BASIS[k], lw=1.3, label=rf"$e'_{k + 1}$")
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_xlabel("phase p (hour)", labelpad=1)
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=6, loc="best", handlelength=1.4)
    save(fig, "P13_E_vs_Eprime")

    # P14: N' đoạn tương lai (sau khi cộng μ_p), từng ngày một đường
    fig, ax = panel()
    for n in range(Np):
        ax.plot(hours, r["inv"](r["y_seg"][VAR][:, n] + r["mean"][VAR][:, 0], VAR), color=ORANGE, lw=0.9,
                alpha=0.45 + 0.55 * n / max(Np - 1, 1))
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_xlabel("phase p (hour)", labelpad=1)
    ax.set_ylabel("MW", labelpad=1)
    ax.text(0.03, 0.97, rf"$N'={Np}$ future days", transform=ax.transAxes, va="top", fontsize=6)
    save(fig, "P14_future_segments")

    # P15: dự báo so với thực tế (MW): 3 ngày cuối của quá khứ + S bước
    hist = 72
    fig, ax = panel(2.0, 1.1)
    th = np.arange(-hist, 0)
    tf = np.arange(r["S"])
    ax.plot(th, r["inv"](r["x"][VAR][-hist:], VAR), color=INK, lw=0.8)
    ax.plot(tf, r["inv"](r["y"][VAR], VAR), color=INK, lw=0.8, label="ground truth")
    ax.plot(tf, r["inv"](r["pred"][VAR], VAR), color=ORANGE, lw=1.1, ls="--", label="DyBAS-Net")
    ax.axvline(0, color=MUTED, lw=0.5, ls=":")
    ax.set_xticks([-72, -48, -24, 0, 24, 48, 72, 96])
    ax.set_xlabel("hour (0 = forecast start)", labelpad=1)
    ax.set_ylabel("Total load (MW)", labelpad=1)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", handlelength=1.6, borderaxespad=0.1, ncol=2)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    save(fig, "P15_forecast_vs_truth")


def draw_intro():
    """Ba hình của Introduction, chữ tiếng Anh, dùng làm hình con (cùng dữ liệu Load)."""
    here = Path.cwd()
    os.chdir(CSV.parent)                       # make_intro_figure.load đọc "load_new.csv" ở thư mục hiện tại
    try:
        seg, weekend, corr, recon, roll = intro_load()
    finally:
        os.chdir(here)
    P = seg.shape[1]
    fig, ax = panel()
    for day, off in zip(seg, weekend):
        ax.plot(range(P), day, color=ORANGE if off else BLUE, lw=0.8 if off else 0.5, alpha=0.9 if off else 0.45)
    ax.plot(range(P), seg.mean(0), color=INK, lw=1.2)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("hour of day", labelpad=1)
    ax.set_ylabel("MW", labelpad=1)
    save(fig, "I01_daily_overlay")

    rec, window, err = recon
    shown = 5
    t = np.arange(shown * P)
    fig, ax = panel(1.9, 1.05)
    ax.plot(t, window[:shown].ravel(), color=INK, lw=0.8, label="actual")
    ax.plot(t, rec[INTRO_R][:shown].ravel(), color=ORANGE, lw=1.0, label=f"from R = {INTRO_R} typical days")
    ax.set_xticks(np.arange(0, shown * P + 1, P), [str(d) for d in range(1, shown + 2)])
    ax.set_xlim(0, shown * P - 1)
    ax.set_ylim(top=window[:shown].max() * 1.3)
    ax.set_xlabel("day", labelpad=1)
    ax.set_ylabel("MW", labelpad=1)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", ncol=2, handlelength=1.4, borderaxespad=0.1)
    save(fig, "I02_low_rank_reconstruction")

    fig, ax = panel(1.9, 1.05)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.fill_between(roll.index, roll.values, 0, where=roll.values >= 0, color=BLUE, alpha=0.45, lw=0,
                    interpolate=True)
    ax.fill_between(roll.index, roll.values, 0, where=roll.values < 0, color=RED, alpha=0.55, lw=0, interpolate=True)
    ax.plot(roll.index, roll.values, color=INK, lw=0.4)
    ax.set_ylim(-1, 1.35)
    ax.set_ylabel("corr.", labelpad=1)
    ax.text(0.01, 0.99, "load vs. temperature (30-day rolling)", transform=ax.transAxes, va="top", fontsize=5.5)
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
    save(fig, "I03_load_temperature_corr")


def sheet():
    """Ghép mọi ảnh thành một tờ có mã (P01, …) để đính kèm một lần."""
    files = sorted(OUT.glob("[PI][0-9][0-9]_*.png"))
    cols = 4
    rows = -(-len(files) // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.0))
    for a in axs.ravel():
        a.axis("off")
    for a, f in zip(axs.ravel(), files):
        a.imshow(plt.imread(f))
        a.set_title(f.stem, fontsize=8, loc="left")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "panel_sheet.png", dpi=150, facecolor="white")
    plt.close(fig)
    return files


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = run_model()
    draw_all(r)
    draw_intro()
    files = sheet()
    print(f"window {r['start']} → forecast {r['fstart']} (+{r['S']} h); MSE (scaled) {r['mse']:.4f}; "
          f"{len(files)} panels in {OUT}")


if __name__ == "__main__":
    main()
