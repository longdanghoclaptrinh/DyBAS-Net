"""Hình 1 của paper (Introduction): hai tính chất của dữ liệu phụ tải Hà Nội (load_new.csv).

(a) 14 đoạn ngày liên tiếp của tổng phụ tải chồng lên nhau, (b) tương quan Pearson giữa các đoạn đó,
(c) xếp L = 720 giờ thành N = 30 đoạn ngày rồi dựng lại chỉ từ R ngày điển hình (xấp xỉ hạng R bằng SVD): R = 6 đã
    gần trùng số liệu thật; sai số tương đối trung bình trên mọi cửa sổ ghi ở chú giải,
(d) tương quan cuộn 30 ngày giữa phụ tải sinh hoạt và hệ số nhiệt độ: cùng một cặp biến lúc tương quan dương mạnh,
lúc âm.
Ví dụ: python scripts/make_intro_figure.py  → paper/figures/intro_load.pdf (1×4, cho paper) và
       paper/figures/intro_load_2x2.png (2×2, chữ to hơn), slide/intro_load.pdf (2×2, đúng khổ beamer)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import YearLocator, DateFormatter

# Màu theo bảng tham chiếu của skill dataviz (xanh/cam là hai slot đầu, xanh↔đỏ là cặp phân kỳ).
BLUE, ORANGE, RED, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#e34948", "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f4f9ff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
START = "2023-03-06"   # thứ Hai, hai tuần làm việc bình thường (không có Tết/nghỉ lễ)
DAYS, P, N, R = 14, 24, 30, 6

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "pdf.fonttype": 42,
                     "mathtext.fontset": "stix"})


def low_rank(m, r):
    """Xấp xỉ hạng r: mỗi ngày là tổ hợp tuyến tính của r "ngày điển hình" (vector kỳ dị phải)."""
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    return (u[:, :r] * s[:r]) @ vt[:r]


def load():
    df = pd.read_csv("data/load_new.csv", parse_dates=["timestamp"]).set_index("timestamp")
    total = df["total_consumption_mw"]
    days = total[START:].iloc[:DAYS * P]
    seg = days.values.reshape(DAYS, P)
    weekend = days.index[::P].dayofweek.values >= 5   # thứ Bảy, Chủ nhật

    v = total.values
    err = {1: [], R: []}
    for s in range(0, len(v) - N * P + 1, P):
        m = v[s:s + N * P].reshape(N, P)
        for r in err:
            err[r].append(np.abs(low_rank(m, r) - m).mean() / m.mean())
    window = total[START:].values[:N * P].reshape(N, P)
    recon = {r: low_rank(window, r) for r in err}, window, {r: float(np.mean(e)) for r, e in err.items()}

    daily = df[["residential_consumption_mw", "temperature_factor"]].resample("D").mean()
    roll = daily.iloc[:, 0].rolling(30).corr(daily.iloc[:, 1]).dropna()
    return seg, weekend, np.corrcoef(seg), recon, roll


def draw(fig, ax, seg, weekend, corr, recon, roll, small):
    """small = 7 cho paper; các cỡ chữ khác tỉ lệ theo."""
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)

    a = ax[0]
    for day, off in zip(seg, weekend):
        a.plot(range(P), day, color=ORANGE if off else BLUE, lw=1.0 if off else 0.7, alpha=0.9 if off else 0.45)
    a.plot(range(P), seg.mean(0), color=INK, lw=1.5)
    a.plot([], [], color=BLUE, lw=1, label="ngày làm việc")
    a.plot([], [], color=ORANGE, lw=1, label="cuối tuần")
    a.plot([], [], color=INK, lw=1.5, label="trung bình")
    a.set_xticks([0, 6, 12, 18, 23])
    a.set_xlabel("Giờ trong ngày")
    a.set_ylabel("Tổng phụ tải (MW)")
    a.set_title("(a) 14 đoạn ngày liên tiếp")
    a.grid(axis="y", color=GRID, lw=0.5)
    a.legend(frameon=False, fontsize=small, loc="upper left", handlelength=1.2, borderaxespad=0.1, labelspacing=0.2)

    a = ax[1]
    im = a.imshow(corr, cmap=SEQ, vmin=0.4, vmax=1, aspect="auto")
    a.set_xticks([0, 6, 13], ["1", "7", "14"])
    a.set_yticks([0, 6, 13], ["1", "7", "14"])
    a.set_xlabel("Ngày")
    a.set_ylabel("Ngày")
    a.set_title("(b) Tương quan Pearson")
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=small - 1, length=2)
    cb.outline.set_linewidth(0.4)

    a = ax[2]
    rec, window, err = recon
    shown = 7   # vẽ 7 ngày đầu của cửa sổ 30 ngày cho dễ nhìn
    t = np.arange(shown * P)
    pct = lambda r: f"{err[r] * 100:.1f}".replace(".", ",")
    a.plot(t, window[:shown].ravel(), color=INK, lw=1.0, label="thực tế")
    a.plot(t, rec[1][:shown].ravel(), color="#a8a7a0", lw=0.9, ls="--", label=f"1 ngày điển hình (sai {pct(1)}%)")
    a.plot(t, rec[R][:shown].ravel(), color=ORANGE, lw=1.1, label=f"{R} ngày điển hình (sai {pct(R)}%)")
    a.set_xticks(np.arange(0, shown * P + 1, P), [str(d) for d in range(1, shown + 2)])
    a.set_xlim(0, shown * P - 1)
    a.set_ylim(top=window[:shown].max() * 1.45)
    a.set_xlabel("Ngày")
    a.set_ylabel("Tổng phụ tải (MW)")
    a.set_title("(c) Dựng lại từ vài ngày điển hình")
    a.legend(frameon=False, fontsize=small, loc="upper left", handlelength=1.4, borderaxespad=0.1, labelspacing=0.2)
    a.grid(axis="y", color=GRID, lw=0.5)

    a = ax[3]
    a.axhline(0, color=MUTED, lw=0.6)
    a.fill_between(roll.index, roll.values, 0, where=roll.values >= 0, color=BLUE, alpha=0.45, lw=0, interpolate=True,
                   label="dương")
    a.fill_between(roll.index, roll.values, 0, where=roll.values < 0, color=RED, alpha=0.55, lw=0, interpolate=True,
                   label="âm")
    a.plot(roll.index, roll.values, color=INK, lw=0.5)
    a.set_ylim(-1, 1.3)
    a.set_yticks([-1, -0.5, 0, 0.5, 1])
    a.set_ylabel("Hệ số tương quan")
    a.xaxis.set_major_locator(YearLocator())
    a.xaxis.set_major_formatter(DateFormatter("%Y"))
    a.set_title("(d) Tương quan phụ tải – nhiệt độ")
    a.legend(frameon=False, fontsize=small, loc="upper center", ncol=2, handlelength=1.2, borderaxespad=0.1)
    a.grid(axis="y", color=GRID, lw=0.5)


def main():
    data = load()
    seg, weekend, corr, recon, roll = data
    iu = np.triu_indices(DAYS, 1)

    fig, ax = plt.subplots(1, 4, figsize=(7.16, 1.95), gridspec_kw={"width_ratios": [0.95, 0.85, 1.3, 1.65]})
    draw(fig, ax, *data, small=7)
    fig.tight_layout(pad=0.3, w_pad=0.9)
    out = Path("paper/figures/intro_load.pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=200)

    with plt.rc_context({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10,
                         "ytick.labelsize": 10}):
        fig, ax = plt.subplots(2, 2, figsize=(10, 6.2))
        draw(fig, ax.ravel(), *data, small=10)
        fig.tight_layout(pad=0.5, h_pad=1.2, w_pad=1.5)
        fig.savefig(out.with_name("intro_load_2x2.png"), dpi=200)

    # Bản cho beamer 4:3 (khung chữ rộng ~12 cm): vẽ đúng kích thước thật để chữ không bị thu nhỏ.
    with plt.rc_context({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5}):
        fig, ax = plt.subplots(2, 2, figsize=(4.75, 3.15))
        draw(fig, ax.ravel(), *data, small=6.5)
        fig.tight_layout(pad=0.2, h_pad=0.6, w_pad=0.8)
        fig.savefig("slide/intro_load.pdf")

    print(out, f"corr mean {corr[iu].mean():.3f} min {corr[iu].min():.3f}; R={R} recon err {recon[2][R]:.3f} (R=1 {recon[2][1]:.3f}); "
          f"rolling corr {roll.min():.2f}..{roll.max():.2f}, share<0 {(roll < 0).mean():.2f}")


if __name__ == "__main__":
    main()
