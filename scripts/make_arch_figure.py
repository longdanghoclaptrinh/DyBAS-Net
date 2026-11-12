"""Sơ đồ kiến trúc DyBAS-Net (một hình vector), giữ đúng các khối của sơ đồ trong slide.

Mỗi khối gồm tiêu đề, một hình nhỏ vẽ từ đại lượng trung gian THẬT của mô hình (checkpoint Load, L = 720, S = 96,
seed 2021, cùng cửa sổ test với scripts/make_arch_panels.py) và số phương trình tương ứng trong paper (đọc từ paper/DyBAS-Net.aux, nên luôn khớp với bản paper đã biên dịch).
Khổ hình = bề rộng figure* IEEE hai cột (7.16 in), chữ 6–7 pt, nên chèn thẳng vào paper (width=\\textwidth)
và slide (width=\\linewidth) mà không phải co giãn chữ.

Ra: paper/figures/dybas_architecture.pdf (+ .png 300 dpi). Ví dụ: python scripts/make_arch_figure.py
"""

from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_arch_panels import BASIS, BLUE, DIV, INK, MUTED, ORANGE, RED, SEQ, VAR, run_model  # noqa: E402

OUT = Path("paper/figures")
W, H = 7.16, 3.30                                   # inch

STYLE = {  # (nền, viền)
    "op": ("#f4f4f2", "#8f8e89"),                   # phép biến đổi không tham số
    "lin": ("#eaf2fc", "#2a78d6"),                  # lớp tuyến tính chia sẻ giữa các biến (độc lập kênh)
    "mix": ("#fdf1ea", "#eb6834"),                  # nhánh trộn liên biến
}
GREEN = "#2e9e5b"

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 6.5,
                     "axes.linewidth": 0.4, "axes.edgecolor": "#b9b8b3", "savefig.dpi": 300})

fig = plt.figure(figsize=(W, H))
bg = fig.add_axes([0, 0, 1, 1])
bg.set_xlim(0, W)
bg.set_ylim(0, H)
bg.axis("off")


# ---------------------------------------------------------------- hình học

def box(x, y, w, h, title, kind, size=7):
    face, edge = STYLE[kind]
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.045", fc=face, ec=edge, lw=0.8,
                                zorder=2))
    bg.text(x + w / 2, y + h - 0.085, title, ha="center", va="center", fontsize=size, weight="bold", color=INK,
            zorder=3)


AUX = Path("paper/DyBAS-Net.aux")
EQ = {m.group(1): m.group(2)
      for m in re.finditer(r"\\newlabel\{(eq:[^}]+)\}\{\{(\d+)\}", AUX.read_text(encoding="utf-8"))}


def eq(first, last=None):
    """Nhãn tham chiếu 'Eq. (n)' hoặc 'Eq. (m)–(n)' theo label trong paper."""
    return f"Eq. ({EQ[first]})" if last is None else f"Eq. ({EQ[first]})–({EQ[last]})"


def formula(x, y, w, lines, size=6.5):
    for k, s in enumerate(lines):
        bg.text(x + w / 2, y + 0.115 + 0.135 * (len(lines) - 1 - k), s, ha="center", va="center", fontsize=size,
                color=INK, zorder=3)


def shape(x, y, w, s):
    bg.text(x + w / 2, y - 0.075, s, ha="center", va="center", fontsize=5.8, color=MUTED, style="italic")


def inset(x, y, w, h, n_lines, *, pad_x=0.07, top=0.19, xlabel=None, ylabel=None):
    """Trục con trong khối: chừa chỗ cho tiêu đề (trên) và công thức (dưới)."""
    bottom = 0.06 + 0.135 * n_lines + (0.09 if xlabel else 0.0)
    left = pad_x + (0.07 if ylabel else 0.0)
    ax = fig.add_axes([(x + left) / W, (y + bottom) / H, (w - left - pad_x) / W, (h - bottom - top) / H])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.patch.set_alpha(0)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=5.5, labelpad=1, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=5.5, labelpad=1, color=MUTED)
    return ax


def framed(ax):
    for s in ax.spines.values():
        s.set_visible(True)


def arrow(p, q, *, ls="-", color=INK, lw=0.7):
    bg.annotate("", xy=q, xytext=p, zorder=1,
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0))


def path(points, *, ls="-", color=INK, lw=0.7, head=True):
    xs, ys = zip(*points)
    bg.plot(xs[:-1] if head else xs, ys[:-1] if head else ys, color=color, lw=lw, ls=ls, zorder=1,
            solid_capstyle="butt")
    if head:
        arrow(points[-2], points[-1], ls=ls, color=color, lw=lw)


# ---------------------------------------------------------------- dữ liệu thật

r = run_model()
P, N = r["seg"].shape[1:]
R, Np = r["e"].shape[2], r["y_seg"].shape[2]
hours = np.arange(1, P + 1)


def sym(m):
    lim = np.abs(m).max()
    return TwoSlopeNorm(0, -lim, lim)


# ---------------------------------------------------------------- bố cục

BW, GAP, BH, BY = 0.86, 0.10, 1.22, 0.30            # khối hàng dưới
X0 = 0.04
xs = [X0 + k * (BW + GAP) for k in range(4)]        # Input, Segment, Period Norm, Basis
TAP = xs[3] + BW + 0.08                             # điểm rẽ E lên nhánh trộn
OPLUS = TAP + 0.20
xs += [OPLUS + 0.08 + 0.15 + k * (BW + GAP) for k in range(3)]   # Forecasting, Add μ, Forecast
MID = BY + BH / 2

TH, TY = 1.17, 1.92                                 # khối hàng trên; CrossV rộng hơn (công thức có mặt nạ)
TX0, TGAP = 1.20, 0.15
TWS = [1.355] * 4
txs = [TX0 + sum(TWS[:k]) + k * TGAP for k in range(4)]
TW = TWS[0]

# nhóm trộn kênh
gx0, gy0, gx1, gy1 = TX0 - 0.08, TY - 0.09, txs[-1] + TWS[-1] + 0.07, H - 0.03
bg.add_patch(FancyBboxPatch((gx0, gy0), gx1 - gx0, gy1 - gy0, boxstyle="round,pad=0,rounding_size=0.06",
                            fc="#fffaf6", ec=ORANGE, lw=0.8, ls=(0, (4, 2.5)), zorder=0))
bg.text(gx0 + 0.08, gy1 - 0.10, "Channel Mixing in Basis Space", ha="left", va="center", fontsize=7.5,
        weight="bold", color=ORANGE)

# ---------------------------------------------------------------- hàng dưới: nhánh độc lập kênh

# 1. Input
x = xs[0]
box(x, BY, BW, BH, "Input", "op")
formula(x, BY, BW, [eq("eq:input")])
ax = inset(x, BY, BW, BH, 1, xlabel=r"time $t$")
week = 7 * P
for k, (c, col) in enumerate(zip((0, 2, 4), (BLUE, ORANGE, RED))):
    s = r["x"][c][-week:]
    s = (s - s.min()) / (np.ptp(s) + 1e-9)
    ax.plot(s + 1.3 * (2 - k), color=col, lw=0.5)
for k, (name, col) in enumerate(zip((r"$x^{(1)}$", r"$x^{(2)}$", r"$x^{(C)}$"), (BLUE, ORANGE, RED))):
    ax.text(-4, 1.3 * (2 - k) + 0.5, name, fontsize=5.5, color=col, va="center", ha="right")
ax.set_xlim(-62, week)
ax.set_ylim(-0.1, 3.7)

# 2. Segment
x = xs[1]
box(x, BY, BW, BH, "Segment", "op")
formula(x, BY, BW, [eq("eq:segment")])
ax = inset(x, BY, BW, BH, 1, xlabel=r"segment $n$", ylabel=r"phase $p$")
ax.imshow(r["seg_raw"][VAR], cmap=SEQ, aspect="auto", interpolation="nearest")
framed(ax)

# 3. Period Norm
x = xs[2]
box(x, BY, BW, BH, "Period Norm", "op")
formula(x, BY, BW, [eq("eq:phase-mean", "eq:period-norm")])
ax = inset(x, BY, BW, BH, 1, xlabel=r"phase $p$")
for n in range(N):
    ax.plot(hours, r["seg_raw"][VAR][:, n], color=BLUE, lw=0.35, alpha=0.3)
ax.plot(hours, r["mean"][VAR][:, 0], color=INK, lw=1.1)
ax.text(0.02, 0.98, r"$\mu_p$", transform=ax.transAxes, va="top", fontsize=6.5)

# 4. Basis Extraction
x = xs[3]
box(x, BY, BW, BH, "Basis Extraction", "lin", size=6.8)
formula(x, BY, BW, [eq("eq:basis")])
ax = inset(x, BY, BW, BH, 1, xlabel=r"phase $p$, $R$ bases")
for k in range(R):
    ax.plot(hours, r["e"][VAR][:, k], color=BASIS[k], lw=0.7)

# ⊕
bg.add_patch(Circle((OPLUS, MID), 0.08, fc="white", ec=INK, lw=0.8, zorder=3))
bg.plot([OPLUS - 0.045, OPLUS + 0.045], [MID, MID], color=INK, lw=0.8, zorder=4)
bg.plot([OPLUS, OPLUS], [MID - 0.045, MID + 0.045], color=INK, lw=0.8, zorder=4)
bg.text(OPLUS, MID - 0.17, eq("eq:correct"), ha="center", va="center", fontsize=6.0)

# 5. Segment Forecasting
x = xs[4]
box(x, BY, BW, BH, "Segment Forecasting", "lin", size=6.3)
formula(x, BY, BW, [eq("eq:forecast")])
ax = inset(x, BY, BW, BH, 1, xlabel=r"phase $p$, $N'$ segments")
for n in range(Np):
    ax.plot(hours, r["y_seg"][VAR][:, n], color=ORANGE, lw=0.7, alpha=0.4 + 0.6 * n / max(Np - 1, 1))

# 6. Add μ_p & Flatten
x = xs[5]
box(x, BY, BW, BH, r"Add $\mu_p$ & Flatten", "op")
formula(x, BY, BW, [eq("eq:merge")])
ax = inset(x, BY, BW, BH, 1, xlabel=r"step $t=1,\dots,S$")
ax.plot(r["y_seg"][VAR].T.ravel(), color="#a9a8a3", lw=0.6)
ax.plot((r["y_seg"][VAR] + r["mean"][VAR]).T.ravel(), color=ORANGE, lw=0.8)
ax.text(0.98, 0.02, r"$+\mu_p$", transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color=ORANGE)

# 7. Forecast
x = xs[6]
box(x, BY, BW, BH, r"Forecast $\hat{\mathcal{Y}}$", "op")
ax = inset(x, BY, BW, BH, 0, xlabel=r"history | horizon $S$")
hist = 72
ax.plot(np.arange(-hist, 0), r["x"][VAR][-hist:], color=INK, lw=0.5)
ax.plot(np.arange(r["S"]), r["y"][VAR], color=INK, lw=0.5)
ax.plot(np.arange(r["S"]), r["pred"][VAR], color=ORANGE, lw=0.8, ls=(0, (2.5, 1)))
ax.axvline(-0.5, color=MUTED, lw=0.4, ls=":")
lo, hi = ax.get_ylim()
ax.set_ylim(lo, hi + 0.3 * (hi - lo))
ax.plot([], [], color=INK, lw=0.5, label="truth")
ax.plot([], [], color=ORANGE, lw=0.8, ls=(0, (2.5, 1)), label="pred.")
ax.legend(frameon=False, fontsize=4.8, loc="upper left", ncol=2, handlelength=1.3, columnspacing=0.6,
          borderaxespad=0, handletextpad=0.3)

# mũi tên hàng dưới
for a, b in zip(xs[:3], xs[1:4]):
    arrow((a + BW, MID), (b, MID))
bg.plot([xs[3] + BW, OPLUS - 0.08], [MID, MID], color=INK, lw=0.7, zorder=1)
bg.add_patch(Circle((TAP, MID), 0.018, color=INK, zorder=3))
arrow((OPLUS + 0.08, MID), (xs[4], MID))
for a, b in zip(xs[4:6], xs[5:7]):
    arrow((a + BW, MID), (b, MID))

# μ_p bỏ qua (nét đứt) từ Period Norm tới Add μ_p
yb = 0.07
ux0, ux1 = xs[2] + 0.80 * BW, xs[5] + 0.80 * BW
path([(ux0, BY), (ux0, yb), (ux1, yb), (ux1, BY)], ls=(0, (3, 2)), color=MUTED, lw=0.6)
bg.text((ux0 + ux1) / 2, yb + 0.06, r"$+\boldsymbol{\mu}^{(i)}$  (phase mean, skip)", ha="center", va="center",
        fontsize=6, color=MUTED)

# ---------------------------------------------------------------- hàng trên: trộn liên biến

var_note = dict(fontsize=5.5, labelpad=1, color=MUTED)

# 8. Flatten + Embed
x, TW = txs[0], TWS[0]
box(x, TY, TW, TH, "Flatten + Embed", "mix")
formula(x, TY, TW, [eq("eq:flatten", "eq:embed")])
ax = inset(x, TY, TW, TH, 1, pad_x=0.18, xlabel=r"variable ($C$)", ylabel=r"feature ($D$)")
ax.imshow(r["h_enc"].T, cmap=DIV, norm=sym(r["h_enc"]), aspect="auto", interpolation="nearest")
framed(ax)

# 9. CrossV Block
x, TW = txs[1], TWS[1]
box(x, TY, TW, TH, "CrossV Block", "mix")
formula(x, TY, TW, [eq("eq:cross1", "eq:dropfusion")])
ax = inset(x, TY, TW, TH, 1, pad_x=0.40, top=0.19, xlabel="to variable", ylabel="from var.")
mix = r["W1"] @ r["W2"]
ax.imshow(mix, cmap=DIV, norm=sym(mix), aspect="equal", interpolation="nearest")
framed(ax)

# 10. IntraV Block
x, TW = txs[2], TWS[2]
box(x, TY, TW, TH, "IntraV Block", "mix")
formula(x, TY, TW, [eq("eq:cross-res", "eq:intra")])
ax = inset(x, TY, TW, TH, 1, pad_x=0.10, xlabel=r"feature $d$ (one variable)")
ax.plot(r["x_intra"][VAR], color="#a9a8a3", lw=0.6)
ax.plot(r["x_hat"][VAR], color=ORANGE, lw=0.7)
ax.text(0.01, 0.98, r"$\mathcal{X}_{intra}$", transform=ax.transAxes, va="top", fontsize=5.5, color=MUTED)
ax.text(0.99, 0.98, r"$\hat{\mathcal{X}}_{intra}$", transform=ax.transAxes, va="top", ha="right", fontsize=5.5,
        color=ORANGE)
lo, hi = ax.get_ylim()
ax.set_ylim(lo, hi + 0.25 * (hi - lo))

# 11. Projection
x, TW = txs[3], TWS[3]
box(x, TY, TW, TH, "Projection (zero-init)", "mix")
formula(x, TY, TW, [eq("eq:proj")])
ax = inset(x, TY, TW, TH, 1, pad_x=0.10, xlabel=r"phase $p$")
ax.axhline(0, color="#c9c8c3", lw=0.4)
for k in range(R):
    ax.plot(hours, r["de"][VAR][:, k], color=BASIS[k], lw=0.7)
ax.text(0.01, 0.98, r"$\Delta\mathbf{E}$", transform=ax.transAxes, va="top", fontsize=6.5)

# mũi tên hàng trên
TMID = TY + TH / 2
for a, w, b in zip(txs[:3], TWS, txs[1:]):
    arrow((a + w, TMID), (b, TMID))

# E lên nhánh trộn, ΔE về ⊕
YB = (BY + BH + gy0) / 2 + 0.02
ex = txs[0] + 0.12
path([(TAP, MID), (TAP, YB), (ex, YB), (ex, TY)])
bg.text((TAP + ex) / 2, YB - 0.075, r"$\mathbf{E}$", ha="center", va="center", fontsize=7)
dx = txs[3] + TWS[3] - 0.15
YD = YB + 0.07
path([(dx, TY), (dx, YD), (OPLUS, YD), (OPLUS, MID + 0.08)])
bg.text(OPLUS + 0.05, (MID + YD) / 2 + 0.05, r"$\Delta\mathbf{E}$", ha="left", va="center", fontsize=7)

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"dybas_architecture.{ext}", facecolor="white")
print("saved", OUT / "dybas_architecture.pdf")
