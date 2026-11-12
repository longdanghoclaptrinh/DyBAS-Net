"""Đồ thị đánh đổi độ chính xác – tốc độ cho slide: mỗi bộ dữ liệu một hình.

Trục hoành: MSE trung bình 4 horizon của bộ dữ liệu đó (slide/tables/avg.tex, slide/tables/new_load.tex).
Trục tung: Infer Time (ms, thang log) đo một lần ở cấu hình Electricity (slide/tables/efficiency.tex), nên mọi hình
dùng chung một giá trị thời gian cho mỗi mô hình. Đọc thẳng từ các bảng đang hiển thị để hình luôn khớp với bảng.
Đường đứt nét: biên Pareto (không mô hình nào vừa có MSE thấp hơn vừa nhanh hơn).
Mô hình có MSE quá lớn (> 1.4 × MSE tốt nhất) được đặt ở mép phải, kèm giá trị thật.

Ra: slide/tradeoff/<dataset>.pdf. Ví dụ: python scripts/make_tradeoff_plots.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

TABLES = Path("slide/tables")
OUT = Path("slide/tradeoff")
OURS = "DyBAS-Net"
ACCENT, INK, MUTED, GRID, BASE = "#d4442f", "#1f1e1c", "#6b6a66", "#e7e6e2", "#8f8e89"
CLIP = 1.4

plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.edgecolor": "#b9b8b3", "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.dpi": 300})


def rows(name):
    """Các dòng dữ liệu của một bảng .tex: (tên mô hình, [ô...]) với ô đã bỏ định dạng."""
    out = []
    for line in (TABLES / name).read_text(encoding="utf-8").splitlines():
        if "&" not in line or line.lstrip().startswith(("\\multirow", "&", "Mô hình", "\\cmidrule")):
            continue
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        model = re.sub(r"~?\\cite\{[^}]*\}", "", cells[0])
        model = re.sub(r"\\textbf\{([^}]*)\}", r"\1", model).strip()
        nums = []
        for c in cells[1:]:
            c = c.replace(r"\,", "")
            m = re.search(r"\d+(?:\.\d+)?", re.sub(r"\\textcolor\[HTML\]\{[0-9A-F]+\}", "", c))
            nums.append(float(m.group(0)) if m else np.nan)
        out.append((model, nums))
    return out


def load():
    infer = {m: v[3] for m, v in rows("efficiency.tex")}
    mse = {}
    for m, v in rows("avg.tex"):
        for k, ds in enumerate(["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]):
            mse.setdefault(ds, {})[m] = v[2 * k]
    for m, v in rows("new_load.tex"):
        for k, ds in enumerate(["Tetouan", "Panama"]):
            mse.setdefault(ds, {})[m] = v[2 * k]
    return mse, infer


def pareto(points):
    """Các điểm không bị trội (MSE và thời gian đều nhỏ là tốt), sắp theo MSE."""
    pts = sorted(points)
    front, best_t = [], np.inf
    for x, t in pts:
        if t < best_t:
            front.append((x, t))
            best_t = t
    return front


def _candidates(away=None):
    """Vị trí thử (dx, dy theo point): gần trước, xa sau; mỗi vòng ưu tiên bên phải, hoặc ưu tiên hướng `away`
    (góc, độ) nếu có, tức hướng ra xa điểm lân cận gần nhất."""
    angles = [0, -30, 30, -60, 60, 180, 150, -150, 120, -120, 90, -90]
    if away is not None:
        angles = sorted(angles, key=lambda a: abs((a - away + 180) % 360 - 180))
    return [(r * np.cos(np.radians(a)), r * np.sin(np.radians(a))) for r in (7, 13, 20, 28, 38, 50) for a in angles]


def _dist(box, pt):
    """Khoảng cách từ điểm tới hình chữ nhật (0 nếu nằm trong)."""
    dx = max(box[0] - pt[0], 0, pt[0] - box[2])
    dy = max(box[1] - pt[1], 0, pt[1] - box[3])
    return float(np.hypot(dx, dy))


def place_labels(ax, items, fig, taken=(), extra_points=()):
    """Đặt nhãn tham lam, hai vòng. Vòng 1: nằm trong khung, không đè nhãn/điểm/vùng đã chiếm, và gần điểm của chính
    nó hơn hẳn mọi điểm khác. Vòng 2 (cụm quá dày): bỏ điều kiện gần nhất, bù bằng đường dẫn tới điểm."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(renderer)
    pts = [tuple(ax.transData.transform((x, y))) for _, x, y, _ in items] + list(extra_points)
    point_boxes = [(px - 5, py - 5, px + 5, py + 5) for px, py in pts]
    labels = list(taken)

    def overlaps(box, others):
        return any(not (box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3]) for o in others)

    for k, ((name, x, y, style), own) in enumerate(zip(items, pts)):
        others = [q for j, q in enumerate(pts) if j != k]
        blockers = labels + [b for j, b in enumerate(point_boxes) if j != k]
        chosen = None
        away = None
        if name == OURS and others:   # nhãn mô hình đề xuất: ưu tiên hướng ra xa điểm gần nhất
            q = min(others, key=lambda q: np.hypot(q[0] - own[0], q[1] - own[1]))
            away = float(np.degrees(np.arctan2(own[1] - q[1], own[0] - q[0])))
        for strict in (True, False):
            for dx, dy in _candidates(away):
                ha = "left" if dx > 1 else "right" if dx < -1 else "center"
                va = "bottom" if dy > 1 else "top" if dy < -1 else "center"
                t = ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, **style)
                bb = t.get_window_extent(renderer)
                t.remove()
                box = (bb.x0 - 1.5, bb.y0 - 1.5, bb.x1 + 1.5, bb.y1 + 1.5)
                if not (box[0] >= ax_bb.x0 and box[2] <= ax_bb.x1 and box[1] >= ax_bb.y0 and box[3] <= ax_bb.y1):
                    continue
                if overlaps(box, blockers) or overlaps(box, [point_boxes[k]]):
                    continue
                if strict and any(_dist(box, q) < _dist(box, own) + 3 for q in others):
                    continue
                chosen = (dx, dy, ha, va, box, not strict)
                break
            if chosen:
                break
        if chosen is None:
            chosen = (7, 0, "left", "center", None, True)
        dx, dy, ha, va, box, forced = chosen
        lead = forced or np.hypot(dx, dy) > 10 or name == OURS
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    arrowprops=dict(arrowstyle="-", color="#8f8e89", lw=0.6, shrinkA=1.5, shrinkB=3.5)
                    if lead else None, **style)
        if box:
            labels.append(box)


def plot(ds, mse, infer):
    models = [m for m in mse if m in infer and not np.isnan(mse[m])]
    best = min(mse[m] for m in models)
    xmax = best * CLIP
    inside = [m for m in models if mse[m] <= xmax]
    outside = sorted((m for m in models if mse[m] > xmax), key=lambda m: infer[m])
    xmin = best - 0.22 * (xmax - best)
    xs = [mse[m] for m in inside]
    right = max(xs) + 0.10 * (max(xs) - xmin)

    fig, ax = plt.subplots(figsize=(4.6, 2.75))
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    front = pareto([(mse[m], infer[m]) for m in models])

    for m in inside:
        ours = m == OURS
        ax.scatter(mse[m], infer[m], s=95 if ours else 24, color=ACCENT if ours else BASE, zorder=4 if ours else 3,
                   edgecolor="white", linewidth=0.8, marker="*" if ours else "o")
    edge = right
    for m in outside:
        ax.scatter(edge, infer[m], s=22, color=BASE, marker=">", edgecolor="white", linewidth=0.6, zorder=3,
                   clip_on=False)

    ax.set_xlim(xmin, right)
    lo = min(infer[m] for m in models) / 6.0
    hi = max(infer[m] for m in models) * 2.2
    ax.set_ylim(lo, hi)
    ticks = [t for t in (1, 3, 10, 30, 100, 300, 1000, 3000, 10000) if lo <= t <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticklabels([f"{t:,}".replace(",", "\u2009") for t in ticks])
    ax.set_xlabel(f"MSE trung bình 4 horizon trên {ds}  (càng nhỏ càng chính xác)", color=INK)
    ax.set_ylabel("Infer Time (ms, log)\n(càng nhỏ càng nhanh)", color=INK)

    base = dict(fontsize=6.6, color=MUTED)
    items = [(OURS, mse[OURS], infer[OURS], dict(fontsize=7.6, weight="bold", color=INK))]
    items += [(m, mse[m], infer[m], base) for m in inside if m != OURS]
    fig.tight_layout(pad=0.3)
    taken = []
    if outside:   # mô hình ngoài khung: dấu ▶ ở mép phải, tên và MSE gom vào một ghi chú
        listing = ", ".join(f"{m} {mse[m]:.2f}" for m in sorted(outside, key=lambda m: -infer[m]))
        note = ax.text(0.01, 0.99, r"$\blacktriangleright$" + f" ngoài khung (MSE): {listing}", transform=ax.transAxes, ha="left",
                       va="top", fontsize=6.2, color=MUTED)
        fig.canvas.draw()
        bb = note.get_window_extent(fig.canvas.get_renderer())
        taken.append((bb.x0 - 2, bb.y0 - 2, bb.x1 + 2, bb.y1 + 2))
    place_labels(ax, items, fig, taken, [tuple(ax.transData.transform((edge, infer[m]))) for m in outside])
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{ds}.pdf")
    fig.savefig(OUT / f"{ds}.png", dpi=200)
    plt.close(fig)
    on_front = OURS in {m for m in models if (mse[m], infer[m]) in front}
    print(f"{ds}: {len(inside)} trong khung, {len(outside)} ngoài khung, DyBAS-Net trên biên Pareto: {on_front}")


def main():
    mse, infer = load()
    for ds in mse:
        plot(ds, mse[ds], infer)


if __name__ == "__main__":
    main()
