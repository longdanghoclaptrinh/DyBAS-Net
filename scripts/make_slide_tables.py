"""Sinh các bảng LaTeX cho slide (beamer) từ results/*.json; ghi vào slide/tables/*.tex.

Không chép tay số liệu. Mô hình DyTimeBase trong mã nguồn được gọi là DyBAS-Net trên slide.
Ví dụ: python scripts/make_slide_tables.py
"""

import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_all import ABLATION, DATASETS, HORIZONS, SEEDS  # noqa: E402
from make_report_tables import BRANCH, COLS, load_all  # noqa: E402
from run_suite import TB_INDIVIDUAL, TB_OFFICIAL  # noqa: E402

OUT = Path("slide/tables")
NAME = {"DyTimeBase": r"\textbf{DyBAS-Net}", "TimeBase*": r"TimeBase$^{*}$", "TimeBase": r"TimeBase$^{\dagger}$"}
STRONG = ["DyTimeBase", "TimeBase*", "TimeBase", "SparseTSF", "FITS", "PatchTST", "iTransformer", "DLinear",
          "TimeMixer"]


def name(c):
    return NAME.get(c, c)


def mark(values, v, digits=3):
    ranked = sorted({round(x, digits) for x in values})
    s = f"{v:.{digits}f}"
    if round(v, digits) == ranked[0]:
        return rf"\textbf{{{s}}}"
    return rf"\underline{{{s}}}" if len(ranked) > 1 and round(v, digits) == ranked[1] else s


def mark_color(values, v, digits=3):
    """Đỏ đậm = tốt nhất, xanh dương = thứ hai (so sau khi làm tròn)."""
    ranked = sorted({round(x, digits) for x in values})
    s = f"{v:.{digits}f}"
    if round(v, digits) == ranked[0]:
        return rf"\textcolor{{red}}{{\textbf{{{s}}}}}"
    return rf"\textcolor{{blue}}{{{s}}}" if len(ranked) > 1 and round(v, digits) == ranked[1] else s


def wrap(body, cols, width=r"\textwidth"):
    return "\n".join([rf"\resizebox{{{width}}}{{!}}{{%", rf"\begin{{tabular}}{{{cols}}}", r"\toprule", *body,
                      r"\bottomrule", r"\end{tabular}}"]) + "\n"


def avg_table(table):
    head = [r"\multirow{2}{*}{Mô hình} & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{d}}}" for d in DATASETS) + r"\\",
            "".join(rf"\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(DATASETS))),
            " & " + " & ".join("MSE & MAE" for _ in DATASETS) + r"\\", r"\midrule"]
    rows = []
    cols = [c for c in COLS if c != "TimeBase"]          # chỉ giữ TimeBase đo trên máy này (TimeBase*)
    for c in cols:
        cells = []
        for ds in DATASETS:
            for i in (0, 1):
                cells.append(mark_color([table[(ds, "Avg")][k][i] for k in cols], table[(ds, "Avg")][c][i]))
        rows.append(f"{name(c)} & " + " & ".join(cells) + r"\\")
        if c == "TimeBase*":
            rows.append(r"\midrule")
    return wrap(head + rows, "l" + "cc" * len(DATASETS))


def horizon_table(table, datasets):
    head = ["Dataset & $S$ & " + " & ".join(name(c) for c in STRONG) + r"\\", r"\midrule"]
    rows = []
    for ds in datasets:
        for k, h in enumerate(HORIZONS):
            vals = [table[(ds, h)][c][0] for c in COLS]          # hạng tính trên cả 13 mô hình
            lab = rf"\multirow{{4}}{{*}}{{{ds}}}" if k == 0 else ""
            rows.append(f"{lab} & {h} & " + " & ".join(mark_color(vals, table[(ds, h)][c][0]) for c in STRONG)
                        + r"\\")
        if ds != datasets[-1]:
            rows.append(r"\midrule")
    return wrap(head + rows, "ll" + "c" * len(STRONG))


def h2h_table(table):
    head = [r"Đối thủ & Thắng & Hòa & Thua & DyBAS-Net tốt hơn (MSE TB) & DyBAS-Net kém hơn (MSE TB)\\", r"\midrule"]
    rows = []
    for c in COLS[1:]:
        w = t = l = 0
        for ds in DATASETS:
            for h in HORIZONS:
                a, b = round(table[(ds, h)]["DyTimeBase"][0], 3), round(table[(ds, h)][c][0], 3)
                w, t, l = w + (a < b), t + (a == b), l + (a > b)
        better = [ds for ds in DATASETS if table[(ds, "Avg")]["DyTimeBase"][0] < table[(ds, "Avg")][c][0]]
        worse = [ds for ds in DATASETS if ds not in better]
        rows.append(f"{name(c)} & {w} & {t} & {l} & {', '.join(better) or '--'} & {', '.join(worse) or '--'}" + r"\\")
    return wrap(head + rows, "lcccll")


def versus_table(table):
    head = ["Dataset & " + " & ".join(f"$S={h}$" for h in HORIZONS) + r" & Trung bình\\", r"\midrule"]
    rows = []
    for ds in DATASETS:
        cells = []
        for h in HORIZONS + ("Avg",):
            a, b = table[(ds, h)]["DyTimeBase"][0], table[(ds, h)]["TimeBase*"][0]
            d = 100 * (a - b) / b
            s = f"${d:+.1f}\\%$"
            cells.append(rf"\textcolor{{ForestGreen}}{{{s}}}" if d < 0 else rf"\textcolor{{mqred}}{{{s}}}")
        rows.append(f"{ds} & " + " & ".join(cells) + r"\\")
    return wrap(head + rows, "lccccc", r"0.75\textwidth")


def efficiency_table():
    r = json.loads(Path("results/bench_efficiency.json").read_text(encoding="utf-8"))
    ref = r["DyTimeBase"]["infer_ms_t8"]

    def count(v):
        return f"{v / 1e9:.2f}G" if v >= 1e9 else f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.2f}K"

    def th(v, fmt=",.0f"):
        return format(v, fmt).replace(",", r"\,")
    head = [r"Mô hình & Params & MACs & Max Mem (MB) & Epoch Time (s) & Infer Time (ms) & So với DyBAS-Net\\",
            r"\midrule"]
    rows = []
    for n, x in sorted(r.items(), key=lambda kv: kv[1]["infer_ms_t8"]):
        mem = rf"$>${th(x['mem_over_gb'] * 1024)}$^{{\ddagger}}$" if "mem_over_gb" in x else th(x["max_mem_mb"])
        label = r"\textbf{DyBAS-Net}" if n == "DyTimeBase" else n.replace(" (cũ)", "")
        rows.append(f"{label} & {count(x['params'])} & {count(x['macs'])} & {mem} & {th(x['epoch_s'])} & "
                    f"{th(x['infer_ms_t8'], ',.2f')} & {th(x['infer_ms_t8'] / ref, ',.2f')}$\\times$" + r"\\")
    return wrap(head + rows, "lrrrrrr")


def ablation_table(runs):
    labels = {"DyTimeBase (đầy đủ)": r"\textbf{DyBAS-Net (đầy đủ)}", "bỏ CrossV": r"\quad bỏ CrossVBlock",
              "bỏ IntraV": r"\quad bỏ IntraVBlock",
              "bỏ cả nhánh trộn kênh (= TimeBase\\*)": r"\quad bỏ cả nhánh ($=$ TimeBase$^{*}$)",
              "đầy đủ, mix_lr 2e-3": r"đầy đủ, $\alpha_M = 2\cdot10^{-3}$"}
    vals = {lab: [statistics.mean(runs[ds][f"{v}_S{h}_seed{s}"]["mse"] for h in HORIZONS for s in SEEDS)
                  for ds in DATASETS] for lab, v in ABLATION.items()}
    means = {k: statistics.mean(v) for k, v in vals.items()}
    head = ["Biến thể & " + " & ".join(DATASETS) + r" & Trung bình\\", r"\midrule"]
    rows = []
    for lab, v in vals.items():
        cells = []
        for i, x in enumerate(v + [means[lab]]):
            col = [r[i] for r in vals.values()] if i < len(DATASETS) else list(means.values())
            cells.append(mark_color(col, x, digits=4))
        rows.append(f"{labels[lab]} & " + " & ".join(cells) + r"\\")
        if lab.startswith("bỏ cả"):
            rows.append(r"\midrule")
    return wrap(head + rows, "l" + "c" * (len(DATASETS) + 1))


def hparam_table():
    head = [r" & \multicolumn{3}{c}{Theo $S=96/192/336/720$} & \multicolumn{3}{c}{Nhánh trộn biến}\\",
            r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
            r"Dataset & $P$ & $R$ & Riêng biến & $D$ & $p_1$ & $p_2$\\", r"\midrule"]
    rows = []
    for ds in DATASETS:
        v = [TB_OFFICIAL[ds][int(h)] for h in HORIZONS]
        j = lambda i: "/".join(f"{x[i]:g}" for x in v)  # noqa: E731
        d, p1, p2 = BRANCH[ds]
        rows.append(f"{ds} & {j(3)} & {j(4)} & {'có' if TB_INDIVIDUAL[ds] else 'không'} & {d} & {p1:g} & {p2:g}"
                    + r"\\")
    return wrap(head + rows, "lllcccc")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, table = load_all()
    files = {"avg.tex": avg_table(table), "h_etth.tex": horizon_table(table, ["ETTh1", "ETTh2"]),
             "h_ettm.tex": horizon_table(table, ["ETTm1", "ETTm2"]), "h_weather.tex": horizon_table(table, ["Weather"]),
             "h2h.tex": h2h_table(table), "versus.tex": versus_table(table), "efficiency.tex": efficiency_table(),
             "ablation.tex": ablation_table(runs), "hparams.tex": hparam_table()}
    for n, text in files.items():
        (OUT / n).write_text(text, encoding="utf-8")
    print("wrote", ", ".join(files))


if __name__ == "__main__":
    main()
