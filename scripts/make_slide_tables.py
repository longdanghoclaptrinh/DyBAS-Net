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
from compare_all import OURS, PUBLISHED, variant  # noqa: E402
from make_report_tables import BRANCH, COLS, INFER, load_all  # noqa: E402
from run_suite import TB_OFFICIAL  # noqa: E402
from run_fix import TUNED as FIX_TUNED  # noqa: E402

OUT = Path("slide/tables")
SHOWN = [c for c in COLS if c != "TimeBase"]   # chỉ giữ TimeBase đo trên máy này (TimeBase*)
# Tự chạy bằng code chính thức trên GPU (scripts/run_external.py → results/external_<bộ>_L720.json).
EXTERNAL = {"MS-FLOW": "MSFLOW", "GTR": "GTR"}
# Bảng độ chính xác của slide: MS-FLOW thay DyTimeNet; GTR (tự chạy) thay DLinear (số công bố). DUET, LightGTS (chạy dở) và TimeEmb đã chạy (results/external_*), không đưa vào.
SELF = ["DyTimeBase", "TimeBase*", "MS-FLOW", "CycleNet/Linear", "CycleNet/MLP", "TQNet", "GTR"]
PUB = [c for c in PUBLISHED if c not in ("TimeBase", "DLinear")]   # SparseTSF: số công bố (bản tự chạy sparsetsf_off giữ trong results/suite_*, không đưa vào)
SHORT = {}   # giữ tên như bài CycleNet: CycleNet/Linear, CycleNet/MLP
NAME = {"DyTimeBase": r"\textbf{DyBAS-Net}", "TimeBase*": r"TimeBase", "TimeBase": r"TimeBase$^{\dagger}$", **SHORT}
STRONG = ["DyTimeBase", "TimeBase*", "CycleNet/Linear", "CycleNet/MLP", "TQNet", "SparseTSF", "FITS", "PatchTST",
          "iTransformer", "DLinear", "TimeMixer"]
RED, BLUE = "FF0000", "0000FF"   # tốt nhất / thứ hai (bảng μ ± σ: xanh = chưa phân biệt được với tốt nhất)
# Số công bố (results/published_timebase_table10.json, Table 10 của TimeBase): trích dẫn bài gốc cạnh tên mô hình.
CITE = {"TimeBase": "Huang2025TimeBase", "SparseTSF": "Lin2024SparseTSF", "TimeMixer": "Wang2024TimeMixer",
        "FITS": "Xu2024FITS", "iTransformer": "Liu2024iTransformer", "DLinear": "Zeng2023DLinear",
        "PatchTST": "Nie2023PatchTST", "TimesNet": "Wu2023TimesNet", "FEDformer": "Zhou2022FEDformer",
        "Autoformer": "Wu2021Autoformer", "Informer": "Zhou2021Informer"}
EXTRA = ("Load",)   # bộ dữ liệu thêm (mọi mô hình tự chạy), chỉ có ở bảng siêu tham số và bảng riêng


def name(c, cite=False):
    return NAME.get(c, c) + (rf"~\cite{{{CITE[c]}}}" if cite and c in CITE else "")


def seed_stats(runs, v, hs):
    """[(μ, σ) của MSE, (μ, σ) của MAE]: trung bình trên các horizon hs của từng seed, rồi μ, σ qua 3 seed."""
    per_seed = [[statistics.mean(runs[f"{v}_S{h}_seed{s}"][m] for h in hs) for s in SEEDS] for m in ("mse", "mae")]
    return [(statistics.mean(x), statistics.stdev(x)) for x in per_seed]


def bench_stats(runs, table):
    """(dataset, horizon hoặc "Avg") → mô hình → [(μ, σ) MSE, (μ, σ) MAE]; số công bố có σ = None."""
    out = {}
    for ds in DATASETS:
        ext = json.loads(Path(f"results/external_{ds}_L720.json").read_text(encoding="utf-8"))
        for key in HORIZONS + ("Avg",):
            hs = HORIZONS if key == "Avg" else (key,)
            out[(ds, key)] = {**{c: seed_stats(ext, EXTERNAL[c], hs) if c in EXTERNAL else
                                 seed_stats(runs[ds], variant(ds, c), hs) for c in SELF},
                              **{c: [(table[(ds, key)][c][i], None) for i in (0, 1)] for c in PUB}}
    return out


def overlap(a, b):
    """Khoảng [μ − σ, μ + σ] của a và b chồng lấn (σ = None: số công bố, một giá trị)."""
    return a[0] - (a[1] or 0) <= b[0] + (b[1] or 0) and b[0] - (b[1] or 0) <= a[0] + (a[1] or 0)


def compare(a, b):
    """−1: a tốt hơn b rõ ràng (khoảng μ ± σ không chồng lấn, μ_a nhỏ hơn); 1: kém hơn; 0: chưa phân biệt được."""
    return 0 if overlap(a, b) else (-1 if a[0] < b[0] else 1)


def mark_std(cells, c, digits=3):
    """μ ± σ; đỏ đậm = μ thấp nhất, xanh = khoảng μ ± σ chồng lấn với khoảng của mô hình tốt nhất."""
    best = min(cells, key=lambda k: cells[k][0])
    m, sd = cells[c]
    s = f"{m:.{digits}f}" + (rf"{{\tiny$\pm${sd:.{digits}f}}}" if sd is not None else "")
    if c == best:
        return rf"\textcolor[HTML]{{{RED}}}{{\textbf{{{s}}}}}"
    return rf"\textcolor[HTML]{{{BLUE}}}{{{s}}}" if overlap(cells[c], cells[best]) else s


def mark(values, v, digits=3):
    ranked = sorted({round(x, digits) for x in values})
    s = f"{v:.{digits}f}"
    if round(v, digits) == ranked[0]:
        return rf"\textbf{{{s}}}"
    return rf"\underline{{{s}}}" if len(ranked) > 1 and round(v, digits) == ranked[1] else s


def mark_color(values, v, digits=3, third=False):
    """Đỏ đậm = tốt nhất, xanh dương = thứ hai, (tùy chọn) cam (Chocolate) = thứ ba (so sau khi làm tròn)."""
    ranked = sorted({round(x, digits) for x in values})
    s = f"{v:.{digits}f}"
    if round(v, digits) == ranked[0]:
        return rf"\textcolor[HTML]{{{RED}}}{{\textbf{{{s}}}}}"
    if len(ranked) > 1 and round(v, digits) == ranked[1]:
        return rf"\textcolor[HTML]{{{BLUE}}}{{\textbf{{{s}}}}}"
    if third and len(ranked) > 2 and round(v, digits) == ranked[2]:
        return rf"\textcolor{{Chocolate}}{{{s}}}"
    return s


def wrap(body, cols, width=r"\textwidth"):
    return "\n".join([rf"\resizebox{{{width}}}{{!}}{{%", rf"\begin{{tabular}}{{{cols}}}", r"\toprule", *body,
                      r"\bottomrule", r"\end{tabular}}"]) + "\n"


def avg_table(stats):
    head = [r"\multirow{2}{*}{Mô hình} & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{d}}}" for d in DATASETS) + r"\\",
            "".join(rf"\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(DATASETS))),
            " & " + " & ".join("MSE & MAE" for _ in DATASETS) + r"\\", r"\midrule"]
    rows = []
    cols = SELF + PUB
    for c in cols:
        cells = []
        for ds in DATASETS:
            for i in (0, 1):
                cells.append(mark_std({k: stats[(ds, "Avg")][k][i] for k in cols}, c))
        rows.append(f"{name(c, cite=c in PUBLISHED)} & " + " & ".join(cells) + r"\\")
        if c == SELF[-1]:                    # hết nhóm tự chạy, tiếp theo là số công bố
            rows.append(r"\midrule")
    return wrap(head + rows, "l" + "cc" * len(DATASETS))


def horizon_table(table, datasets):
    head = ["Dataset & $S$ & " + " & ".join(name(c) for c in STRONG) + r"\\", r"\midrule"]
    rows = []
    for ds in datasets:
        for k, h in enumerate(HORIZONS):
            vals = [table[(ds, h)][c][0] for c in SHOWN]         # hạng trên mọi mô hình trừ TimeBase công bố
            lab = rf"\multirow{{4}}{{*}}{{{ds}}}" if k == 0 else ""
            rows.append(f"{lab} & {h} & " + " & ".join(mark_color(vals, table[(ds, h)][c][0]) for c in STRONG)
                        + r"\\")
        if ds != datasets[-1]:
            rows.append(r"\midrule")
    return wrap(head + rows, "ll" + "c" * len(STRONG))


def h2h_table(stats):
    """Thắng/hòa/thua theo MSE với tiêu chuẩn compare (μ ± σ qua 3 seed); 20 ô = 5 dataset × 4 horizon."""
    head = [r"Đối thủ & Thắng & Hòa & Thua & DyBAS-Net tốt hơn & Chưa phân biệt & DyBAS-Net kém hơn\\",
            r" & \multicolumn{3}{c}{(20 ô dataset $\times$ $S$)} & \multicolumn{3}{c}{(MSE trung bình 4 horizon)}\\",
            r"\midrule"]
    rows = []
    for c in (SELF + PUB)[1:]:
        res = [compare(stats[(ds, h)]["DyTimeBase"][0], stats[(ds, h)][c][0]) for ds in DATASETS for h in HORIZONS]
        avg = {ds: compare(stats[(ds, "Avg")]["DyTimeBase"][0], stats[(ds, "Avg")][c][0]) for ds in DATASETS}
        lists = [", ".join(ds for ds in DATASETS if avg[ds] == k) or "--" for k in (-1, 0, 1)]
        rows.append(f"{name(c, cite=c in PUBLISHED)} & {res.count(-1)} & {res.count(0)} & {res.count(1)} & "
                    + " & ".join(lists) + r"\\")
    return wrap(head + rows, "lccclll")


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
    ref = r["DyTimeBase"][INFER]

    def count(v):
        return f"{v / 1e9:.2f}G" if v >= 1e9 else f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.2f}K"

    def th(v, fmt=",.0f"):
        return format(v, fmt).replace(",", r"\,")
    head = [r"Mô hình & Params & MACs & Epoch Time (s) & Infer Time (ms) & So với DyBAS-Net\\",
            r"\midrule"]
    rows = []
    # MS-FLOW, GTR thay DyTimeNet, DLinear như các bảng độ chính xác (số đo vẫn giữ trong results/bench_efficiency.json).
    for n, x in sorted(((n, x) for n, x in r.items() if n not in ("DyTimeNet (cũ)", "DLinear")),
                       key=lambda kv: kv[1][INFER]):
        epoch = th(x["epoch_s"]) + (r"$^{\ddagger}$" if "mem_over_gb" in x else "")
        label = r"\textbf{DyBAS-Net}" if n == "DyTimeBase" else SHORT.get(n, n.replace(" (cũ)", ""))
        rows.append(f"{label} & {count(x['params'])} & {count(x['macs'])} & {epoch} & "
                    f"{th(x[INFER], ',.2f')} & {th(x[INFER] / ref, ',.2f')}$\\times$" + r"\\")
    return wrap(head + rows, "lrrrrr")


def ablation_table(runs):
    labels = {"DyTimeBase (đầy đủ)": r"\textbf{DyBAS-Net (đầy đủ)}", "bỏ CrossV": r"\quad bỏ CrossVBlock",
              "bỏ IntraV": r"\quad bỏ IntraVBlock",
              "bỏ cả nhánh trộn kênh (= TimeBase\\*)": r"\quad bỏ cả nhánh ($=$ TimeBase)",
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
    head = [r" & \multicolumn{2}{c}{Theo $S=96/192/336/720$} & \multicolumn{2}{c}{Nhánh trộn biến}\\",
            r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
            r"Dataset & $P$ & $R$ & $p_1$ & $p_2$\\", r"\midrule"]   # D = 64 cho mọi dataset; individual chỉ Weather
    rows = []
    for ds in list(DATASETS) + list(NEW_LOAD):   # ba bộ mới: run_suite.PR_TUNED (Load không đưa vào slide)
        v = [TB_OFFICIAL[ds][int(h)] for h in HORIZONS]
        j = lambda i: "/".join(f"{x[i]:g}" for x in v)  # noqa: E731
        _, p1, p2 = BRANCH.get(ds) or (64, *FIX_TUNED[ds])
        rows.append(f"{ds} & {j(3)} & {j(4)} & {p1:g} & {p2:g}" + r"\\")
    return wrap(head + rows, "lllcc")


# Bộ phụ tải điện thực tế (load_new.csv): mọi mô hình tự chạy, 3 seed, siêu tham số chính thức của ETTh1 (cùng tần
# số giờ). Pipeline của repo: results/suite_Load_L720*.json, results/fix_Load_L720.json; TSLib và FITS:
# results/external_Load_L720.json (scripts/run_external.py).
LOAD_FILES = ["results/suite_Load_L720.json", "results/suite_Load_L720_b.json", "results/fix_Load_L720.json",
              "results/external_Load_L720.json"]
LOAD_MODELS = {"DyTimeBase": "dytbase_2stage",   # như compare_all.TUNED
               "TimeBase*": "timebase_off", "CycleNet/Linear": "cyclenet_off",
               "CycleNet/MLP": "cyclenet_mlp_off", "TQNet": "tqnet_off", "SparseTSF": "sparsetsf_off", "FITS": "FITS",
               "DLinear": "DLinear", "iTransformer": "iTransformer"}


def load_table():
    runs = {}
    for f in LOAD_FILES:
        runs.update(json.loads(Path(f).read_text(encoding="utf-8")))
    hs = [int(h) for h in HORIZONS]
    val = {c: {h: [statistics.mean(runs[f"{v}_S{h}_seed{s}"][m] for s in SEEDS) for m in ("mse", "mae")] for h in hs}
           for c, v in LOAD_MODELS.items()}
    for c in val:
        val[c]["Avg"] = [statistics.mean(val[c][h][i] for h in hs) for i in (0, 1)]
    cols = hs + ["Avg"]
    label = {h: f"$S={h}$" for h in hs} | {"Avg": "Trung bình"}
    head = [r"\multirow{2}{*}{Mô hình} & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{label[h]}}}" for h in cols) + r"\\",
            "".join(rf"\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(cols))),
            " & " + " & ".join("MSE & MAE" for _ in cols) + r"\\", r"\midrule"]
    rows = [f"{name(c)} & " + " & ".join(mark_color([val[k][h][i] for k in val], val[c][h][i]) for h in cols for i in (0, 1))
            + r"\\" for c in sorted(val, key=lambda c: val[c]["Avg"][0])]   # MSE trung bình tăng dần
    return wrap(head + rows, "l" + "cc" * len(cols))


# Ba bộ phụ tải đa biến công khai (scripts/prepare_load_data.py): cùng tập mô hình như bảng Load, 3 seed; P, R theo
# run_suite.PR_TUNED, p1/p2 theo run_fix.TUNED.
NEW_LOAD = ("Tetouan", "Panama")   # PJM: đã chạy (results/*_PJM_L720.json) nhưng không đưa vào slide
NEW_LOAD_MODELS = {**{c: v for c, v in LOAD_MODELS.items() if c != "DLinear"}, **EXTERNAL,
                   "PatchTST": "PatchTST", "TimeMixer": "TimeMixer"}   # TSLib, script ETTh1 (scripts/run_tslib_load.sh); TimeMixer/Panama: lr 1e-3 (run_external.RUN_OVERRIDE)


def new_load_table():
    runs = {}
    for ds in NEW_LOAD:
        runs[ds] = {}
        for f in (f"results/suite_{ds}_L720.json", f"results/fix_{ds}_L720.json", f"results/external_{ds}_L720.json"):
            runs[ds].update(json.loads(Path(f).read_text(encoding="utf-8")))
    hs = [int(h) for h in HORIZONS]
    val = {c: {ds: seed_stats(runs[ds], v, hs) for ds in NEW_LOAD} for c, v in NEW_LOAD_MODELS.items()}
    head = [r"\multirow{2}{*}{Mô hình} & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{d}}}" for d in NEW_LOAD) + r"\\",
            "".join(rf"\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(NEW_LOAD))),
            " & " + " & ".join("MSE & MAE" for _ in NEW_LOAD) + r"\\", r"\midrule"]
    order = sorted(val, key=lambda c: statistics.mean(val[c][ds][0][0] for ds in NEW_LOAD))   # MSE TB tăng dần
    rows = [f"{name(c)} & " + " & ".join(mark_std({k: val[k][ds][i] for k in val}, c)
                                         for ds in NEW_LOAD for i in (0, 1)) + r"\\" for c in order]
    return wrap(head + rows, "l" + "cc" * len(NEW_LOAD))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, table = load_all()
    stats = bench_stats(runs, table)
    files = {"avg.tex": avg_table(stats), "h_etth.tex": horizon_table(table, ["ETTh1", "ETTh2"]),
             "h_ettm.tex": horizon_table(table, ["ETTm1", "ETTm2"]), "h_weather.tex": horizon_table(table, ["Weather"]),
             "h2h.tex": h2h_table(stats), "versus.tex": versus_table(table), "efficiency.tex": efficiency_table(),
             "ablation.tex": ablation_table(runs), "hparams.tex": hparam_table(), "load.tex": load_table(),
             "new_load.tex": new_load_table()}
    for n, text in files.items():
        (OUT / n).write_text(text, encoding="utf-8")
    print("wrote", ", ".join(files))


if __name__ == "__main__":
    main()
