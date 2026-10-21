"""Sinh các bảng LaTeX cho Chương 3 (Thực nghiệm) của báo cáo DyTimeBase từ file kết quả JSON.

Không chép tay số liệu: mọi bảng đọc từ results/*.json. Ghi vào docs/tables/*.tex, được \\input trong
docs/dytimebase_phuong_phap_luan.tex. Ví dụ: python scripts/make_report_tables.py
"""

import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_all import ABLATION, DATASETS, HORIZONS, PUBLISHED, SEEDS  # noqa: E402
from run_suite import TB_INDIVIDUAL, TB_OFFICIAL  # noqa: E402

OUT = Path("docs/tables")
BRANCH = {"ETTh1": (64, 0.1, 0.1), "ETTh2": (64, 0.6, 0.6), "ETTm1": (64, 0.1, 0.2), "ETTm2": (32, 0.1, 0.1),
          "Weather": (64, 0.1, 0.1)}      # Đồ án I, Bảng 3.1: d_model, p1, p2
COLS = ["DyTimeBase", "TimeBase*"] + PUBLISHED
HEAD = {"DyTimeBase": r"\textbf{DyTimeBase}", "TimeBase*": r"TimeBase$^{*}$", "TimeBase": r"TimeBase$^{\dagger}$"}


def load_all():
    pub = json.loads(Path("results/published_timebase_table10.json").read_text(encoding="utf-8"))["results"]
    runs = {ds: json.loads(Path(p).read_text(encoding="utf-8")) for ds, p in DATASETS.items()}
    table = {}
    for ds in DATASETS:
        for h in HORIZONS:
            row = {}
            for label, v in (("DyTimeBase", "dytbase_off"), ("TimeBase*", "timebase_off")):
                rs = [runs[ds][f"{v}_S{h}_seed{s}"] for s in SEEDS]
                row[label] = (statistics.mean(r["mse"] for r in rs), statistics.mean(r["mae"] for r in rs),
                              statistics.stdev(r["mse"] for r in rs))
            for m in PUBLISHED:
                row[m] = (pub[ds][m][h]["mse"], pub[ds][m][h]["mae"], None)
            table[(ds, h)] = row
        table[(ds, "Avg")] = {c: (statistics.mean(table[(ds, h)][c][0] for h in HORIZONS),
                                  statistics.mean(table[(ds, h)][c][1] for h in HORIZONS), None) for c in COLS}
    return runs, table


def mark(values, v):
    """Đậm = tốt nhất, gạch chân = thứ hai (so sau khi làm tròn 3 chữ số)."""
    ranked = sorted({round(x, 3) for x in values})
    s = f"{v:.3f}"
    if round(v, 3) == ranked[0]:
        return rf"\textbf{{{s}}}"
    return rf"\underline{{{s}}}" if len(ranked) > 1 and round(v, 3) == ranked[1] else s


def accuracy_table(table, metric, idx):
    lines = [r"\begin{sidewaystable}", r"\centering",
             rf"\caption{{Kết quả dự báo dài hạn ({metric}), độ dài đầu vào $L=720$ cho mọi mô hình. "
             r"DyTimeBase và TimeBase$^{*}$: tự chạy, trung bình 3 seed. Các cột còn lại (TimeBase$^{\dagger}$ là số "
             r"công bố): trích từ Table 10 của~\cite{timebase}. \textbf{Đậm}: tốt nhất; \underline{gạch chân}: thứ hai.}",
             rf"\label{{tab:acc-{metric.lower()}}}", r"\scriptsize", r"\setlength{\tabcolsep}{3.2pt}",
             r"\begin{tabular}{@{}ll" + "c" * len(COLS) + "@{}}", r"\toprule",
             "Dataset & $S$ & " + " & ".join(HEAD.get(c, c) for c in COLS) + r"\\", r"\midrule"]
    for ds in DATASETS:
        for k, h in enumerate(HORIZONS + ("Avg",)):
            vals = [table[(ds, h)][c][idx] for c in COLS]
            name = rf"\multirow{{5}}{{*}}{{{ds}}}" if k == 0 else ""
            hlab = "TB" if h == "Avg" else h
            lines.append(f"{name} & {hlab} & " + " & ".join(mark(vals, v) for v in vals) + r"\\")
        lines.append(r"\midrule" if ds != list(DATASETS)[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{sidewaystable}"]
    return "\n".join(lines) + "\n"


def head_to_head(table):
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{DyTimeBase đối đầu từng mô hình theo MSE: số ô thắng / hòa / thua trên 20 ô (5 dataset $\times$ "
             r"4 độ dài dự báo, làm tròn 3 chữ số) và các dataset DyTimeBase có MSE trung bình tốt hơn hoặc kém hơn.}",
             r"\label{tab:h2h}", r"\small", r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}lcccll@{}}", r"\toprule",
             r"Mô hình & Thắng & Hòa & Thua & DyTimeBase tốt hơn & DyTimeBase kém hơn\\", r"\midrule"]
    for c in COLS[1:]:
        w = t = l = 0
        for ds in DATASETS:
            for h in HORIZONS:
                a, b = round(table[(ds, h)]["DyTimeBase"][0], 3), round(table[(ds, h)][c][0], 3)
                w, t, l = w + (a < b), t + (a == b), l + (a > b)
        better = [ds for ds in DATASETS if table[(ds, "Avg")]["DyTimeBase"][0] < table[(ds, "Avg")][c][0]]
        worse = [ds for ds in DATASETS if ds not in better]
        lines.append(f"{HEAD.get(c, c)} & {w} & {t} & {l} & {', '.join(better) or '--'} & {', '.join(worse) or '--'}"
                     r"\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def versus_timebase(table):
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{DyTimeBase so với TimeBase$^{*}$ chạy lại trong cùng quy trình (cùng xương sống, cùng siêu tham số "
             r"xương sống). MSE trung bình $\pm$ độ lệch chuẩn qua 3 seed; cột cuối là chênh lệch tương đối.}",
             r"\label{tab:vs-timebase}", r"\small", r"\begin{tabular}{@{}llccr@{}}", r"\toprule",
             r"Dataset & $S$ & DyTimeBase & TimeBase$^{*}$ & Chênh lệch\\", r"\midrule"]
    for ds in DATASETS:
        for k, h in enumerate(HORIZONS + ("Avg",)):
            a, b = table[(ds, h)]["DyTimeBase"], table[(ds, h)]["TimeBase*"]
            fa = f"{a[0]:.4f}" + (rf" $\pm$ {a[2]:.4f}" if a[2] is not None else "")
            fb = f"{b[0]:.4f}" + (rf" $\pm$ {b[2]:.4f}" if b[2] is not None else "")
            if a[0] < b[0]:
                fa = rf"\textbf{{{fa}}}"
            else:
                fb = rf"\textbf{{{fb}}}"
            name = rf"\multirow{{5}}{{*}}{{{ds}}}" if k == 0 else ""
            lines.append(f"{name} & {'TB' if h == 'Avg' else h} & {fa} & {fb} & ${100 * (a[0] - b[0]) / b[0]:+.1f}\\%$"
                         r"\\")
        lines.append(r"\midrule" if ds != list(DATASETS)[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def ablation(runs):
    rows = {}
    for label, v in ABLATION.items():
        rows[label] = [statistics.mean(runs[ds][f"{v}_S{h}_seed{s}"]["mse"] for h in HORIZONS for s in SEEDS)
                       for ds in DATASETS]
    tex_label = {"DyTimeBase (đầy đủ)": r"DyTimeBase (đầy đủ)", "bỏ CrossV": r"\quad bỏ CrossVBlock",
                 "bỏ IntraV": r"\quad bỏ IntraVBlock",
                 "bỏ cả nhánh trộn kênh (= TimeBase\\*)": r"\quad bỏ cả nhánh trộn kênh ($=$ TimeBase$^{*}$)",
                 "đầy đủ, mix_lr 2e-3": r"DyTimeBase, $\alpha_M = 2\cdot10^{-3}$"}
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Nghiên cứu loại bỏ (ablation) các khối của nhánh trộn kênh: MSE trung bình trên 4 độ dài dự báo và "
             r"3 seed. Dòng cuối giữ nguyên kiến trúc, chỉ tăng tốc độ học của nhánh trộn kênh từ $5\cdot10^{-4}$ lên "
             r"$2\cdot10^{-3}$. \textbf{Đậm}: tốt nhất mỗi cột.}",
             r"\label{tab:ablation}", r"\small",
             r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}l" + "c" * (len(DATASETS) + 1) + "@{}}",
             r"\toprule", "Biến thể & " + " & ".join(DATASETS) + r" & Trung bình\\", r"\midrule"]
    means = {k: statistics.mean(v) for k, v in rows.items()}
    for label, vals in rows.items():
        cells = []
        for i, v in enumerate(vals + [means[label]]):
            col = [r[i] for r in rows.values()] if i < len(DATASETS) else list(means.values())
            cells.append(rf"\textbf{{{v:.4f}}}" if v == min(col) else f"{v:.4f}")
        lines.append(f"{tex_label[label]} & " + " & ".join(cells) + r"\\")
        if label == "bỏ cả nhánh trộn kênh (= TimeBase\\*)":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def efficiency():
    r = json.loads(Path("results/bench_efficiency.json").read_text(encoding="utf-8"))
    ref = r["DyTimeBase"]["infer_ms_t8"]

    def count(v):
        return f"{v / 1e9:.2f}G" if v >= 1e9 else f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.2f}K"

    def th(v, fmt=",.0f"):                  # dấu cách hẹp phân tách hàng nghìn, chỉ trong ô số
        return format(v, fmt).replace(",", r"\,")
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Hiệu năng tính toán trên cùng một máy CPU, cấu hình như Table~3 của~\cite{timebase}: Electricity "
             r"($C=321$), $L=S=720$. Max Mem: đỉnh bộ nhớ RAM khi huấn luyện với batch 12; Epoch Time: thời gian một "
             r"bước huấn luyện (batch 12) nhân số bước của một epoch (1\,414); Infer Time: dự báo 1 mẫu, 8 luồng. "
             r"Sắp xếp theo Infer Time.}",
             r"\label{tab:efficiency}", r"\small", r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Mô hình & Params & MACs & Max Mem (MB) & Epoch Time (s) & Infer Time (ms) & So với DyTimeBase\\",
             r"\midrule"]
    for n, x in sorted(r.items(), key=lambda kv: kv[1]["infer_ms_t8"]):
        mem = rf"$>${th(x['mem_over_gb'] * 1024)}$^{{\ddagger}}$" if "mem_over_gb" in x else th(x["max_mem_mb"])
        name = r"\textbf{DyTimeBase}" if n == "DyTimeBase" else n.replace("(cũ)", "(Đồ án I)")
        lines.append(f"{name} & {count(x['params'])} & {count(x['macs'])} & {mem} & {th(x['epoch_s'])} & "
                     f"{th(x['infer_ms_t8'], ',.2f')} & {th(x['infer_ms_t8'] / ref, ',.2f')}$\\times$" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}}", "",
              r"\footnotesize{$^{\ddagger}$ Vượt ngưỡng 11 GB RAM ở batch 12; Epoch Time của mô hình này đo ở batch 4.}",
              r"\end{table}"]
    return "\n".join(lines) + "\n"


def hyperparams():
    lines = [r"\begin{table}[H]", r"\centering",
             r"\caption{Siêu tham số của DyTimeBase. Xương sống: theo script chính thức của TimeBase~\cite{timebase} "
             r"cho từng độ dài dự báo $S\in\{96,192,336,720\}$. Nhánh trộn kênh: giữ nguyên bộ đã tối ưu của DyTimeNet "
             r"(Đồ án I, Bảng 3.1).}",
             r"\label{tab:hparams}", r"\small", r"\resizebox{\textwidth}{!}{\begin{tabular}{@{}llllllccc@{}}", r"\toprule",
             r" & \multicolumn{5}{c}{Xương sống (theo $S=96/192/336/720$)} & & \multicolumn{2}{c}{Nhánh trộn kênh}\\",
             r"\cmidrule(lr){2-6}\cmidrule(l){8-9}",
             r"Dataset & $P$ & $R$ & $\alpha_B$ & Batch & $\lambda_{orth}$ & Riêng kênh & $D$ & $p_1$ / $p_2$\\",
             r"\midrule"]
    for ds in DATASETS:
        v = [TB_OFFICIAL[ds][int(h)] for h in HORIZONS]
        j = lambda i: "/".join(f"{x[i]:g}" for x in v)  # noqa: E731
        d, p1, p2 = BRANCH[ds]
        lines.append(f"{ds} & {j(3)} & {j(4)} & {j(2)} & {j(1)} & {j(0)} & "
                     f"{'có' if TB_INDIVIDUAL[ds] else 'không'} & {d} & {p1:g} / {p2:g}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, table = load_all()
    files = {"acc_mse.tex": accuracy_table(table, "MSE", 0), "acc_mae.tex": accuracy_table(table, "MAE", 1),
             "h2h.tex": head_to_head(table), "vs_timebase.tex": versus_timebase(table), "ablation.tex": ablation(runs),
             "efficiency.tex": efficiency(), "hparams.tex": hyperparams()}
    for name, text in files.items():
        (OUT / name).write_text(text, encoding="utf-8")
    print("wrote", ", ".join(files))


if __name__ == "__main__":
    main()
