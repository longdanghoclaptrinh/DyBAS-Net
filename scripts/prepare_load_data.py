"""Chuẩn bị ba bộ phụ tải đa biến mới thành CSV cho loader (cột `date` + các biến số theo thứ tự).

- Tetouan (UCI #849 / Kaggle fedesoriano): powerconsumption.csv, 10 phút, 8 biến, đã sạch → chỉ đổi tên cột thời gian.
- Panama (Mendeley byx7sztj59): continuous dataset.csv, 1 giờ; bỏ 3 cột lịch Holiday_ID/holiday/school (phân loại,
  nhị phân, không nên dự báo) → 13 biến.
- PJM (Kaggle Hourly Energy Consumption): 10 vùng có chung đoạn 2013-06-01 01:00 → 2018-08-03 00:00 (EKPC bắt đầu
  muộn nhất). Mỗi vùng: sắp theo thời gian, lấy trung bình giờ trùng (đổi giờ mùa thu), nội suy tuyến tính giờ thiếu
  (đổi giờ mùa xuân, cùng 1 giờ lẻ của AEP) → 10 biến. Không sửa giá trị bất thường (DEOK 2018-04-22 giữ nguyên).
Ví dụ: python scripts/prepare_load_data.py  → data/tetouan.csv, data/panama.csv, data/pjm.csv
"""

from pathlib import Path

import pandas as pd

DATA = Path("data")
PJM_REGIONS = ["AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ", "EKPC", "FE", "PJME", "PJMW"]
PJM_START, PJM_END = "2013-06-01 01:00", "2018-08-03 00:00"


def tetouan():
    df = pd.read_csv(DATA / "powerconsumption.csv")
    df.insert(0, "date", pd.to_datetime(df.pop("Datetime"), format="%m/%d/%Y %H:%M"))
    return df


def panama():
    df = pd.read_csv(DATA / "continuous dataset.csv").drop(columns=["Holiday_ID", "holiday", "school"])
    df.insert(0, "date", pd.to_datetime(df.pop("datetime")))
    return df


def pjm():
    full = pd.date_range(PJM_START, PJM_END, freq="h")
    cols = {}
    for region in PJM_REGIONS:
        raw = pd.read_csv(DATA / "PJM" / f"{region}_hourly.csv")
        s = raw.set_index(pd.to_datetime(raw["Datetime"]))[f"{region}_MW"]
        s = s.groupby(level=0).mean().reindex(full)
        print(f"  {region}: nội suy {s.isna().sum()} giờ thiếu")
        cols[region] = s.interpolate("linear")
    df = pd.DataFrame(cols).rename_axis("date").reset_index()
    return df


if __name__ == "__main__":
    for name, build in [("tetouan", tetouan), ("panama", panama), ("pjm", pjm)]:
        try:
            df = build()
        except FileNotFoundError as e:   # chỉ chuẩn bị những bộ đã tải file gốc
            print(f"{name}: bỏ qua, thiếu {e.filename}")
            continue
        t = df["date"]
        assert t.is_monotonic_increasing and not t.duplicated().any() and t.diff()[1:].nunique() == 1
        assert not df.isna().any().any()
        df.to_csv(DATA / f"{name}.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
        print(f"{name}.csv: {len(df)} dòng × {df.shape[1] - 1} biến, {t.iloc[0]} → {t.iloc[-1]}")
