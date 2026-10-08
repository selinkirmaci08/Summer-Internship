# plot_closure_pattern.py
# Shows the weekday closure pattern in the stress test (2022-2023).

import re, glob, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

RUN_DIR     = Path(__file__).parent.parent
TEST_ROOT   = Path("C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/data/test_set")
PRIMARY_DIR = TEST_ROOT / "holdout_primary"
STRESS_DIR  = TEST_ROOT / "holdout_stress"
PROD_CSV    = PRIMARY_DIR / "productdata" / "productdata.csv"
OUT_PNG     = RUN_DIR / "artifacts" / "plots" / "closure_pattern.png"
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

prod_df = pd.read_csv(PROD_CSV, sep="|", encoding="latin-1", on_bad_lines="skip", low_memory=False)
prod_df.columns = [c.lstrip("\ufeff").strip() for c in prod_df.columns]
bread_articles = set(
    prod_df[prod_df["argcod"].astype(str).str.strip() == "1"]["artcod"]
    .astype(str).str.strip().str.zfill(8)
)

def parse_xml(fpath, bread_set):
    try:
        root = ET.parse(fpath).getroot()
    except Exception:
        return None, 0.0
    sd = root.get("salesdate", "")
    use_new = len(root.findall("sale")) > 0
    txn_tag = "sale" if use_new else "order"
    det_tag = "saledetail" if use_new else "orderdetail"
    bq = 0.0
    for txn in root.findall(txn_tag):
        for det in txn.findall(det_tag):
            art = det.get("articlereference", "").strip().zfill(8)
            try: qty = float(det.get("quantity", "0").replace(",", "."))
            except: qty = 0.0
            if art in bread_set: bq += qty
    return sd, bq

def load_dir(xml_dir):
    xmls = glob.glob(str(xml_dir / "**" / "*.xml"), recursive=True)
    dtf = {}
    for fpath in xmls:
        m = re.search(r"PT_Sales_\d+_(\d{8})_(\d{14})\.xml", Path(fpath).name)
        if m:
            d, ts = m.group(1), m.group(2)
            if d not in dtf or ts > dtf[d][0]:
                dtf[d] = (ts, fpath)
    recs = []
    for d, (_, fpath) in sorted(dtf.items()):
        sd, bq = parse_xml(fpath, bread_articles)
        if sd: recs.append({"date": pd.Timestamp(sd), "bread_quantity": bq})
    return pd.DataFrame(recs).sort_values("date").reset_index(drop=True)

print("Parsing stress test XMLs...")
df = load_dir(STRESS_DIR / "salesdata" / "raw_xml" / "01")
df["dow"]      = df["date"].dt.dayofweek
df["is_closed"] = (df["bread_quantity"] == 0).astype(int)

# Build a full calendar grid (every day from Jan 2022 to Feb 2023)
full_range = pd.date_range("2022-01-01", "2023-02-08", freq="D")
df_full = pd.DataFrame({"date": full_range})
df_full["dow"]   = df_full["date"].dt.dayofweek
df_full["week"]  = df_full["date"].dt.isocalendar().week.astype(int)
df_full["year"]  = df_full["date"].dt.year
df_full["month"] = df_full["date"].dt.month

# Merge: 0=no data, 1=open, 2=closed
df_full = df_full.merge(df[["date", "bread_quantity", "is_closed"]], on="date", how="left")
# status: 0=missing, 1=open, 2=closed
df_full["status"] = 0  # missing
df_full.loc[df_full["bread_quantity"].notna() & (df_full["is_closed"] == 0), "status"] = 1  # open
df_full.loc[df_full["bread_quantity"].notna() & (df_full["is_closed"] == 1), "status"] = 2  # closed

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.patch.set_facecolor("white")

# Colors: missing=lightgray, open=green, closed=red
CMAP   = ListedColormap(["#E0E0E0", "#4CAF50", "#EF5350"])
LABELS = ["No data", "Open", "Closed"]

# ── Panel 1: Calendar heatmap ─────────────────────────────────────────────────
ax1 = axes[0]

# Build week × DOW grid
# Use sequential week number across the full period
df_full["seq_week"] = ((df_full["date"] - df_full["date"].min()).dt.days // 7)
n_weeks = df_full["seq_week"].max() + 1
grid = np.zeros((7, n_weeks))  # rows=DOW, cols=week

for _, row in df_full.iterrows():
    grid[int(row["dow"]), int(row["seq_week"])] = row["status"]

im = ax1.imshow(grid, cmap=CMAP, vmin=0, vmax=2, aspect="auto", interpolation="nearest")

# Month labels on x-axis
month_ticks = []
month_labels = []
for yr in [2022, 2023]:
    for mo in range(1, 13):
        first_day = pd.Timestamp(f"{yr}-{mo:02d}-01")
        if first_day < pd.Timestamp("2022-01-01") or first_day > pd.Timestamp("2023-02-08"):
            continue
        seq = (first_day - df_full["date"].min()).days // 7
        month_ticks.append(seq)
        month_labels.append(first_day.strftime("%b\n%Y") if mo == 1 else first_day.strftime("%b"))

ax1.set_xticks(month_ticks)
ax1.set_xticklabels(month_labels, fontsize=7)
ax1.set_yticks(range(7))
ax1.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=9)

# Vertical line between 2022 and 2023
jan23_seq = (pd.Timestamp("2023-01-01") - df_full["date"].min()).days // 7
ax1.axvline(jan23_seq - 0.5, color="black", linewidth=2, linestyle="--", alpha=0.7)
ax1.text(jan23_seq + 0.3, -0.8, "2023\n(complete)", fontsize=7, color="black", ha="left")
ax1.text(jan23_seq - 1.5, -0.8, "2022\n(incomplete)", fontsize=7, color="gray", ha="right")

patches = [mpatches.Patch(color="#4CAF50", label="Open"),
           mpatches.Patch(color="#EF5350", label="Closed"),
           mpatches.Patch(color="#E0E0E0", label="No data")]
ax1.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.9)
ax1.set_title("Stress Test Calendar (2022–2023)\nGreen=Open  Red=Closed  Gray=No XML data", fontsize=10, fontweight="bold", pad=10)
ax1.set_xlabel("Month", fontsize=9)

# ── Panel 2: Closure rate by DOW (stress period) ──────────────────────────────
ax2 = axes[1]
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
bar_colors = ["#EF5350", "#EF5350", "#EF5350", "#EF5350", "#EF5350", "#4CAF50", "#4CAF50"]

# Only days where we have data
df_data = df_full[df_full["status"] != 0]
closure_rate = [
    100 * (df_data[df_data["dow"] == d]["status"] == 2).mean()
    for d in range(7)
]
n_days = [len(df_data[df_data["dow"] == d]) for d in range(7)]

bars = ax2.bar(dow_names, closure_rate, color=bar_colors, alpha=0.85,
               edgecolor="white", linewidth=0.5)

for i, (bar, rate, n) in enumerate(zip(bars, closure_rate, n_days)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{rate:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.text(bar.get_x() + bar.get_width()/2, -5,
             f"n={n}", ha="center", va="top", fontsize=7, color="gray")

ax2.set_title("Closure Rate by Day of Week\n(Stress Period, days with XML data)", fontsize=10, fontweight="bold", pad=10)
ax2.set_xlabel("Day of week", fontsize=9)
ax2.set_ylabel("% of days closed (zero bread sales)", fontsize=9)
ax2.set_ylim(0, 85)
ax2.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top", "right"]].set_visible(False)

# ── Panel 3: Data coverage by month ───────────────────────────────────────────
ax3 = axes[2]

months = pd.date_range("2022-01", "2023-03", freq="MS")[:-1]
coverage = []
open_rate = []
for mo in months:
    end = (mo + pd.offsets.MonthEnd(0)).date()
    start = mo.date()
    total_days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    in_range = df_full[(df_full["date"].dt.date >= start) & (df_full["date"].dt.date <= end)]
    n_with_data = (in_range["status"] != 0).sum()
    n_open = (in_range["status"] == 1).sum()
    coverage.append(100 * n_with_data / total_days)
    open_rate.append(100 * n_open / total_days)

x = range(len(months))
labels = [m.strftime("%b\n%Y") if m.month == 1 else m.strftime("%b") for m in months]

ax3.bar(x, coverage,  color="#90A4AE", alpha=0.8, label="Days with data")
ax3.bar(x, open_rate, color="#4CAF50", alpha=0.85, label="Open days")

ax3.axvline(11.5, color="black", linewidth=2, linestyle="--", alpha=0.7)
ax3.text(9,  97, "2022\n(incomplete\ndata)", ha="center", fontsize=8, color="gray")
ax3.text(13, 97, "2023\n(complete\ndata)", ha="center", fontsize=8, color="black")

ax3.set_title("Monthly Data Coverage\n(% of calendar days)", fontsize=10, fontweight="bold", pad=10)
ax3.set_xlabel("Month", fontsize=9)
ax3.set_ylabel("% of calendar days", fontsize=9)
ax3.set_xticks(list(x))
ax3.set_xticklabels(labels, fontsize=7)
ax3.set_ylim(0, 110)
ax3.legend(fontsize=8, loc="lower right")
ax3.grid(axis="y", alpha=0.3)
ax3.spines[["top", "right"]].set_visible(False)

# ── Final layout ───────────────────────────────────────────────────────────────
fig.suptitle("Stress Test (2022–2023): Weekend-Dominant Operation Pattern",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PNG}")
plt.close()
