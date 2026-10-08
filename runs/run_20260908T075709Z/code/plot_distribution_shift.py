# plot_distribution_shift.py
# Generates a presentation plot explaining the 2022-2023 distribution shift.

import re, glob, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

RUN_DIR     = Path(__file__).parent.parent
FE4_CSV     = RUN_DIR / "data" / "iteration_019_feature_engineering.csv"
TEST_ROOT   = Path("C:/Users/selin/OneDrive/Desktop/All/KU Leuven/internship-project/General Plan/data/test_set")
PRIMARY_DIR = TEST_ROOT / "holdout_primary"
STRESS_DIR  = TEST_ROOT / "holdout_stress"
PROD_CSV    = PRIMARY_DIR / "productdata" / "productdata.csv"
OUT_PNG     = RUN_DIR / "artifacts" / "plots" / "distribution_shift.png"
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

# ── Bread article map ──────────────────────────────────────────────────────────
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

print("Parsing test XMLs...")
df_primary = load_dir(PRIMARY_DIR / "salesdata" / "raw_xml" / "01")
df_stress  = load_dir(STRESS_DIR  / "salesdata" / "raw_xml" / "01")

print("Loading training data...")
df_train = pd.read_csv(str(FE4_CSV), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df_train_all = df_train.copy()  # includes COVID rows for full timeline

# ── Combine full timeline ──────────────────────────────────────────────────────
df_all = pd.concat([
    df_train_all[["date", "bread_quantity"]].assign(period="Training (2010–2020)"),
    df_primary[["date", "bread_quantity"]].assign(period="Primary (2021)"),
    df_stress[["date", "bread_quantity"]].assign(period="Stress (2022–2023)"),
], ignore_index=True).sort_values("date")

df_all["year_month"] = df_all["date"].dt.to_period("M")
df_all["dow"] = df_all["date"].dt.dayofweek
df_all["is_open"] = (df_all["bread_quantity"] > 0).astype(int)

# Monthly averages (all days)
monthly_all = df_all.groupby("year_month")["bread_quantity"].mean().reset_index()
monthly_all["date"] = monthly_all["year_month"].dt.to_timestamp()

# Monthly averages (open days only)
monthly_open = df_all[df_all["bread_quantity"] > 0].groupby("year_month")["bread_quantity"].mean().reset_index()
monthly_open["date"] = monthly_open["year_month"].dt.to_timestamp()

# DOW averages by period
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
periods = ["Training (2010–2020)", "Primary (2021)", "Stress (2022–2023)"]
colors  = ["#2196F3", "#4CAF50", "#F44336"]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.patch.set_facecolor("white")

PERIOD_COLORS = {
    "Training (2010–2020)": "#2196F3",
    "Primary (2021)":       "#4CAF50",
    "Stress (2022–2023)":   "#F44336",
}

# ── Panel 1: Monthly average over full timeline ────────────────────────────────
ax1 = axes[0]

# Shade regions
train_end   = pd.Timestamp("2020-12-31")
primary_end = pd.Timestamp("2021-12-31")
stress_end  = pd.Timestamp("2023-03-01")
ax1.axvspan(pd.Timestamp("2010-01-01"), train_end,   alpha=0.08, color="#2196F3")
ax1.axvspan(pd.Timestamp("2021-01-01"), primary_end, alpha=0.12, color="#4CAF50")
ax1.axvspan(pd.Timestamp("2022-01-01"), stress_end,  alpha=0.12, color="#F44336")

# COVID shading
ax1.axvspan(pd.Timestamp("2020-03-18"), pd.Timestamp("2020-06-08"), alpha=0.25, color="orange", label="COVID lockdown")

# Monthly lines
ax1.plot(monthly_all["date"],  monthly_all["bread_quantity"],
         color="black", linewidth=1.5, alpha=0.6, label="Monthly avg (all days)")
ax1.plot(monthly_open["date"], monthly_open["bread_quantity"],
         color="gray", linewidth=1.2, alpha=0.5, linestyle="--", label="Monthly avg (open days)")

# Vertical separators
ax1.axvline(pd.Timestamp("2021-01-01"), color="gray", linestyle=":", linewidth=1)
ax1.axvline(pd.Timestamp("2022-01-01"), color="gray", linestyle=":", linewidth=1)

ax1.text(pd.Timestamp("2015-06-01"), 530, "Training", ha="center", fontsize=9,
         color="#2196F3", fontweight="bold")
ax1.text(pd.Timestamp("2021-07-01"), 530, "Primary", ha="center", fontsize=9,
         color="#4CAF50", fontweight="bold")
ax1.text(pd.Timestamp("2022-08-01"), 530, "Stress", ha="center", fontsize=9,
         color="#F44336", fontweight="bold")

ax1.set_title("Monthly Average Bread Quantity (2010–2023)", fontsize=11, fontweight="bold", pad=10)
ax1.set_xlabel("Year", fontsize=10)
ax1.set_ylabel("Daily bread quantity (units)", fontsize=10)
ax1.set_ylim(0, 570)
ax1.legend(fontsize=8, loc="lower left")
ax1.grid(axis="y", alpha=0.3)
ax1.tick_params(axis="x", labelsize=8)
ax1.spines[["top", "right"]].set_visible(False)

# ── Panel 2: DOW average by period (all days) ──────────────────────────────────
ax2 = axes[1]
x = np.arange(7)
width = 0.28

for i, (period, color) in enumerate(PERIOD_COLORS.items()):
    subset = df_all[df_all["period"] == period]
    means  = [subset[subset["dow"] == d]["bread_quantity"].mean() for d in range(7)]
    bars = ax2.bar(x + (i - 1) * width, means, width, label=period,
                   color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

ax2.set_title("Average Bread Quantity by Day of Week\n(All Days — including closures)", fontsize=11, fontweight="bold", pad=10)
ax2.set_xlabel("Day of week", fontsize=10)
ax2.set_ylabel("Daily bread quantity (units)", fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(dow_names, fontsize=9)
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_ylim(0, 450)

# ── Panel 3: DOW average (open days only) + closure rate ──────────────────────
ax3 = axes[2]
ax3b = ax3.twinx()

for i, (period, color) in enumerate(PERIOD_COLORS.items()):
    subset = df_all[(df_all["period"] == period) & (df_all["bread_quantity"] > 0)]
    means  = [subset[subset["dow"] == d]["bread_quantity"].mean() for d in range(7)]
    ax3.bar(x + (i - 1) * width, means, width, label=period,
            color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

# Closure rate for stress only (line on secondary axis)
stress_sub = df_all[df_all["period"] == "Stress (2022–2023)"]
closure_rate = [
    100 * (stress_sub[stress_sub["dow"] == d]["bread_quantity"] == 0).mean()
    for d in range(7)
]
ax3b.plot(x, closure_rate, color="#F44336", marker="o", linewidth=2,
          markersize=5, linestyle="-", label="Stress closure rate (%)", zorder=5)
ax3b.set_ylabel("Closure rate in stress period (%)", fontsize=9, color="#F44336")
ax3b.tick_params(axis="y", labelcolor="#F44336")
ax3b.set_ylim(0, 100)
ax3b.spines["right"].set_edgecolor("#F44336")

ax3.set_title("Average Bread Quantity by Day of Week\n(Open Days Only + Closure Rate)", fontsize=11, fontweight="bold", pad=10)
ax3.set_xlabel("Day of week", fontsize=10)
ax3.set_ylabel("Daily bread quantity (units)", fontsize=10)
ax3.set_xticks(x)
ax3.set_xticklabels(dow_names, fontsize=9)
ax3.legend(fontsize=8, loc="upper left")
ax3b.legend(fontsize=8, loc="upper right")
ax3.grid(axis="y", alpha=0.3)
ax3.spines[["top"]].set_visible(False)
ax3.set_ylim(0, 450)

# ── Final layout ───────────────────────────────────────────────────────────────
fig.suptitle("Distribution Shift Analysis: Training vs Primary (2021) vs Stress (2022–2023)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PNG}")
plt.close()
