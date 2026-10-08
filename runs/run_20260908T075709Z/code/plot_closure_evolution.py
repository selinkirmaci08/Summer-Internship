# plot_closure_evolution.py
# Shows how closed/zero-sales days evolved from training through test periods.

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
OUT_PNG     = RUN_DIR / "artifacts" / "plots" / "closure_evolution.png"
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

# ── Load training data ─────────────────────────────────────────────────────────
print("Loading training data...")
df_train = pd.read_csv(str(FE4_CSV), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
df_train["is_closed"] = (df_train["bread_quantity"] == 0).astype(int)

# ── Load test XMLs ─────────────────────────────────────────────────────────────
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
df_primary["is_closed"] = (df_primary["bread_quantity"] == 0).astype(int)
df_stress["is_closed"]  = (df_stress["bread_quantity"]  == 0).astype(int)

# ── Combine all ────────────────────────────────────────────────────────────────
df_all = pd.concat([
    df_train[["date", "bread_quantity", "is_closed"]].assign(period="Training"),
    df_primary[["date", "bread_quantity", "is_closed"]].assign(period="Primary"),
    df_stress[["date", "bread_quantity", "is_closed"]].assign(period="Stress"),
], ignore_index=True).sort_values("date")

df_all["year"]  = df_all["date"].dt.year
df_all["month"] = df_all["date"].dt.month
df_all["dow"]   = df_all["date"].dt.dayofweek
df_all["ym"]    = df_all["date"].dt.to_period("M")

# ── Yearly closure rate ────────────────────────────────────────────────────────
yearly = df_all.groupby("year").agg(
    total=("is_closed", "count"),
    closed=("is_closed", "sum")
).reset_index()
yearly["closure_rate"] = 100 * yearly["closed"] / yearly["total"]
yearly["period"] = yearly["year"].apply(
    lambda y: "Training" if y <= 2020 else ("Primary" if y == 2021 else "Stress"))

# ── DOW closure rate by period ─────────────────────────────────────────────────
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
periods   = ["Training", "Primary", "Stress"]
p_colors  = {"Training": "#2196F3", "Primary": "#4CAF50", "Stress": "#F44336"}

dow_rates = {}
for p in periods:
    sub = df_all[df_all["period"] == p]
    dow_rates[p] = [100 * sub[sub["dow"] == d]["is_closed"].mean() for d in range(7)]

# ── Monthly closure rate timeline ─────────────────────────────────────────────
monthly = df_all.groupby("ym").agg(
    total=("is_closed", "count"),
    closed=("is_closed", "sum")
).reset_index()
monthly["date"]         = monthly["ym"].dt.to_timestamp()
monthly["closure_rate"] = 100 * monthly["closed"] / monthly["total"]
monthly["period"] = monthly["date"].apply(
    lambda d: "Training" if d.year <= 2020 else ("Primary" if d.year == 2021 else "Stress"))

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.patch.set_facecolor("white")

# ── Panel 1: Yearly closure rate bar chart ────────────────────────────────────
ax1 = axes[0]
bar_colors = [p_colors[p] for p in yearly["period"]]
bars = ax1.bar(yearly["year"].astype(str), yearly["closure_rate"],
               color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)

for bar, rate in zip(bars, yearly["closure_rate"]):
    if rate > 1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{rate:.0f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

# COVID annotation
ax1.annotate("COVID\nexcluded\nfrom training", xy=("2020", 33.2),
             xytext=("2017", 45),
             arrowprops=dict(arrowstyle="->", color="orange", lw=1.5),
             fontsize=7.5, color="orange", ha="center")

legend_patches = [mpatches.Patch(color=p_colors[p], label=p, alpha=0.85) for p in periods]
ax1.legend(handles=legend_patches, fontsize=8, loc="upper left")
ax1.set_title("Zero-Sales Days per Year\n(% of all recorded days)", fontsize=10, fontweight="bold", pad=10)
ax1.set_xlabel("Year", fontsize=9)
ax1.set_ylabel("% days with zero bread sales", fontsize=9)
ax1.set_ylim(0, 60)
ax1.grid(axis="y", alpha=0.3)
ax1.tick_params(axis="x", labelsize=7.5, rotation=45)
ax1.spines[["top", "right"]].set_visible(False)

# ── Panel 2: Monthly closure rate timeline ────────────────────────────────────
ax2 = axes[1]

# Background shading
ax2.axvspan(pd.Timestamp("2010-01-01"), pd.Timestamp("2020-12-31"), alpha=0.06, color="#2196F3")
ax2.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"), alpha=0.10, color="#4CAF50")
ax2.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2023-03-01"),  alpha=0.10, color="#F44336")
ax2.axvspan(pd.Timestamp("2020-03-18"), pd.Timestamp("2020-06-08"),  alpha=0.25, color="orange")

for period, color in p_colors.items():
    sub = monthly[monthly["period"] == period]
    ax2.plot(sub["date"], sub["closure_rate"], color=color,
             linewidth=2 if period == "Stress" else 1.5,
             alpha=0.9, marker="o", markersize=3)

ax2.axvline(pd.Timestamp("2021-01-01"), color="gray", linestyle=":", linewidth=1)
ax2.axvline(pd.Timestamp("2022-01-01"), color="gray", linestyle=":", linewidth=1)

ax2.text(pd.Timestamp("2015-06-01"), 72, "Training", color="#2196F3", fontsize=9, fontweight="bold", ha="center")
ax2.text(pd.Timestamp("2021-07-01"), 72, "Primary",  color="#4CAF50", fontsize=9, fontweight="bold", ha="center")
ax2.text(pd.Timestamp("2022-07-01"), 72, "Stress",   color="#F44336", fontsize=9, fontweight="bold", ha="center")
ax2.text(pd.Timestamp("2020-04-20"), 55, "COVID", color="orange", fontsize=7.5, ha="center", rotation=90)

ax2.set_title("Monthly Zero-Sales Rate Over Time", fontsize=10, fontweight="bold", pad=10)
ax2.set_xlabel("Year", fontsize=9)
ax2.set_ylabel("% days with zero bread sales", fontsize=9)
ax2.set_ylim(0, 80)
ax2.grid(axis="y", alpha=0.3)
ax2.tick_params(axis="x", labelsize=8)
ax2.spines[["top", "right"]].set_visible(False)

# ── Panel 3: DOW closure rate by period ───────────────────────────────────────
ax3 = axes[2]
x     = np.arange(7)
width = 0.28

for i, period in enumerate(periods):
    bars = ax3.bar(x + (i - 1) * width, dow_rates[period], width,
                   label=period, color=p_colors[period], alpha=0.85,
                   edgecolor="white", linewidth=0.5)

ax3.set_title("Zero-Sales Rate by Day of Week\n(per period)", fontsize=10, fontweight="bold", pad=10)
ax3.set_xlabel("Day of week", fontsize=9)
ax3.set_ylabel("% days with zero bread sales", fontsize=9)
ax3.set_xticks(x)
ax3.set_xticklabels(dow_names, fontsize=9)
ax3.legend(fontsize=8)
ax3.grid(axis="y", alpha=0.3)
ax3.spines[["top", "right"]].set_visible(False)
ax3.set_ylim(0, 85)

# Annotate stress bars
for i, rate in enumerate(dow_rates["Stress"]):
    if rate > 5:
        ax3.text(x[i] + width, rate + 1, f"{rate:.0f}%",
                 ha="center", va="bottom", fontsize=7, color="#F44336", fontweight="bold")

# ── Final layout ───────────────────────────────────────────────────────────────
fig.suptitle("Evolution of Store Closure Pattern: Training → Primary (2021) → Stress (2022–2023)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PNG}")
plt.close()
