"""
선형 회귀 채널 시각화.

최근 300봉의 BTC 4h 차트에 채널을 그려 확인.
실행 후 reports/reg_channel_chart.png 로 저장.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from core.indicators import reg_channel, sma

# ── 데이터 로드 ───────────────────────────────────────────────────────
df = pd.read_parquet("data/ohlcv_full_BTCUSDT_4h.parquet")
df = df.tail(600).copy()  # 최근 600봉 (약 100일)

center, upper, lower = reg_channel(df, window=100)
df["reg_center"] = center
df["reg_upper"]  = upper
df["reg_lower"]  = lower
df["sma200"]     = sma(df["close"], 200)

# 최근 300봉만 플롯
plot_df = df.tail(300).copy()

# ── 시각화 ────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                gridspec_kw={"height_ratios": [3, 1]},
                                sharex=True)
fig.suptitle("BTC/USDT 4h — Linear Regression Channel", fontsize=14, fontweight="bold")

# 캔들 차트 (간략히 라인으로)
ax1.plot(plot_df.index, plot_df["close"], color="#222222", linewidth=1.0, label="Close", zorder=3)

# 채널 영역
ax1.fill_between(plot_df.index, plot_df["reg_lower"], plot_df["reg_upper"],
                 alpha=0.12, color="#2196F3", label="Channel band")
ax1.plot(plot_df.index, plot_df["reg_center"], color="#2196F3",
         linewidth=1.5, linestyle="--", label="Regression center", zorder=2)
ax1.plot(plot_df.index, plot_df["reg_upper"], color="#F44336",
         linewidth=1.2, linestyle="-", label="Upper channel (resistance)", zorder=2)
ax1.plot(plot_df.index, plot_df["reg_lower"], color="#4CAF50",
         linewidth=1.2, linestyle="-", label="Lower channel (support)", zorder=2)

# SMA200
ax1.plot(plot_df.index, plot_df["sma200"], color="#FF9800",
         linewidth=1.0, linestyle=":", label="SMA200", zorder=2, alpha=0.7)

# 하단 채널 근처 구간 강조 (눌림목 진입 후보)
near_lower = plot_df["close"] <= plot_df["reg_lower"] * 1.01
ax1.scatter(plot_df.index[near_lower], plot_df["close"][near_lower],
            color="#4CAF50", s=30, zorder=5, label="Near lower (entry zone)")

ax1.set_ylabel("Price (USDT)", fontsize=11)
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.3)

# 채널 폭 (relative) — 하단 패널
channel_width = (plot_df["reg_upper"] - plot_df["reg_lower"]) / plot_df["reg_center"] * 100
ax2.fill_between(plot_df.index, 0, channel_width, alpha=0.4, color="#9C27B0")
ax2.plot(plot_df.index, channel_width, color="#9C27B0", linewidth=1.0)
ax2.set_ylabel("Channel Width (%)", fontsize=10)
ax2.set_xlabel("Date", fontsize=10)
ax2.grid(True, alpha=0.3)

# 기울기 양/음 배경 색
slope = plot_df["reg_center"].diff(5)
for i in range(len(plot_df) - 1):
    color = "#E8F5E9" if slope.iloc[i] >= 0 else "#FFEBEE"
    ax1.axvspan(plot_df.index[i], plot_df.index[i + 1], alpha=0.08, color=color, linewidth=0)

plt.tight_layout()
import os
os.makedirs("reports", exist_ok=True)
out = "reports/reg_channel_chart.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"저장 완료: {out}")

# 채널 통계 요약
print("\n[채널 현황 — 최근 10봉]")
summary = plot_df[["close", "reg_lower", "reg_center", "reg_upper"]].tail(10)
summary["pos_in_channel"] = (
    (summary["close"] - summary["reg_lower"]) /
    (summary["reg_upper"] - summary["reg_lower"] + 1e-12) * 100
).round(1)
summary.columns = ["close", "lower", "center", "upper", "pos(%)"]
summary = summary.round(0)
print(summary.to_string())
print("\n  pos(%) 0 = 하단 채널, 100 = 상단 채널, 50 = 중심선")
