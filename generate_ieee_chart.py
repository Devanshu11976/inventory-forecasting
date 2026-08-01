import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import os

# Configure Times New Roman font for IEEE publication style
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams['mathtext.fontset'] = 'stix'

# Data from Table II
models = ['Moving Average\n(7-day MA)', 'ARIMA\n(2,1,2)', 'XGBoost\n(Proposed)']
mae = [2209.85, 2232.67, 663.05]
rmse = [3036.87, 2981.35, 1148.99]
mape = [21.72, 21.99, 9.35]
smape = [48.36, 48.99, 35.62]

# Colors: Monochrome grayscale palette (IEEE standard)
# Black, Medium Gray, Light Gray with hatch patterns
colors = ['#1a1a1a', '#666666', '#cccccc']
hatches = ['///', '\\\\', '..']

# Create 2-panel figure for IEEE two-column or full-width column
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8), dpi=300)

x = np.arange(len(models))
width = 0.28

# Panel 1: Absolute Error Metrics (MAE & RMSE)
rects1 = ax1.bar(x - width/2, mae, width, label='MAE', color='#222222', edgecolor='black', hatch='///')
rects2 = ax1.bar(x + width/2, rmse, width, label='RMSE', color='#888888', edgecolor='black', hatch='..')

ax1.set_ylabel('Error Value (Units)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
ax1.set_title('(a) Absolute Error Metrics (MAE & RMSE)', fontsize=10, fontweight='bold', fontfamily='Times New Roman', pad=8)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=9, fontfamily='Times New Roman')
ax1.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=8)
ax1.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
ax1.set_ylim(0, 3600)

# Add value labels above bars for Panel 1
for bar in rects1:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=7.5, fontfamily='Times New Roman', fontweight='bold')

for bar in rects2:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=7.5, fontfamily='Times New Roman', fontweight='bold')

# Panel 2: Percentage Error Metrics (MAPE & sMAPE)
rects3 = ax2.bar(x - width/2, mape, width, label='MAPE (%)', color='#444444', edgecolor='black', hatch='\\\\')
rects4 = ax2.bar(x + width/2, smape, width, label='sMAPE (%)', color='#aaaaaa', edgecolor='black', hatch='xx')

ax2.set_ylabel('Error Percentage (%)', fontsize=10, fontweight='bold', fontfamily='Times New Roman')
ax2.set_title('(b) Percentage Error Metrics (MAPE & sMAPE)', fontsize=10, fontweight='bold', fontfamily='Times New Roman', pad=8)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=9, fontfamily='Times New Roman')
ax2.legend(loc='upper right', frameon=True, edgecolor='black', fontsize=8)
ax2.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
ax2.set_ylim(0, 58)

# Add value labels above bars for Panel 2
for bar in rects3:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=7.5, fontfamily='Times New Roman', fontweight='bold')

for bar in rects4:
    height = bar.get_height()
    ax2.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=7.5, fontfamily='Times New Roman', fontweight='bold')

plt.tight_layout()

# Save outputs to workspace and artifacts
out_w = r'c:\Users\devan\Desktop\research paper\figure3_model_comparison_ieee.png'
out_a = r'C:\Users\devan\.gemini\antigravity-ide\brain\9b15e131-adb3-4ca1-8d22-537f6da449b2\figure3_model_comparison_ieee.png'

plt.savefig(out_w, dpi=300, bbox_inches='tight')
plt.savefig(out_a, dpi=300, bbox_inches='tight')
plt.close()

print('Saved monochrome IEEE model comparison chart to:', out_w)
