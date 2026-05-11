"""
METR-confound illustration: even when the true per-model Wright exponent is
sublinear (b = 0.79), the observed across-model trajectory can have an apparent
log-log slope of ~2.5 when tooling drifts between evaluations.

The setup mirrors Vlad's R script (analyses/scaling_function_panels.R in the
metr-wright-law project): three top panels sweep a, b, c independently in
the function y = a * x^b + c. The bottom panel shows 10 models, each with
its own (a_t, c_t) but the same b = 0.79, evaluated at increasing x_t. The
trajectory across those points looks much steeper than any individual curve.

If each model t is one year, the observed trajectory's apparent doubling time
in y-space is computed and printed.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib import gridspec
from matplotlib.cm import viridis

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Optional Montserrat font; falls back to default
for f in ['/tmp/Montserrat-Regular.ttf', '/tmp/Montserrat-Bold.ttf', '/tmp/Montserrat-Medium.ttf']:
    if os.path.exists(f):
        fm.fontManager.addfont(f)
font_family = 'Montserrat' if any('Montserrat' in n.name for n in fm.fontManager.ttflist) else 'DejaVu Sans'
mpl.rcParams.update({
    'axes.spines.right': False, 'axes.spines.top': False,
    'font.family': font_family, 'font.weight': 'regular',
    'axes.titleweight': 'regular',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

# ============================================================
# Parameters (match the R script)
# ============================================================
a0, b0, c0 = 1.0, 0.79, 1.0

def scale_fn(x, a, b, c):
    return a * x**b + c

x_dense = np.linspace(1e-3, 100, 300)  # avoid x=0 for the log-log panel; tiny offset

# ============================================================
# Top row — three sweep panels
# ============================================================
a_vals = np.linspace(1, 101, 10)
b_vals = np.linspace(0, 2, 10)
c_vals = np.linspace(1, 101, 10)

fig = plt.figure(figsize=(16, 14))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.8], hspace=0.55, wspace=0.30)

# --- Panel a: sweep a, b=0.79, c=1 ---
ax_a = fig.add_subplot(gs[0, 0])
for i, a in enumerate(a_vals):
    color = viridis(i / (len(a_vals) - 1))
    y = scale_fn(x_dense, a, b0, c0)
    ax_a.plot(x_dense, y, color=color, lw=1.5)
    ax_a.annotate(f'{int(round(y[-1])):,}', xy=(x_dense[-1], y[-1]),
                  xytext=(4, 0), textcoords='offset points',
                  fontsize=9, va='center', ha='left')
ax_a.set_xlim(0, 120)
ax_a.set_title(f'Sweep a (b = {b0}, c = {c0})', fontsize=13)
ax_a.set_xlabel('x'); ax_a.set_ylabel('y')

# --- Panel b: sweep b, a=1, c=1 ---
ax_b = fig.add_subplot(gs[0, 1])
for i, b in enumerate(b_vals):
    color = viridis(i / (len(b_vals) - 1))
    y = scale_fn(x_dense, a0, b, c0)
    ax_b.plot(x_dense, y, color=color, lw=1.5)
y_top = scale_fn(x_dense[-1], a0, b_vals[-1], c0)
ax_b.annotate(f'{int(round(y_top)):,}', xy=(x_dense[-1], y_top),
              xytext=(4, 0), textcoords='offset points',
              fontsize=9, va='center', ha='left')
ax_b.set_xlim(0, 120)
ax_b.set_title(f'Sweep b (a = {a0}, c = {c0})', fontsize=13)
ax_b.set_xlabel('x'); ax_b.set_ylabel('y')

# --- Panel c: sweep c, a=1, b=0.79 ---
ax_c = fig.add_subplot(gs[0, 2])
for i, c in enumerate(c_vals):
    color = viridis(i / (len(c_vals) - 1))
    y = scale_fn(x_dense, a0, b0, c)
    ax_c.plot(x_dense, y, color=color, lw=1.5)
    ax_c.annotate(f'{int(round(y[-1])):,}', xy=(x_dense[-1], y[-1]),
                  xytext=(4, 0), textcoords='offset points',
                  fontsize=9, va='center', ha='left')
ax_c.set_xlim(0, 120)
ax_c.set_title(f'Sweep c (a = {a0}, b = {b0})', fontsize=13)
ax_c.set_xlabel('x'); ax_c.set_ylabel('y')

# ============================================================
# Bottom row — moving-target demonstration
# ============================================================
n_models = 10
t_seq = np.arange(1, n_models + 1)
a_t = np.linspace(1, 101, n_models)
c_t = np.linspace(1, 101, n_models)
x_t = np.linspace(10, 100, n_models)
y_t = scale_fn(x_t, a_t, b0, c_t)

# Apparent power-law fit through observed trajectory
log_x = np.log10(x_t)
log_y = np.log10(y_t)
apparent_b, apparent_intercept = np.polyfit(log_x, log_y, 1)

# Apparent doubling time if each successive model is 6 months apart
MONTHS_PER_MODEL = 6
n_doublings = np.log2(y_t[-1] / y_t[0])
months_elapsed = (t_seq[-1] - t_seq[0]) * MONTHS_PER_MODEL
doubling_time_months = months_elapsed / n_doublings

xx = np.linspace(1, 100, 400)

ax_demo = fig.add_subplot(gs[1, :])
for t, a, c in zip(t_seq, a_t, c_t):
    color = viridis((t - 1) / (n_models - 1))
    y = scale_fn(xx, a, b0, c)
    ax_demo.plot(xx, y, color=color, lw=1.2, alpha=0.75)
    ax_demo.annotate(f'a = {a:.1f}, c = {c:.1f}',
                     xy=(xx[-1], y[-1]),
                     xytext=(6, 0), textcoords='offset points',
                     fontsize=9, va='center', ha='left', color='#333')

ax_demo.plot(x_t, y_t, color='black', lw=1.8, linestyle='--', zorder=5)
ax_demo.scatter(x_t, y_t, color='black', s=44, zorder=6)

ax_demo.set_xlim(0, 130)
ax_demo.set_xlabel('x', fontsize=12)
ax_demo.set_ylabel('y', fontsize=12)

# Title and subtitle stacked above the plot (no overlap)
title_text = f"Moving target: a and c grow each model (b held at {b0})"
subtitle_text = (
    f"Each colored curve is one model's true scaling. Black dashed = observed trajectory across models.\n"
    f"Apparent log–log slope ≈ {apparent_b:.2f}, even though each underlying curve has b = {b0}.\n"
    f"If successive models are 6 months apart, the trajectory looks like exponential growth doubling every "
    f"{doubling_time_months:.1f} months — though every individual model is on a sublinear curve."
)
ax_demo.text(0, 1.18, title_text, transform=ax_demo.transAxes,
             fontsize=14, fontweight='bold', va='top', ha='left', color='#222')
ax_demo.text(0, 1.10, subtitle_text, transform=ax_demo.transAxes,
             fontsize=10.5, va='top', ha='left', color='#555')

# Suptitle: equation + the key conceptual note
fig.text(0.5, 0.985, r'$y = a \cdot x^{b} + c$',
         fontsize=16, fontweight='bold', ha='center', va='top')
fig.text(0.5, 0.955,
         "Of the three parameters, only b determines whether returns are diminishing (b < 1), "
         "constant (b = 1), or accelerating (b > 1).\nSweeps of a or c change the level of observed "
         "values but not the curvature.",
         fontsize=11, ha='center', va='top', color='#444')

out_path = os.path.join(FIG_DIR, 'metr_confound_illustration.png')
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f'saved: {out_path}')

print()
print('=== Moving-target summary ===')
print(f'True per-model Wright exponent (b):          {b0}')
print(f'Apparent log-log slope of trajectory:        {apparent_b:.3f}')
print(f'Inflation factor:                            {apparent_b/b0:.2f}×')
print(f'If successive models are 6 months apart:')
print(f'  y_1 = {y_t[0]:.1f}, y_10 = {y_t[-1]:.1f}')
print(f'  Apparent doubling time:                    {doubling_time_months:.1f} months')
